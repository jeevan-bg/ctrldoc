"""End-to-end L0 ingest pipeline.

`ingest_document` glues every L0 component together: parse → coref →
chunk → embed → NER + canonicalise → summarise → persist. Components
are injected so each layer can be swapped (e.g. a stub NER tagger
in tests, the real GLiNER backend in production).

Returned `IngestStats` is the at-a-glance summary every caller can
log or assert against.

SPEC-REF: §4.1 (ingest pipeline), §8.6 family 1 (completeness)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from ctrldoc.eval.claim_extraction import ClaimExtractor
from ctrldoc.extract.claim_persistence import claim_from_tuple
from ctrldoc.extract.within_doc_edges import WithinDocEdgeInferer
from ctrldoc.ingest.chunker import DEFAULT_MAX_TOKENS, chunk_sections
from ctrldoc.ingest.coref import CorefResolver, resolve_sections
from ctrldoc.ingest.embedder import Embedder
from ctrldoc.ingest.ner import EntityMention, NERTagger, canonicalize
from ctrldoc.ingest.parser import ParsedSection, Parser
from ctrldoc.ingest.summarizer import Summarizer, summarize_sections
from ctrldoc.models import Chunk, Entity
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index
from ctrldoc.store.vectors import VectorIndex
from ctrldoc.versioning import content_hash


class IngestStats(BaseModel):
    """Per-run summary of the ingest pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections_parsed: NonNegativeInt
    chunks_indexed: NonNegativeInt
    entities_indexed: NonNegativeInt
    embedded_tokens: NonNegativeInt
    claims_extracted: NonNegativeInt = 0
    typed_edges_inferred: NonNegativeInt = 0


class IncrementalStats(BaseModel):
    """Per-run summary of an incremental re-ingest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections_added: NonNegativeInt
    sections_changed: NonNegativeInt
    sections_removed: NonNegativeInt
    sections_unchanged: NonNegativeInt
    chunks_added: NonNegativeInt
    chunks_removed: NonNegativeInt


def ingest_document(
    *,
    source: str | Path,
    parser: Parser,
    coref: CorefResolver,
    ner: NERTagger,
    ner_labels: list[str],
    embedder: Embedder,
    summarizer: Summarizer,
    store: Store,
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    max_chunk_tokens: int = DEFAULT_MAX_TOKENS,
    claim_extractor: ClaimExtractor | None = None,
    edge_inferer: WithinDocEdgeInferer | None = None,
    doc_id: str | None = None,
) -> IngestStats:
    """Run the full L0 pipeline against `source` and persist into `store`.

    Optional `claim_extractor` plugs the §6.2 universal-tuple
    extractor into the pipeline: per chunk emit SVO tuples, adapt
    each into a persisted `Claim` (content-hashed id), write through
    `store.append_claim`. Without an extractor the pipeline runs the
    pre-§6.2 path and zero claims land in the store.

    Optional `edge_inferer` runs after claim persistence: a
    `WithinDocEdgeInferer` reads the just-persisted claims for the
    doc, runs the §6.3 Galois floor (and the optional §6.5 Tier-2
    NLI overlay) over every pair, and writes every emitted
    `TypedEdge` through `store.append_typed_edge`. Without the
    inferer no typed-edges land. Requires `claim_extractor` to also
    be set — without persisted claims there is nothing to pair.

    `doc_id` is the parent document's logical id used as the
    content-hash prefix for claim identity. Defaults to the source
    path's stem when omitted.
    """
    parsed = parser.parse(source)
    parsed = resolve_sections(parsed, coref)

    chunks, sections = chunk_sections(parsed, max_tokens=max_chunk_tokens)
    chunks = [chunk.model_copy(update={"embedding_id": f"emb/{chunk.id}"}) for chunk in chunks]

    vectors = embedder.embed_batch([c.text for c in chunks])

    entities_by_id: dict[str, Entity] = {}
    for chunk in chunks:
        mentions = ner.tag(chunk.text, labels=ner_labels)
        for entity in canonicalize(mentions, chunk_id=chunk.id):
            _merge_entity(entities_by_id, entity)

    chunks_by_section: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_section.setdefault(chunk.section_id, []).append(chunk)
    sections = summarize_sections(
        sections,
        body_for=lambda s: " ".join(c.text for c in chunks_by_section.get(s.id, [])),
        summarizer=summarizer,
    )

    store.add_chunks(chunks)
    store.add_sections(sections)
    store.add_entities(entities_by_id.values())
    for chunk, vector in zip(chunks, vectors, strict=True):
        vector_index.add(chunk.id, vector)
        bm25_index.add(chunk.id, chunk.text)

    claims_extracted = 0
    typed_edges_inferred = 0
    if claim_extractor is not None:
        effective_doc_id = doc_id if doc_id is not None else Path(source).stem
        claims_extracted = _extract_and_persist_claims(
            chunks=chunks,
            extractor=claim_extractor,
            store=store,
            doc_id=effective_doc_id,
        )
        if edge_inferer is not None:
            typed_edges_inferred = _infer_and_persist_typed_edges(
                store=store,
                inferer=edge_inferer,
                doc_id=effective_doc_id,
            )

    return IngestStats(
        sections_parsed=len(sections),
        chunks_indexed=len(chunks),
        entities_indexed=len(entities_by_id),
        embedded_tokens=sum(c.token_count for c in chunks),
        claims_extracted=claims_extracted,
        typed_edges_inferred=typed_edges_inferred,
    )


def ingest_document_incremental(
    *,
    source: str | Path,
    parser: Parser,
    coref: CorefResolver,
    ner: NERTagger,
    ner_labels: list[str],
    embedder: Embedder,
    summarizer: Summarizer,
    store: Store,
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    max_chunk_tokens: int = DEFAULT_MAX_TOKENS,
) -> IncrementalStats:
    """Re-ingest `source` and only re-index sections whose body changed.

    Section identity uses `ParsedSection.id` (deterministic from the
    parser). A section is "changed" when its content hash differs from
    the hash of the concatenated chunks stored under that id. Sections
    present only in the new parse are added; sections present only in
    the store are removed (chunks + section row).
    """
    parsed_sections = resolve_sections(parser.parse(source), coref)
    parsed_by_id: dict[str, ParsedSection] = {s.id: s for s in parsed_sections}
    new_hashes = {sid: content_hash(s.text) for sid, s in parsed_by_id.items()}

    stored_section_ids = {s.id for s in store.iter_sections()}
    stored_chunks_by_section: dict[str, list[Chunk]] = {}
    for chunk in store.iter_chunks():
        stored_chunks_by_section.setdefault(chunk.section_id, []).append(chunk)
    stored_hashes = {
        section_id: content_hash(" ".join(c.text for c in chunks))
        for section_id, chunks in stored_chunks_by_section.items()
    }

    added_ids = set(parsed_by_id) - stored_section_ids
    removed_ids = stored_section_ids - set(parsed_by_id)
    common_ids = set(parsed_by_id) & stored_section_ids
    changed_ids = {sid for sid in common_ids if new_hashes[sid] != stored_hashes.get(sid)}
    unchanged_ids = common_ids - changed_ids

    chunks_removed = 0

    for section_id in sorted(removed_ids):
        removed_chunks = store.delete_chunks_for_section(section_id)
        for chunk_id in removed_chunks:
            vector_index.remove(chunk_id)
            bm25_index.remove(chunk_id)
        chunks_removed += len(removed_chunks)
        store.delete_section(section_id)

    for section_id in sorted(changed_ids):
        removed_chunks = store.delete_chunks_for_section(section_id)
        for chunk_id in removed_chunks:
            vector_index.remove(chunk_id)
            bm25_index.remove(chunk_id)
        chunks_removed += len(removed_chunks)

    to_process_ids = changed_ids | added_ids
    to_process = [parsed_by_id[sid] for sid in parsed_by_id if sid in to_process_ids]
    chunks_added = 0

    if to_process:
        chunks, sections = chunk_sections(to_process, max_tokens=max_chunk_tokens)
        chunks = [c.model_copy(update={"embedding_id": f"emb/{c.id}"}) for c in chunks]
        vectors = embedder.embed_batch([c.text for c in chunks])

        entities_by_id: dict[str, Entity] = {}
        for chunk in chunks:
            mentions = ner.tag(chunk.text, labels=ner_labels)
            for entity in canonicalize(mentions, chunk_id=chunk.id):
                _merge_entity(entities_by_id, entity)

        chunks_by_section_local: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            chunks_by_section_local.setdefault(chunk.section_id, []).append(chunk)
        sections = summarize_sections(
            sections,
            body_for=lambda s: " ".join(c.text for c in chunks_by_section_local.get(s.id, [])),
            summarizer=summarizer,
        )

        store.add_chunks(chunks)
        store.add_sections(sections)
        store.add_entities(entities_by_id.values())
        for chunk, vector in zip(chunks, vectors, strict=True):
            vector_index.add(chunk.id, vector)
            bm25_index.add(chunk.id, chunk.text)
        chunks_added = len(chunks)

    return IncrementalStats(
        sections_added=len(added_ids),
        sections_changed=len(changed_ids),
        sections_removed=len(removed_ids),
        sections_unchanged=len(unchanged_ids),
        chunks_added=chunks_added,
        chunks_removed=chunks_removed,
    )


def _infer_and_persist_typed_edges(
    *,
    store: Store,
    inferer: WithinDocEdgeInferer,
    doc_id: str,
) -> int:
    """Run the within-doc inferer over the just-persisted claims for this doc.

    The inferer iterates the store's `iter_claims_for_doc(doc_id)` so
    the input is always a stable snapshot, then persists every emitted
    `TypedEdge`. Idempotency is enforced at the storage layer (PRIMARY
    KEY on `(src_id, dst_id, type)`), so a re-ingest of the same doc
    never produces duplicate rows.
    """
    claims = list(store.iter_claims_for_doc(doc_id))
    if len(claims) < 2:
        return 0
    edges = inferer.infer(claims)
    for edge in edges:
        store.append_typed_edge(edge)
    return len(edges)


def _extract_and_persist_claims(
    *,
    chunks: list[Chunk],
    extractor: ClaimExtractor,
    store: Store,
    doc_id: str,
) -> int:
    """Run the extractor over every chunk and persist resulting claims.

    Iteration order is the chunk order returned by `chunk_sections`
    (deterministic by section / char_start). Tuples are deduped
    within a chunk by the extractor itself; the content-hashed
    `claim_id` then guarantees idempotency across re-runs.
    """
    persisted = 0
    for chunk in chunks:
        tuples = extractor.extract(chunk.text)
        for tuple_ in tuples:
            claim = claim_from_tuple(doc_id=doc_id, chunk=chunk, tuple_=tuple_)
            store.append_claim(claim)
            persisted += 1
    return persisted


def _merge_entity(into: dict[str, Entity], entity: Entity) -> None:
    existing = into.get(entity.id)
    if existing is None:
        into[entity.id] = entity
        return
    aliases = sorted({*existing.aliases, *entity.aliases})
    chunk_ids = sorted({*existing.mention_chunk_ids, *entity.mention_chunk_ids})
    into[entity.id] = existing.model_copy(
        update={"aliases": aliases, "mention_chunk_ids": chunk_ids}
    )


def _all_mentions(by_chunk: dict[str, list[EntityMention]]) -> Iterable[EntityMention]:
    for mentions in by_chunk.values():
        yield from mentions


__all__ = [
    "IncrementalStats",
    "IngestStats",
    "ingest_document",
    "ingest_document_incremental",
]
