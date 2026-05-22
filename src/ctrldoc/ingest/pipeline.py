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

from ctrldoc.ingest.chunker import DEFAULT_MAX_TOKENS, chunk_sections
from ctrldoc.ingest.coref import CorefResolver, resolve_sections
from ctrldoc.ingest.embedder import Embedder
from ctrldoc.ingest.ner import EntityMention, NERTagger, canonicalize
from ctrldoc.ingest.parser import Parser
from ctrldoc.ingest.summarizer import Summarizer, summarize_sections
from ctrldoc.models import Chunk, Entity
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index
from ctrldoc.store.vectors import VectorIndex


class IngestStats(BaseModel):
    """Per-run summary of the ingest pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections_parsed: NonNegativeInt
    chunks_indexed: NonNegativeInt
    entities_indexed: NonNegativeInt
    embedded_tokens: NonNegativeInt


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
) -> IngestStats:
    """Run the full L0 pipeline against `source` and persist into `store`."""
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

    return IngestStats(
        sections_parsed=len(sections),
        chunks_indexed=len(chunks),
        entities_indexed=len(entities_by_id),
        embedded_tokens=sum(c.token_count for c in chunks),
    )


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


__all__ = ["IngestStats", "ingest_document"]
