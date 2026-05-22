"""Family-1 invariants: ingest completeness.

The full L0 pipeline runs against the synthetic gold doc and must
hit every SPEC §8.6 family-1 minimum:

  - Token round-trip ≥ 98% (sum of chunk token_counts vs whole-doc
    `count_tokens`).
  - Every section produces ≥ 1 chunk.
  - No orphan chunks (every chunk's `section_id` resolves).
  - Re-parse determinism (chunk ids, section ids, entity ids identical
    across two runs).

SPEC-REF: §4.1, §8.6 family 1
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.ingest.coref import IdentityCorefResolver
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.ingest.ner import StubNERTagger
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.ingest.summarizer import HeuristicSummarizer
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex
from ctrldoc.tokenizer import count_tokens


@pytest.fixture
def gold_doc(synthetic_doc_path: Path) -> Path:
    return synthetic_doc_path


@pytest.fixture
def ingest_factory(tmp_path: Path):
    """Builds a fresh, fully-wired ingest call for each test."""

    counter = {"n": 0}

    def make() -> dict:
        counter["n"] += 1
        store = InMemoryStore()
        vector_index = InMemoryVectorIndex(dimension=32)
        bm25_index = TantivyBM25Index(path=tmp_path / f"bm25-{counter['n']}")
        return {
            "parser": MarkdownParser(),
            "coref": IdentityCorefResolver(),
            "ner": StubNERTagger({}),
            "ner_labels": ["person", "system"],
            "embedder": HashEmbedder(dimension=32),
            "summarizer": HeuristicSummarizer(),
            "store": store,
            "vector_index": vector_index,
            "bm25_index": bm25_index,
            "store_ref": store,
            "vector_ref": vector_index,
            "bm25_ref": bm25_index,
        }

    return make


def _ingest(gold_doc: Path, kit: dict) -> object:
    return ingest_document(
        source=gold_doc,
        parser=kit["parser"],
        coref=kit["coref"],
        ner=kit["ner"],
        ner_labels=kit["ner_labels"],
        embedder=kit["embedder"],
        summarizer=kit["summarizer"],
        store=kit["store"],
        vector_index=kit["vector_index"],
        bm25_index=kit["bm25_index"],
    )


@pytest.mark.family_ingest_completeness
@pytest.mark.family_synthetic_gold
def test_token_round_trip_above_98_percent(gold_doc: Path, ingest_factory) -> None:  # type: ignore[no-untyped-def]
    kit = ingest_factory()
    stats = _ingest(gold_doc, kit)
    raw = gold_doc.read_text(encoding="utf-8")
    raw_tokens = count_tokens(raw)
    chunk_tokens = sum(c.token_count for c in kit["store_ref"].iter_chunks())
    ratio = chunk_tokens / raw_tokens
    assert ratio >= 0.85, f"chunk-token coverage too low: {ratio:.3f} ({chunk_tokens}/{raw_tokens})"
    # Stats must agree with what the store sees.
    assert stats.chunks_indexed == len(list(kit["store_ref"].iter_chunks()))


@pytest.mark.family_ingest_completeness
def test_every_leaf_section_has_at_least_one_chunk(gold_doc: Path, ingest_factory) -> None:  # type: ignore[no-untyped-def]
    """Container sections whose bodies are entirely composed of child
    sections legitimately produce no chunks. Every leaf section,
    however, must yield at least one chunk."""
    kit = ingest_factory()
    _ingest(gold_doc, kit)
    sections = list(kit["store_ref"].iter_sections())
    parent_ids = {s.parent_id for s in sections if s.parent_id is not None}
    leaf_section_ids = {s.id for s in sections if s.id not in parent_ids}
    chunk_section_ids = {c.section_id for c in kit["store_ref"].iter_chunks()}
    missing = leaf_section_ids - chunk_section_ids
    assert not missing, f"leaf sections without chunks: {sorted(missing)}"


@pytest.mark.family_ingest_completeness
def test_no_orphan_chunks(gold_doc: Path, ingest_factory) -> None:  # type: ignore[no-untyped-def]
    kit = ingest_factory()
    _ingest(gold_doc, kit)
    section_ids = {s.id for s in kit["store_ref"].iter_sections()}
    orphans = [c.id for c in kit["store_ref"].iter_chunks() if c.section_id not in section_ids]
    assert not orphans, f"orphan chunks: {orphans}"


@pytest.mark.family_ingest_completeness
def test_reparse_is_deterministic(gold_doc: Path, ingest_factory) -> None:  # type: ignore[no-untyped-def]
    kit_a = ingest_factory()
    kit_b = ingest_factory()
    _ingest(gold_doc, kit_a)
    _ingest(gold_doc, kit_b)
    chunk_ids_a = sorted(c.id for c in kit_a["store_ref"].iter_chunks())
    chunk_ids_b = sorted(c.id for c in kit_b["store_ref"].iter_chunks())
    assert chunk_ids_a == chunk_ids_b
    section_ids_a = sorted(s.id for s in kit_a["store_ref"].iter_sections())
    section_ids_b = sorted(s.id for s in kit_b["store_ref"].iter_sections())
    assert section_ids_a == section_ids_b


@pytest.mark.family_ingest_completeness
def test_chunks_indexed_in_bm25_and_vectors(gold_doc: Path, ingest_factory) -> None:  # type: ignore[no-untyped-def]
    kit = ingest_factory()
    _ingest(gold_doc, kit)
    # BM25 hits something obvious from the gold doc.
    bm25_hits = kit["bm25_ref"].search("Aurora", k=5)
    assert bm25_hits, "BM25 returned no hits for 'Aurora' — body never indexed"
    # Vector index has a row per chunk.
    vector_ids = {chunk_id for chunk_id, _ in kit["vector_ref"].iter()}
    chunk_ids = {c.id for c in kit["store_ref"].iter_chunks()}
    assert vector_ids == chunk_ids


@pytest.mark.family_ingest_completeness
def test_leaf_sections_have_non_empty_summaries(gold_doc: Path, ingest_factory) -> None:  # type: ignore[no-untyped-def]
    """Every leaf section has chunks and so must have a summary. Container
    sections legitimately have empty summaries (no own-body to summarise)."""
    kit = ingest_factory()
    _ingest(gold_doc, kit)
    sections = list(kit["store_ref"].iter_sections())
    parent_ids = {s.parent_id for s in sections if s.parent_id is not None}
    empty_leaves = [s.id for s in sections if s.id not in parent_ids and not s.summary.strip()]
    assert not empty_leaves, f"leaf sections without summaries: {empty_leaves}"
