"""Contract tests for the BM25 lexical index (Tantivy-backed).

The `BM25Index` protocol covers lexical retrieval over chunks. The
Tantivy backend persists to a directory and exposes BM25-ranked
top-k search. Idempotency by `chunk_id` lets the ingest layer
re-index a chunk without leaving a duplicate copy behind.

SPEC-REF: §4.2 (BM25 lexical), §4.3 (retrieval)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.store.bm25 import BM25Index, TantivyBM25Index


def _idx(tmp_path: Path, name: str = "bm25") -> TantivyBM25Index:
    return TantivyBM25Index(path=tmp_path / name)


def test_satisfies_protocol(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        assert isinstance(index, BM25Index)


def test_empty_index_search_returns_empty(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        assert index.search("anything", k=5) == []


def test_add_and_search_finds_document(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "hello world cosmos")
        hits = index.search("hello", k=5)
        assert [h[0] for h in hits] == ["c1"]
        assert hits[0][1] > 0.0


def test_search_ranks_by_relevance(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c-hit", "hello hello hello cosmos")
        index.add("c-mid", "hello cosmos")
        index.add("c-miss", "completely unrelated text")
        hits = index.search("hello", k=3)
        chunk_ids = [h[0] for h in hits]
        assert chunk_ids[0] == "c-hit"
        assert "c-miss" not in chunk_ids


def test_term_match_is_case_insensitive_default(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "Pillar 1 — Stateless Tasks")
        assert [h[0] for h in index.search("stateless", k=1)] == ["c1"]
        assert [h[0] for h in index.search("STATELESS", k=1)] == ["c1"]


def test_search_no_match_returns_empty(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "hello world")
        assert index.search("nonexistent_token_xyz", k=5) == []


def test_re_adding_chunk_id_replaces_text(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "original keyword foxtrot")
        assert [h[0] for h in index.search("foxtrot", k=5)] == ["c1"]
        index.add("c1", "replaced different content")
        # Old text must no longer match.
        assert index.search("foxtrot", k=5) == []
        # New text must match.
        assert [h[0] for h in index.search("replaced", k=5)] == ["c1"]


def test_persistence_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "bm25"
    with TantivyBM25Index(path=path) as index:
        index.add("c1", "persistence works")
        index.add("c2", "another doc")
    with TantivyBM25Index(path=path) as index:
        assert [h[0] for h in index.search("persistence", k=5)] == ["c1"]
        assert {h[0] for h in index.search("doc", k=5)} == {"c2"}


def test_k_zero_returns_empty(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "hello")
        assert index.search("hello", k=0) == []


def test_k_negative_rejected(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "hello")
        with pytest.raises(ValueError):
            index.search("hello", k=-1)


def test_k_larger_than_index_returns_all_matching(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "hello world")
        index.add("c2", "hello cosmos")
        hits = index.search("hello", k=100)
        assert {h[0] for h in hits} == {"c1", "c2"}


def test_empty_query_returns_empty(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        index.add("c1", "hello world")
        assert index.search("", k=5) == []


def test_batched_add_persists_all(tmp_path: Path) -> None:
    with _idx(tmp_path) as index:
        for i in range(10):
            index.add(f"c{i}", f"document number {i} with keyword pineapple")
        hits = index.search("pineapple", k=20)
        assert len(hits) == 10
