"""Integration tests for the `sqlite-vec` VectorIndex backend.

These tests need (1) `sqlite-vec` installed and (2) a Python
interpreter built with loadable-extension support. They skip
cleanly when either is missing.

SPEC-REF: §4.2 (dense vectors), §4.3 (retrieval)
"""

from __future__ import annotations

import math
import sqlite3

import pytest

pytest.importorskip("sqlite_vec", reason="sqlite-vec optional; install ctrldoc[index] to run")

from ctrldoc.store.vectors import InMemoryVectorIndex, VectorDimensionMismatchError, VectorIndex
from ctrldoc.store.vectors_sqlite_vec import SqliteVecVectorIndex


def _loadable_extensions_supported() -> bool:
    try:
        sqlite3.connect(":memory:").enable_load_extension(True)
        return True
    except (sqlite3.NotSupportedError, AttributeError):
        return False


pytestmark = pytest.mark.skipif(
    not _loadable_extensions_supported(),
    reason="Python built without --enable-loadable-sqlite-extensions",
)


def test_satisfies_protocol() -> None:
    assert isinstance(SqliteVecVectorIndex(dimension=3), VectorIndex)


def test_dimension_is_pinned() -> None:
    assert SqliteVecVectorIndex(dimension=4).dimension == 4


def test_zero_dimension_rejected() -> None:
    with pytest.raises(ValueError):
        SqliteVecVectorIndex(dimension=0)
    with pytest.raises(ValueError):
        SqliteVecVectorIndex(dimension=-1)


def test_add_and_iter_round_trip() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    index.add("c1", [1.0, 0.0, 0.0])
    index.add("c2", [0.0, 1.0, 0.0])
    pairs = {chunk_id: tuple(vec) for chunk_id, vec in index.iter()}
    assert set(pairs.keys()) == {"c1", "c2"}
    assert all(
        math.isclose(x, y, abs_tol=1e-5) for x, y in zip(pairs["c1"], [1.0, 0.0, 0.0], strict=True)
    )


def test_add_dim_mismatch_raises() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    with pytest.raises(VectorDimensionMismatchError):
        index.add("c1", [1.0, 0.0])


def test_search_dim_mismatch_raises() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    index.add("c1", [1.0, 0.0, 0.0])
    with pytest.raises(VectorDimensionMismatchError):
        index.search([1.0, 0.0], k=1)


def test_search_orders_by_cosine_similarity() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    index.add("c-near", [1.0, 0.0, 0.0])
    index.add("c-mid", [0.7071, 0.7071, 0.0])
    index.add("c-far", [0.0, 1.0, 0.0])
    hits = index.search([1.0, 0.0, 0.0], k=3)
    assert [chunk_id for chunk_id, _ in hits] == ["c-near", "c-mid", "c-far"]
    # Cosine similarity scores: 1.0, 0.7071, 0.0 (within float tolerance).
    scores = [score for _, score in hits]
    assert math.isclose(scores[0], 1.0, abs_tol=1e-3)
    assert math.isclose(scores[1], 0.7071, abs_tol=1e-3)
    assert math.isclose(scores[2], 0.0, abs_tol=1e-3)


def test_search_truncates_to_k() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    for i, v in enumerate([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        index.add(f"c{i}", v)
    hits = index.search([1.0, 0.0, 0.0], k=2)
    assert len(hits) == 2


def test_search_k_zero_returns_empty() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    index.add("c1", [1.0, 0.0, 0.0])
    assert index.search([1.0, 0.0, 0.0], k=0) == []


def test_search_k_negative_rejected() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    with pytest.raises(ValueError):
        index.search([1.0, 0.0, 0.0], k=-1)


def test_search_empty_index_returns_empty() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    assert index.search([1.0, 0.0, 0.0], k=5) == []


def test_remove_unknown_id_is_noop() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    index.remove("never-added")  # must not raise


def test_remove_evicts_entry() -> None:
    index = SqliteVecVectorIndex(dimension=3)
    index.add("c1", [1.0, 0.0, 0.0])
    index.add("c2", [0.0, 1.0, 0.0])
    index.remove("c1")
    chunk_ids = {chunk_id for chunk_id, _ in index.iter()}
    assert chunk_ids == {"c2"}
    hits = index.search([1.0, 0.0, 0.0], k=5)
    assert [chunk_id for chunk_id, _ in hits] == ["c2"]


def test_add_idempotent_on_same_id() -> None:
    """Re-adding the same chunk_id replaces the embedding."""
    index = SqliteVecVectorIndex(dimension=3)
    index.add("c1", [1.0, 0.0, 0.0])
    index.add("c1", [0.0, 1.0, 0.0])
    chunk_ids = [chunk_id for chunk_id, _ in index.iter()]
    assert chunk_ids == ["c1"]
    # After replacement, search for [0, 1, 0] should yield perfect similarity.
    hits = index.search([0.0, 1.0, 0.0], k=1)
    assert hits[0][0] == "c1"
    assert math.isclose(hits[0][1], 1.0, abs_tol=1e-3)


def test_matches_in_memory_reference() -> None:
    """Behavioural equivalence with `InMemoryVectorIndex` on the same inputs."""
    rng = [
        ("a", [0.1, 0.9, 0.0]),
        ("b", [0.5, 0.5, 0.7071]),
        ("c", [0.9, 0.1, 0.0]),
        ("d", [-0.5, -0.5, 0.0]),
    ]
    ref = InMemoryVectorIndex(dimension=3)
    impl = SqliteVecVectorIndex(dimension=3)
    for cid, v in rng:
        ref.add(cid, v)
        impl.add(cid, v)
    query = [0.3, 0.3, 0.9]
    ref_ids = [cid for cid, _ in ref.search(query, k=4)]
    impl_ids = [cid for cid, _ in impl.search(query, k=4)]
    assert ref_ids == impl_ids
