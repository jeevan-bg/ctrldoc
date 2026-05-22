"""Contract tests for the dense-vector index.

The `VectorIndex` protocol is the seam between L1 storage and the
retrieval planner's dense-similarity search. The in-memory reference
implementation is the behavioural oracle for the persistent backend.

SPEC-REF: §4.2 (dense vectors), §4.3 (retrieval)
"""

from __future__ import annotations

import math

import pytest

from ctrldoc.store.vectors import (
    InMemoryVectorIndex,
    VectorDimensionMismatchError,
    VectorIndex,
)


def test_satisfies_protocol() -> None:
    index = InMemoryVectorIndex(dimension=3)
    assert isinstance(index, VectorIndex)


def test_dimension_is_pinned() -> None:
    index = InMemoryVectorIndex(dimension=4)
    assert index.dimension == 4


def test_zero_dimension_rejected() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorIndex(dimension=0)
    with pytest.raises(ValueError):
        InMemoryVectorIndex(dimension=-1)


def test_add_and_iter() -> None:
    index = InMemoryVectorIndex(dimension=3)
    index.add("c1", [1.0, 0.0, 0.0])
    index.add("c2", [0.0, 1.0, 0.0])
    chunk_ids = {chunk_id for chunk_id, _ in index.iter()}
    assert chunk_ids == {"c1", "c2"}


def test_add_dim_mismatch_raises() -> None:
    index = InMemoryVectorIndex(dimension=3)
    with pytest.raises(VectorDimensionMismatchError):
        index.add("c1", [1.0, 0.0])


def test_search_dim_mismatch_raises() -> None:
    index = InMemoryVectorIndex(dimension=3)
    index.add("c1", [1.0, 0.0, 0.0])
    with pytest.raises(VectorDimensionMismatchError):
        index.search([1.0, 0.0], k=1)


def test_search_empty_returns_empty() -> None:
    assert InMemoryVectorIndex(dimension=3).search([1.0, 0.0, 0.0], k=5) == []


def test_search_returns_top_k_by_cosine() -> None:
    index = InMemoryVectorIndex(dimension=3)
    index.add("c-x", [1.0, 0.0, 0.0])
    index.add("c-y", [0.0, 1.0, 0.0])
    index.add("c-z", [0.0, 0.0, 1.0])
    hits = index.search([0.9, 0.1, 0.0], k=2)
    assert [chunk_id for chunk_id, _ in hits] == ["c-x", "c-y"]
    assert hits[0][1] > hits[1][1]


def test_identity_vector_scores_one() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("c1", [3.0, 4.0])
    [(chunk_id, score)] = index.search([3.0, 4.0], k=1)
    assert chunk_id == "c1"
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_cosine_score_in_unit_interval_of_magnitude() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("c1", [1.0, 0.0])
    [(_, score_same)] = index.search([1.0, 0.0], k=1)
    [(_, score_orth)] = index.search([0.0, 1.0], k=1)
    [(_, score_opp)] = index.search([-1.0, 0.0], k=1)
    assert math.isclose(score_same, 1.0, abs_tol=1e-9)
    assert math.isclose(score_orth, 0.0, abs_tol=1e-9)
    assert math.isclose(score_opp, -1.0, abs_tol=1e-9)


def test_zero_vector_in_index_does_not_crash_search() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("c-zero", [0.0, 0.0])
    index.add("c-real", [1.0, 0.0])
    hits = index.search([1.0, 0.0], k=2)
    chunk_ids = [chunk_id for chunk_id, _ in hits]
    # The non-zero vector must come first; zero-vector is allowed but its
    # similarity is zero (we define 0/0 as 0).
    assert chunk_ids[0] == "c-real"


def test_zero_query_returns_zero_scores() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("c1", [1.0, 0.0])
    hits = index.search([0.0, 0.0], k=1)
    assert math.isclose(hits[0][1], 0.0, abs_tol=1e-9)


def test_add_idempotent_by_chunk_id() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("c1", [1.0, 0.0])
    index.add("c1", [0.0, 1.0])
    hits = index.search([0.0, 1.0], k=1)
    assert hits[0][0] == "c1"
    assert math.isclose(hits[0][1], 1.0, abs_tol=1e-9)
    assert len(list(index.iter())) == 1


def test_k_larger_than_index_returns_all() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("c1", [1.0, 0.0])
    index.add("c2", [0.0, 1.0])
    hits = index.search([1.0, 0.0], k=10)
    assert len(hits) == 2


def test_k_zero_returns_empty() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("c1", [1.0, 0.0])
    assert index.search([1.0, 0.0], k=0) == []


def test_k_negative_rejected() -> None:
    index = InMemoryVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.search([1.0, 0.0], k=-1)
