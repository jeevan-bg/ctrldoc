"""Contract tests for Reciprocal Rank Fusion.

RRF combines multiple ranked lists of ids into one ranking by
summing `1 / (k + rank)` across the lists. The fusion helper also
accepts `StepResult`s straight from the executor so the planner can
chain executor → RRF without manual unpacking.

SPEC-REF: §4.3 (retrieval fusion)
"""

from __future__ import annotations

import math

import pytest

from ctrldoc.retrieval.executor import StepResult
from ctrldoc.retrieval.fusion import (
    DEFAULT_RRF_K,
    fuse_step_results,
    reciprocal_rank_fusion,
)

# --- raw RRF ---


def test_default_k_matches_published_value() -> None:
    assert DEFAULT_RRF_K == 60


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order() -> None:
    out = reciprocal_rank_fusion([["a", "b", "c"]])
    ids = [item_id for item_id, _ in out]
    assert ids == ["a", "b", "c"]


def test_intersecting_items_rank_higher_than_single_appearances() -> None:
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    out = reciprocal_rank_fusion([a, b])
    ids = [item_id for item_id, _ in out]
    # x and y appear in both lists; w and z only once. The intersecting
    # pair must rank above the singletons.
    assert ids.index("x") < ids.index("z")
    assert ids.index("x") < ids.index("w")
    assert ids.index("y") < ids.index("z")
    assert ids.index("y") < ids.index("w")


def test_known_scoring_formula() -> None:
    # rank=1 in one list, rank=2 in another.
    out = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    score_by_id = dict(out)
    expected = 1 / (60 + 1) + 1 / (60 + 2)
    assert math.isclose(score_by_id["a"], expected, abs_tol=1e-12)
    assert math.isclose(score_by_id["b"], expected, abs_tol=1e-12)


def test_k_parameter_changes_scores() -> None:
    a = reciprocal_rank_fusion([["x"]], k=60)
    b = reciprocal_rank_fusion([["x"]], k=10)
    assert a[0][1] != b[0][1]


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_fusion_is_deterministic() -> None:
    lists = [["a", "b", "c"], ["c", "a"]]
    assert reciprocal_rank_fusion(lists) == reciprocal_rank_fusion(lists)


def test_score_descending_order() -> None:
    out = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c"]])
    scores = [score for _, score in out]
    assert scores == sorted(scores, reverse=True)


# --- StepResult adapter ---


def test_fuse_step_results_pulls_chunk_ids() -> None:
    results = [
        StepResult(op="search", chunk_ids=["c1", "c2", "c3"]),
        StepResult(op="search", chunk_ids=["c2", "c1"]),
        StepResult(op="expand", chunk_ids=["c1"]),
    ]
    fused = fuse_step_results(results)
    ids = [chunk_id for chunk_id, _ in fused]
    assert ids[0] == "c1"  # appears in all three


def test_fuse_step_results_skips_neighbors_results() -> None:
    results = [
        StepResult(op="search", chunk_ids=["c1", "c2"]),
        StepResult(op="neighbors", entity_ids=["ent/x"]),
    ]
    fused = fuse_step_results(results)
    assert {chunk_id for chunk_id, _ in fused} == {"c1", "c2"}


def test_fuse_step_results_empty_returns_empty() -> None:
    assert fuse_step_results([]) == []
    assert fuse_step_results([StepResult(op="search")]) == []
