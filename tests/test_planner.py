"""Contract tests for the retrieval planner.

`Planner.plan(prefix, query)` emits a `Plan` that the executor will
run. Two backends ship with the protocol: a heuristic that pairs a
dense and a lexical search (deterministic, no network) and the
Anthropic-backed planner with stub-client tests.

SPEC-REF: §4.3 (retrieval planner)
"""

from __future__ import annotations

import pytest

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.retrieval.dsl import Plan, Search
from ctrldoc.retrieval.planner import HeuristicPlanner, Planner


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are a careful retrieval planner.",
        doc_skeleton="# Section 1\n\nIntro summary.\n",
        entity_glossary="- **ent/sys/aurora** [system]\n",
    )


def test_heuristic_satisfies_protocol() -> None:
    assert isinstance(HeuristicPlanner(), Planner)


def test_heuristic_default_plan_has_two_search_steps() -> None:
    plan = HeuristicPlanner().plan(_prefix(), "What is Aurora?")
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 2
    assert all(isinstance(s, Search) for s in plan.steps)


def test_heuristic_default_views_are_dense_and_lexical() -> None:
    plan = HeuristicPlanner().plan(_prefix(), "any query")
    views = {step.view for step in plan.steps}  # type: ignore[attr-defined]
    assert views == {"dense", "lexical"}


def test_heuristic_default_k_is_configurable() -> None:
    plan = HeuristicPlanner(default_k=4).plan(_prefix(), "any query")
    for step in plan.steps:
        assert step.k == 4  # type: ignore[attr-defined]


def test_heuristic_carries_query_into_each_step() -> None:
    plan = HeuristicPlanner().plan(_prefix(), "what is the consistency model?")
    for step in plan.steps:
        assert step.query == "what is the consistency model?"  # type: ignore[attr-defined]


def test_heuristic_empty_query_returns_empty_plan() -> None:
    plan = HeuristicPlanner().plan(_prefix(), "")
    assert plan.steps == []


def test_heuristic_is_deterministic() -> None:
    p = HeuristicPlanner()
    assert p.plan(_prefix(), "q") == p.plan(_prefix(), "q")


def test_heuristic_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        HeuristicPlanner(default_k=0)
    with pytest.raises(ValueError):
        HeuristicPlanner(default_k=-1)
