"""Contract tests for the retrieval DSL.

`Plan` is a list of `PlanStep`s that the planner LLM emits and the
executor (S-041) runs. Three step variants per SPEC §4.3:
  - `Search(query, view, k)`
  - `Expand(section_id)`
  - `Neighbors(entity_id, hops)`

The DSL has two forms — structured JSON (the LLM's tool-output) and
a Python-call textual form. Both round-trip through `Plan`.

SPEC-REF: §4.3 (retrieval DSL)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.retrieval.dsl import (
    Expand,
    Neighbors,
    Plan,
    Search,
    parse_plan_dsl,
    render_plan_dsl,
)

# --- Search ---


def test_search_required_fields() -> None:
    step = Search(query="q", view="dense", k=8)
    assert step.op == "search"
    assert step.query == "q"
    assert step.view == "dense"
    assert step.k == 8


@pytest.mark.parametrize("view", ["dense", "lexical", "entity"])
def test_search_accepts_all_spec_views(view: str) -> None:
    Search(query="q", view=view, k=4)  # type: ignore[arg-type]


def test_search_unknown_view_rejected() -> None:
    with pytest.raises(ValidationError):
        Search(query="q", view="semantic", k=4)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_k", [0, -1])
def test_search_non_positive_k_rejected(bad_k: int) -> None:
    with pytest.raises(ValidationError):
        Search(query="q", view="dense", k=bad_k)


def test_search_round_trip() -> None:
    step = Search(query="cosmos", view="lexical", k=3)
    payload = step.model_dump()
    assert payload["op"] == "search"
    assert Search.model_validate(payload) == step


# --- Expand ---


def test_expand_round_trip() -> None:
    step = Expand(section_id="sec/intro")
    payload = step.model_dump()
    assert payload["op"] == "expand"
    assert Expand.model_validate(payload) == step


def test_expand_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Expand(section_id="sec/intro", bogus="no")  # type: ignore[call-arg]


# --- Neighbors ---


def test_neighbors_default_hops() -> None:
    step = Neighbors(entity_id="ent/person/sam-altman")
    assert step.hops == 1


@pytest.mark.parametrize("bad_hops", [0, -1])
def test_neighbors_non_positive_hops_rejected(bad_hops: int) -> None:
    with pytest.raises(ValidationError):
        Neighbors(entity_id="ent/e", hops=bad_hops)


def test_neighbors_round_trip() -> None:
    step = Neighbors(entity_id="ent/sys/aurora", hops=2)
    assert Neighbors.model_validate(step.model_dump()) == step


# --- Plan ---


def test_empty_plan() -> None:
    plan = Plan(steps=[])
    assert plan.steps == []


def test_plan_carries_heterogeneous_steps() -> None:
    plan = Plan(
        steps=[
            Search(query="q", view="dense", k=8),
            Expand(section_id="sec/a"),
            Neighbors(entity_id="ent/x"),
        ]
    )
    assert [type(s).__name__ for s in plan.steps] == ["Search", "Expand", "Neighbors"]


def test_plan_round_trip_via_discriminated_union() -> None:
    original = Plan(
        steps=[
            Search(query="q", view="entity", k=4),
            Expand(section_id="sec/a"),
            Neighbors(entity_id="ent/x", hops=2),
        ]
    )
    payload = original.model_dump()
    restored = Plan.model_validate(payload)
    assert restored == original


def test_plan_unknown_op_rejected() -> None:
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [{"op": "imaginary", "x": 1}]})


# --- DSL text round-trip ---


def test_render_dsl_for_each_step_type() -> None:
    plan = Plan(
        steps=[
            Search(query="cosmos", view="dense", k=8),
            Expand(section_id="sec/intro"),
            Neighbors(entity_id="ent/sys/aurora", hops=2),
        ]
    )
    rendered = render_plan_dsl(plan)
    assert "search(" in rendered
    assert "expand(" in rendered
    assert "neighbors(" in rendered
    assert "dense" in rendered
    assert "sec/intro" in rendered
    assert "hops=2" in rendered


def test_parse_dsl_round_trip() -> None:
    text = (
        'search(query="cosmos", view=dense, k=8)\n'
        'expand(section_id="sec/intro")\n'
        'neighbors(entity_id="ent/sys/aurora", hops=2)\n'
    )
    plan = parse_plan_dsl(text)
    assert len(plan.steps) == 3
    assert isinstance(plan.steps[0], Search)
    assert isinstance(plan.steps[1], Expand)
    assert isinstance(plan.steps[2], Neighbors)
    assert plan.steps[0].query == "cosmos"
    assert plan.steps[1].section_id == "sec/intro"
    assert plan.steps[2].hops == 2


def test_dsl_text_round_trip_idempotent() -> None:
    original = Plan(
        steps=[
            Search(query="hello", view="lexical", k=4),
            Neighbors(entity_id="ent/e", hops=1),
        ]
    )
    rendered = render_plan_dsl(original)
    reparsed = parse_plan_dsl(rendered)
    assert reparsed == original


def test_dsl_empty_input_returns_empty_plan() -> None:
    assert parse_plan_dsl("").steps == []
    assert parse_plan_dsl("   \n\n  ").steps == []


def test_dsl_unknown_op_raises() -> None:
    with pytest.raises(ValueError):
        parse_plan_dsl("imaginary(foo=bar)\n")
