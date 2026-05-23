"""Retrieval-planner DSL.

The planner LLM emits a `Plan` — a list of `PlanStep`s the executor
runs against the multi-view index. Three step variants per SPEC §4.3:

  - `Search(query, view, k)` — top-k against the dense / lexical /
    entity view.
  - `Expand(section_id)` — pull every chunk under a section.
  - `Neighbors(entity_id, hops)` — entities reachable through the
    co-mention graph.

The DSL has two forms: structured JSON (the model's tool output) and
a Python-call textual form. Both round-trip through the same `Plan`.

SPEC-REF: §4.3
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


SearchView = Literal["dense", "lexical", "entity"]


class Search(_Strict):
    op: Literal["search"] = "search"
    query: str
    view: SearchView
    k: PositiveInt


class Expand(_Strict):
    op: Literal["expand"] = "expand"
    section_id: str


class Neighbors(_Strict):
    op: Literal["neighbors"] = "neighbors"
    entity_id: str
    hops: PositiveInt = 1


PlanStep = Annotated[Search | Expand | Neighbors, Field(discriminator="op")]


class Plan(_Strict):
    """A flat list of steps the executor runs in order."""

    steps: list[PlanStep]


# --- DSL text round-trip ---


_CALL_RE = re.compile(r"^([a-z_]+)\((.*)\)$")
_ARG_SPLIT_RE = re.compile(r",\s*(?=[a-z_]+=)")


def render_plan_dsl(plan: Plan) -> str:
    """Render a `Plan` in the Python-call textual form."""
    lines: list[str] = []
    for step in plan.steps:
        if isinstance(step, Search):
            lines.append(f'search(query="{_escape(step.query)}", view={step.view}, k={step.k})')
        elif isinstance(step, Expand):
            lines.append(f'expand(section_id="{_escape(step.section_id)}")')
        elif isinstance(step, Neighbors):
            lines.append(f'neighbors(entity_id="{_escape(step.entity_id)}", hops={step.hops})')
    return "\n".join(lines) + ("\n" if lines else "")


def parse_plan_dsl(text: str) -> Plan:
    """Parse the textual DSL form into a `Plan`."""
    steps: list[PlanStep] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _CALL_RE.match(line)
        if match is None:
            raise ValueError(f"unparseable DSL line: {line!r}")
        op = match.group(1)
        args = _parse_args(match.group(2))
        if op == "search":
            steps.append(
                Search(
                    query=str(args["query"]),
                    view=args["view"],  # type: ignore[arg-type]
                    k=_as_int(args["k"]),
                )
            )
        elif op == "expand":
            steps.append(Expand(section_id=str(args["section_id"])))
        elif op == "neighbors":
            steps.append(
                Neighbors(
                    entity_id=str(args["entity_id"]),
                    hops=_as_int(args.get("hops", 1)),
                )
            )
        else:
            raise ValueError(f"unknown DSL op: {op!r}")
    return Plan(steps=steps)


def _parse_args(body: str) -> dict[str, object]:
    if not body.strip():
        return {}
    out: dict[str, object] = {}
    for fragment in _ARG_SPLIT_RE.split(body):
        key, _, value = fragment.partition("=")
        out[key.strip()] = _parse_value(value.strip())
    return out


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int-compatible, got {type(value).__name__}")


def _parse_value(value: str) -> object:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "Expand",
    "Neighbors",
    "Plan",
    "PlanStep",
    "Search",
    "SearchView",
    "parse_plan_dsl",
    "render_plan_dsl",
]
