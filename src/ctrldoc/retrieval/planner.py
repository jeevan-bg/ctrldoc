"""Retrieval planner — protocol and heuristic reference.

`Planner.plan(prefix, query)` turns the cacheable prefix
(`{system_prompt, doc_skeleton, entity_glossary}`) and a user query
into a `Plan` the executor runs. The MVP ships:

  - `HeuristicPlanner` — dense + lexical search at a fixed `k`, no
    network or model dependency.

A separate module (`planner_anthropic.py`) wires the Anthropic-backed
planner with prompt-cache markers on the prefix.

SPEC-REF: §4.3 (planner), §3.1 (cacheable prefix)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.retrieval.dsl import Plan, Search


@runtime_checkable
class Planner(Protocol):
    """Cacheable prefix + query → executable plan."""

    def plan(self, prefix: CacheablePrefix, query: str) -> Plan: ...


class HeuristicPlanner:
    """Default plan: one dense search and one lexical search at `default_k`.

    Useful for unit tests, low-cost runs, and as a baseline the
    production planner must beat in evals.
    """

    def __init__(self, *, default_k: int = 8) -> None:
        if default_k <= 0:
            raise ValueError("default_k must be positive")
        self._default_k = default_k

    def plan(self, prefix: CacheablePrefix, query: str) -> Plan:
        if not query.strip():
            return Plan(steps=[])
        return Plan(
            steps=[
                Search(query=query, view="dense", k=self._default_k),
                Search(query=query, view="lexical", k=self._default_k),
            ]
        )


__all__ = ["HeuristicPlanner", "Planner"]
