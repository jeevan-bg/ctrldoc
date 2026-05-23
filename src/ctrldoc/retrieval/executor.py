"""Retrieval-DSL executor.

Walks a `Plan` step-by-step against an injected `Store`,
`VectorIndex`, `BM25Index`, and `Embedder`. Each step produces one
typed `StepResult` (chunk_ids and/or entity_ids depending on the op).
Fusion across views (Reciprocal Rank Fusion) is a separate module;
the executor here only runs the individual lookups.

SPEC-REF: §4.3 (retrieval executor)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.ingest.embedder import Embedder
from ctrldoc.retrieval.dsl import Expand, Neighbors, Plan, PlanStep, Search
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index
from ctrldoc.store.vectors import VectorIndex


class StepResult(BaseModel):
    """Output of one PlanStep.

    `chunk_ids` and `scores` cover Search + Expand; Neighbors fills
    `entity_ids` and leaves the chunk fields empty. The `op` tag matches
    the step's op so caller code can pattern-match without re-reading
    the plan.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: str
    chunk_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


class PlanExecutor:
    """Run a `Plan` against the multi-view index."""

    def __init__(
        self,
        *,
        store: Store,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        embedder: Embedder,
    ) -> None:
        self._store = store
        self._vector_index = vector_index
        self._bm25_index = bm25_index
        self._embedder = embedder

    def execute(self, plan: Plan) -> list[StepResult]:
        return [self._run(step) for step in plan.steps]

    def _run(self, step: PlanStep) -> StepResult:
        if isinstance(step, Search):
            return self._run_search(step)
        if isinstance(step, Expand):
            return StepResult(
                op="expand",
                chunk_ids=[
                    c.id for c in self._store.iter_chunks() if c.section_id == step.section_id
                ],
            )
        if isinstance(step, Neighbors):
            return StepResult(op="neighbors", entity_ids=self._bfs(step.entity_id, step.hops))
        raise TypeError(f"unsupported plan step: {type(step).__name__}")

    def _run_search(self, step: Search) -> StepResult:
        if step.view == "dense":
            vector = self._embedder.embed(step.query)
            hits = self._vector_index.search(vector, k=step.k)
            return StepResult(
                op="search",
                chunk_ids=[chunk_id for chunk_id, _ in hits],
                scores=[score for _, score in hits],
            )
        if step.view == "lexical":
            hits = self._bm25_index.search(step.query, k=step.k)
            return StepResult(
                op="search",
                chunk_ids=[chunk_id for chunk_id, _ in hits],
                scores=[score for _, score in hits],
            )
        # view == "entity"
        chunk_ids = self._store.chunks_for_entity(step.query)[: step.k]
        return StepResult(op="search", chunk_ids=chunk_ids)

    def _bfs(self, source_id: str, hops: int) -> list[str]:
        visited: set[str] = set()
        frontier: set[str] = {source_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for entity_id in frontier:
                for neighbor in self._store.entity_neighbors(entity_id):
                    if neighbor not in visited and neighbor != source_id:
                        next_frontier.add(neighbor)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return sorted(visited)


__all__ = ["PlanExecutor", "StepResult"]
