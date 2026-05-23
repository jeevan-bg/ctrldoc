"""Reciprocal Rank Fusion over the executor's per-step outputs.

`reciprocal_rank_fusion` is the raw primitive that combines an
arbitrary number of ranked id lists. `fuse_step_results` adapts the
executor's `StepResult`s onto it (pulling `chunk_ids` from each
search / expand result and skipping neighbour traversals).

SPEC-REF: §4.3 (retrieval fusion)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from ctrldoc.retrieval.executor import StepResult

DEFAULT_RRF_K: Final[int] = 60


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Combine ranked id lists via `score = sum(1 / (k + rank))`.

    Returns `(id, score)` pairs in descending score order. Items that
    appear in more lists, or at better positions, rank higher.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    insertion_order: dict[str, int] = {}
    next_order = 0
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            if item_id not in scores:
                insertion_order[item_id] = next_order
                next_order += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1], insertion_order[kv[0]]),
    )


def fuse_step_results(
    results: Iterable[StepResult],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Apply RRF over the `chunk_ids` lists carried by each `StepResult`.

    `Neighbors` results (which populate `entity_ids` rather than
    `chunk_ids`) are skipped — they are inputs to follow-up steps,
    not directly ranked alongside chunk hits.
    """
    chunk_lists = [r.chunk_ids for r in results if r.chunk_ids]
    return reciprocal_rank_fusion(chunk_lists, k=k)


__all__ = ["DEFAULT_RRF_K", "fuse_step_results", "reciprocal_rank_fusion"]
