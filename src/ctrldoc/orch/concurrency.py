"""Semaphore-bounded async fan-out for sub-task concurrency.

`bounded_gather` is the only concurrency primitive the orchestrator
exposes. Callers compose their own coroutines (typically wrapping
sync `TaskClient.call` via `asyncio.to_thread`) and pass them in
along with a per-backend cap. The cap is enforced *at the moment*
of execution — never more than `max_concurrent` in flight at once,
even with hundreds of submitted coros.

Per SPEC §4.7 the per-backend defaults are:

  - Anthropic API: 8
  - Ollama 7B inference: 1
  - Ollama embedder / reranker: 2 each

The values are exposed as module-level constants so callers can
either pin to the spec defaults or read them from `ctrldoc.toml`'s
`[concurrency]` block (S-004).

SPEC-REF: §4.7 (concurrency)
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_ANTHROPIC_CONCURRENT = 8
DEFAULT_OLLAMA_7B_CONCURRENT = 1
DEFAULT_EMBEDDER_CONCURRENT = 2
DEFAULT_RERANKER_CONCURRENT = 2


async def bounded_gather(
    coros: list[Coroutine[Any, Any, T]],
    *,
    max_concurrent: int,
) -> list[T]:
    """Run all coros concurrently with at most `max_concurrent` in flight.

    Results are returned in input order. On the first exception, the
    semaphore is released and `asyncio.gather` cancels in-flight peers
    before propagating the failure to the caller.
    """
    if max_concurrent <= 0:
        raise ValueError(f"max_concurrent must be positive, got {max_concurrent}")
    if not coros:
        return []

    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(coro: Coroutine[Any, Any, T]) -> T:
        async with sem:
            return await coro

    return await asyncio.gather(*(_bounded(c) for c in coros))


__all__ = [
    "DEFAULT_ANTHROPIC_CONCURRENT",
    "DEFAULT_EMBEDDER_CONCURRENT",
    "DEFAULT_OLLAMA_7B_CONCURRENT",
    "DEFAULT_RERANKER_CONCURRENT",
    "bounded_gather",
]
