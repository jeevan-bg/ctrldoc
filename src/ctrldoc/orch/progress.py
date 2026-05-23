"""Streaming progress events.

The orchestrator emits one event per sub-task transition so a
foreground process bar or callback can show liveness within the
§4.7 "never block silently longer than 5s" budget. The event schema
is fixed (`ProgressEvent`), the emitter is a small protocol with
two reference implementations (stdout, callable), and the
`ProgressTracker` coordinates counters, ETA derivation, and
thread-safety so async fan-out (`bounded_gather` from S-064) can
fire events from multiple worker threads without garbling counts.

SPEC-REF: §4.7 (streaming)
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

ProgressEventType = Literal["task_started", "task_completed", "task_failed"]


class ProgressEvent(BaseModel):
    """One transition in the run's lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: ProgressEventType
    task_id: str
    progress: str
    eta_seconds: float | None = None
    cost_so_far_usd: float = 0.0
    error: str | None = None


@runtime_checkable
class ProgressEmitter(Protocol):
    """Anything that consumes a `ProgressEvent`."""

    def emit(self, event: ProgressEvent) -> None: ...


class CallableProgressEmitter:
    """Forwards each event to a user-supplied callable."""

    def __init__(self, fn: Callable[[ProgressEvent], None]) -> None:
        self._fn = fn

    def emit(self, event: ProgressEvent) -> None:
        self._fn(event)


class StdoutProgressEmitter:
    """Writes each event as a single JSON line to stdout."""

    def emit(self, event: ProgressEvent) -> None:
        sys.stdout.write(event.model_dump_json())
        sys.stdout.write("\n")
        sys.stdout.flush()


class ProgressTracker:
    """Counts started/completed/failed sub-tasks and emits events.

    Thread-safe: a lock guards the counters and elapsed-time read so
    `start()`, `complete()`, and `fail()` can be called from worker
    threads driven by `bounded_gather` without losing increments.

    ETA is derived from wall-clock elapsed since the first `start()`
    divided by `completed + failed` — failures count as throughput
    so the estimate reflects the run's actual progress, not its
    success rate.
    """

    def __init__(
        self,
        *,
        total: int,
        emitter: ProgressEmitter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        self._total = total
        self._emitter = emitter
        self._clock = clock
        self._lock = threading.Lock()
        self._completed = 0
        self._cost_so_far = 0.0
        self._started_at: float | None = None

    def start(self, task_id: str) -> None:
        with self._lock:
            if self._started_at is None:
                self._started_at = self._clock()
            event = ProgressEvent(
                event="task_started",
                task_id=task_id,
                progress=f"{self._completed}/{self._total}",
                eta_seconds=self._eta_locked(),
                cost_so_far_usd=self._cost_so_far,
            )
        self._emitter.emit(event)

    def complete(self, task_id: str, *, cost_usd: float = 0.0) -> None:
        self._finish(task_id, event_type="task_completed", cost_usd=cost_usd, error=None)

    def fail(self, task_id: str, *, error: str, cost_usd: float = 0.0) -> None:
        self._finish(task_id, event_type="task_failed", cost_usd=cost_usd, error=error)

    def _finish(
        self,
        task_id: str,
        *,
        event_type: ProgressEventType,
        cost_usd: float,
        error: str | None,
    ) -> None:
        with self._lock:
            self._completed += 1
            self._cost_so_far += cost_usd
            event = ProgressEvent(
                event=event_type,
                task_id=task_id,
                progress=f"{self._completed}/{self._total}",
                eta_seconds=self._eta_locked(),
                cost_so_far_usd=self._cost_so_far,
                error=error,
            )
        self._emitter.emit(event)

    def _eta_locked(self) -> float | None:
        if self._completed <= 0 or self._started_at is None:
            return None
        elapsed = self._clock() - self._started_at
        per_task = elapsed / self._completed
        return per_task * (self._total - self._completed)


__all__ = [
    "CallableProgressEmitter",
    "ProgressEmitter",
    "ProgressEvent",
    "ProgressEventType",
    "ProgressTracker",
    "StdoutProgressEmitter",
]
