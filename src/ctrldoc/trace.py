"""JSONL trace logger for LLM calls.

Every LLM call writes one line to `traces/{run_id}.jsonl`. The file
is append-only and machine-grep-able so cost, latency, token usage,
and cache-hit rates are reconstructable for any past run.

SPEC-REF: §4.7 (observability)
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt


class TraceRecord(BaseModel):
    """One LLM call. Field set matches SPEC §4.7."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task_id: str
    playbook: str
    model: str
    prompt_hash: str
    response_hash: str
    tokens_in: NonNegativeInt
    tokens_out: NonNegativeInt
    cost_usd: NonNegativeFloat
    latency_ms: NonNegativeInt
    cache_hit: bool
    error: str | None


class TraceLogger:
    """Append-only JSONL writer for `TraceRecord`s.

    Thread-safe within a single process; file writes are serialised
    behind a `threading.Lock`. A separate `TraceLogger` per run id
    is the expected usage.
    """

    def __init__(self, *, traces_dir: Path, run_id: str) -> None:
        self._dir = traces_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_id}.jsonl"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: TraceRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def read_trace(path: Path) -> Iterator[TraceRecord]:
    """Yield typed `TraceRecord`s from a JSONL trace file."""
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            yield TraceRecord.model_validate_json(line)


__all__ = ["TraceLogger", "TraceRecord", "read_trace"]
