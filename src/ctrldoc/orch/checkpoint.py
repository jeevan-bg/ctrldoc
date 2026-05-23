"""Per-sub-task checkpoint store for playbook resumability.

A playbook fan-out writes one structured result per item to
`runs/{run_id}/state.json` immediately after each sub-task completes.
A crash mid-run leaves the prior completions intact; a follow-up
invocation reads them back and only fires the still-pending items.

Writes are atomic: results are serialised to `state.json.tmp` first,
then `os.replace`d over `state.json`. A torn `.tmp` from a previous
crash is ignored on load.

The on-disk format carries a `schema_version` and the originating
`run_id` so accidental cross-run loads fail loud rather than silently
mixing state.

SPEC-REF: §4.7 (resumability), §8.6 family 12
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

CHECKPOINT_SCHEMA_VERSION = "1"


class CheckpointStore:
    """Holds per-item structured results for one run."""

    def __init__(self, *, runs_dir: Path, run_id: str) -> None:
        self._run_id = run_id
        self._dir = runs_dir / run_id
        self._state_path = self._dir / "state.json"
        self._results: dict[str, Any] = {}
        if self._state_path.exists():
            self._load()

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def run_id(self) -> str:
        return self._run_id

    def completed(self) -> dict[str, Any]:
        """Snapshot of saved item-id → result entries."""
        return dict(self._results)

    def pending(self, all_ids: list[str] | tuple[str, ...] | Any) -> list[str]:
        """Filter `all_ids` to items not yet saved, preserving order."""
        return [item_id for item_id in all_ids if item_id not in self._results]

    def save(self, item_id: str, result: BaseModel | dict[str, Any]) -> None:
        """Record a completed sub-task and atomically flush to disk."""
        payload = result.model_dump() if isinstance(result, BaseModel) else result
        self._results[item_id] = payload
        self._flush()

    def _load(self) -> None:
        try:
            data = json.loads(self._state_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"checkpoint file is corrupt at {self._state_path}: {exc}") from exc
        if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version mismatch at {self._state_path}: "
                f"expected {CHECKPOINT_SCHEMA_VERSION!r}, got {data.get('schema_version')!r}"
            )
        if data.get("run_id") != self._run_id:
            raise ValueError(
                f"checkpoint run_id mismatch at {self._state_path}: "
                f"expected {self._run_id!r}, got {data.get('run_id')!r}"
            )
        results = data.get("results", {})
        if not isinstance(results, dict):
            raise ValueError(f"checkpoint results must be an object, got {type(results).__name__}")
        self._results = dict(results)

    def _flush(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".json.tmp")
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self._run_id,
            "results": self._results,
        }
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp_path, self._state_path)


__all__ = ["CHECKPOINT_SCHEMA_VERSION", "CheckpointStore"]
