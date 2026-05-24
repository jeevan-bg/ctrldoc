"""L4 verdict ledger — append-only persistence + replay-determinism gate.

SPEC §6.5 demands every L4 verdict be replayable: a downstream consumer
must be able to re-run an operation with the persisted inputs and
recover the same calibrated confidence within a tight tolerance band.
SPEC §11 surfaces the ledger through the CLI (`ctrldoc ledger {list,
show, replay}`) so an auditor can pull any row and re-prove it.

This module is the substrate. It exposes:

* `LedgerEntry` — the typed shape of one persisted row. Mirrors the §8
  `verdict_ledger` table column-for-column and carries the parsed
  `inputs` / `output` / `model_versions` / `paraphrase_votes`
  payloads (no second JSON decode at the call-site).
* `LedgerAppendRequest` — the typed shape of one *new* row, used by
  the orchestrator paths that ship a verdict.
* `VerdictLedger` — the thin facade: `append`, `get`, `list_entries`,
  `replay`. There is *no* `update` / `delete` / `clear` — the append-
  only contract is enforced at the API boundary (no mutator method) on
  top of the schema-level enforcement (no UPDATE / DELETE SQL emitted).
* `ReplayOutcome` — the typed result of one `replay(...)` call. Carries
  the persisted vs. replayed confidences, the absolute delta, the
  tolerance the call was scored against, and a single `is_deterministic`
  bool that thresholds against `REPLAY_TOLERANCE`.
* `REPLAY_TOLERANCE` — the §6.5 release gate, pinned at 0.02.

The `Replayer` callable is the seam between the ledger and whichever
op shipped the verdict. The ledger does not know how to recompute
`coverage` or `compare`; it hands the persisted `inputs` dict to the
caller-supplied replayer and trusts the caller to dispatch through the
§6.10 tool surface. This keeps the ledger free of every op's import
graph and makes the replay path testable in isolation.

SPEC-REF: §6.5 (replayable verdicts), §11 (CLI surface)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.models import UnitInterval
from ctrldoc.provenance import now_iso
from ctrldoc.store.sqlite import SQLiteStore

# ---------------------------------------------------------------------------
# Release-gate constant
# ---------------------------------------------------------------------------


REPLAY_TOLERANCE: float = 0.02
"""§6.5 release gate: a replay is deterministic when
`|persisted - replayed| <= REPLAY_TOLERANCE`. Surfaced as a module
constant so callers (and tests) score against the same number.
"""

_BOUNDARY_EPSILON: float = 1e-9
"""Numerical slack on the `<= REPLAY_TOLERANCE` comparison so a delta
that lands exactly on the boundary (e.g. via `0.40 + REPLAY_TOLERANCE`,
which evaluates to 0.42000000000000004 in IEEE-754) still passes the
gate. The slack is many orders of magnitude tighter than the
calibrated-confidence reporting precision, so it cannot mask a real
non-determinism event.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LedgerEntryNotFoundError(KeyError):
    """Raised when an id passed to `get` or `replay` does not exist."""


# ---------------------------------------------------------------------------
# Typed shapes
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LedgerAppendRequest(_Strict):
    """The shape an orchestrator passes to `VerdictLedger.append`.

    The `timestamp` field is supplied by the caller (rather than
    auto-stamped here) so the orchestrator can use the same monotonic
    clock for the ledger row, the verdict-output object, and the
    proof-trace step list — making the three lines up at replay time.
    """

    workspace_id: str
    operation: str
    inputs: dict[str, Any]
    output: dict[str, Any]
    calibrated_confidence: UnitInterval
    model_versions: dict[str, str]
    paraphrase_votes: dict[str, int] | None = None
    timestamp: str = Field(default_factory=now_iso)


class LedgerEntry(_Strict):
    """One persisted row from the §8 `verdict_ledger` table.

    `id` is the AUTOINCREMENT primary key — monotonic, unique,
    insertion-ordered. The four `*_json`-on-disk columns are decoded
    here into native dicts so callers never have to parse JSON twice.
    """

    id: int
    workspace_id: str
    operation: str
    inputs: dict[str, Any]
    output: dict[str, Any]
    calibrated_confidence: UnitInterval
    model_versions: dict[str, str]
    paraphrase_votes: dict[str, int] | None
    timestamp: str


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


Replayer = Callable[[dict[str, Any]], float]
"""Replay seam: takes the persisted `inputs` dict, returns a calibrated
confidence the ledger compares against the persisted value.
"""


def is_replay_deterministic(
    persisted: float,
    replayed: float,
    *,
    tolerance: float = REPLAY_TOLERANCE,
) -> bool:
    """Apply the §6.5 ±tolerance gate. Stand-alone for non-ledger callers."""
    return abs(persisted - replayed) <= tolerance + _BOUNDARY_EPSILON


class ReplayOutcome(_Strict):
    """The typed result of one `VerdictLedger.replay(...)` call.

    `delta` is the absolute confidence difference; `is_deterministic`
    is the boolean verdict against the gate. Both are exported so the
    CLI's `ledger replay <id>` output can render the raw distance and
    the pass/fail flag side by side.
    """

    entry_id: int
    operation: str
    persisted_confidence: UnitInterval
    replayed_confidence: UnitInterval
    delta: float
    tolerance: float
    is_deterministic: bool


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class VerdictLedger:
    """Thin facade over the §8 `verdict_ledger` table.

    Methods, all of them:

    * `append(request)` — write one row, return the persisted entry.
    * `get(entry_id)` — fetch one entry; raise `LedgerEntryNotFoundError`
      when absent.
    * `list_entries(workspace_id=None)` — iterate entries in append
      order, optionally filtered to one workspace.
    * `replay(entry_id, replayer)` — re-run the recorded operation
      through a caller-supplied replayer and report the determinism
      delta.

    Notice: no `update`, no `delete`, no `clear`. The append-only
    contract is enforced here (no mutator method exists) and at the
    storage layer (no UPDATE / DELETE SQL is emitted against
    `verdict_ledger`). `SQLiteStore.clear_all` is the destructive
    test-cleanup escape hatch and is intentionally outside this
    facade's surface.
    """

    def __init__(self, *, store: SQLiteStore) -> None:
        self._store = store

    # --- write path ---

    def append(self, request: LedgerAppendRequest) -> LedgerEntry:
        """Persist one verdict; return the entry with its assigned id."""
        row_id = self._store.append_ledger_row(
            workspace_id=request.workspace_id,
            operation=request.operation,
            inputs_json=json.dumps(request.inputs, sort_keys=True),
            output_json=json.dumps(request.output, sort_keys=True),
            calibrated_confidence=float(request.calibrated_confidence),
            model_versions_json=json.dumps(request.model_versions, sort_keys=True),
            paraphrase_votes_json=(
                json.dumps(request.paraphrase_votes, sort_keys=True)
                if request.paraphrase_votes is not None
                else None
            ),
            timestamp=request.timestamp,
        )
        return LedgerEntry(
            id=row_id,
            workspace_id=request.workspace_id,
            operation=request.operation,
            inputs=dict(request.inputs),
            output=dict(request.output),
            calibrated_confidence=request.calibrated_confidence,
            model_versions=dict(request.model_versions),
            paraphrase_votes=(
                dict(request.paraphrase_votes) if request.paraphrase_votes is not None else None
            ),
            timestamp=request.timestamp,
        )

    # --- read path ---

    def get(self, entry_id: int) -> LedgerEntry:
        """Fetch one entry by id; raise `LedgerEntryNotFoundError` if absent."""
        row = self._store.get_ledger_row(entry_id)
        if row is None:
            raise LedgerEntryNotFoundError(f"ledger entry not found: id={entry_id}")
        return _row_to_entry(row)

    def list_entries(self, *, workspace_id: str | None = None) -> list[LedgerEntry]:
        """List entries in append order, optionally filtered by workspace."""
        return [
            _row_to_entry(row) for row in self._store.iter_ledger_rows(workspace_id=workspace_id)
        ]

    # --- replay path ---

    def replay(self, entry_id: int, *, replayer: Replayer) -> ReplayOutcome:
        """Re-run an entry through the supplied replayer; score the drift.

        The replayer is handed the persisted `inputs` dict verbatim
        and must return a calibrated confidence in [0, 1]. The ledger
        scores `|persisted - replayed|` against `REPLAY_TOLERANCE` and
        returns both the raw distance and the pass/fail flag.
        """
        entry = self.get(entry_id)
        replayed = float(replayer(dict(entry.inputs)))
        # Clamp the replayed value into [0, 1] so the Pydantic
        # `UnitInterval` round-trip survives a small numeric overshoot
        # from a replayer that returns 1.0000001. Anything materially
        # outside the unit interval indicates a broken replayer and
        # would be caught at the source.
        replayed_clamped = min(max(replayed, 0.0), 1.0)
        delta = abs(entry.calibrated_confidence - replayed_clamped)
        return ReplayOutcome(
            entry_id=entry.id,
            operation=entry.operation,
            persisted_confidence=entry.calibrated_confidence,
            replayed_confidence=replayed_clamped,
            delta=delta,
            tolerance=REPLAY_TOLERANCE,
            is_deterministic=is_replay_deterministic(entry.calibrated_confidence, replayed_clamped),
        )


# ---------------------------------------------------------------------------
# Row hydration
# ---------------------------------------------------------------------------


def _row_to_entry(row: Any) -> LedgerEntry:
    """Hydrate one `sqlite3.Row` into a typed `LedgerEntry`."""
    paraphrase_raw = row["paraphrase_votes_json"]
    paraphrase_votes: dict[str, int] | None = (
        json.loads(paraphrase_raw) if paraphrase_raw is not None else None
    )
    return LedgerEntry(
        id=int(row["id"]),
        workspace_id=row["workspace_id"],
        operation=row["operation"],
        inputs=dict(json.loads(row["inputs_json"])),
        output=dict(json.loads(row["output_json"])),
        calibrated_confidence=float(row["calibrated_confidence"]),
        model_versions=dict(json.loads(row["model_versions_json"])),
        paraphrase_votes=paraphrase_votes,
        timestamp=row["timestamp"],
    )


__all__ = [
    "REPLAY_TOLERANCE",
    "LedgerAppendRequest",
    "LedgerEntry",
    "LedgerEntryNotFoundError",
    "ReplayOutcome",
    "Replayer",
    "VerdictLedger",
    "is_replay_deterministic",
]
