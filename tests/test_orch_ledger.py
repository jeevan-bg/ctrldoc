"""Verdict ledger — append-only persistence + replay determinism.

SPEC §6.5 makes every verdict replayable: a downstream consumer must be
able to re-run an L4 operation with the persisted inputs and recover
the same calibrated confidence within a tight tolerance band. §11
surfaces the ledger through the CLI (`ctrldoc ledger {list, show,
replay}`). Together they pin the L4 "every verdict is replayable"
non-negotiable from §13.

The tests below pin:

* `VerdictLedger.append` writes one row per call; the table is the SQL
  `verdict_ledger` table from §8.
* The persisted shape carries the full §8 column set: workspace id,
  operation name, JSON inputs, JSON output, calibrated confidence,
  model versions, optional paraphrase-vote breakdown, ISO timestamp.
* `list_entries` returns rows in append order via the AUTOINCREMENT id.
* `get` retrieves by id; missing id raises `LedgerEntryNotFoundError`.
* `replay` re-runs the recorded operation through a caller-supplied
  `Replayer` callback and reports the absolute confidence delta against
  the persisted value. The release gate is `delta <= 0.02` per §6.5 —
  the helper `is_replay_deterministic` thresholds against the shipped
  `REPLAY_TOLERANCE` constant.
* Append-only contract: the public API has no update / delete entry
  points. Re-appending an "identical" payload produces a new row with
  a fresh id; nothing overwrites in place.

SPEC-REF: §6.5 (replayable verdicts), §11 (CLI surface)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.orch.ledger import (
    REPLAY_TOLERANCE,
    LedgerAppendRequest,
    LedgerEntry,
    LedgerEntryNotFoundError,
    ReplayOutcome,
    VerdictLedger,
    is_replay_deterministic,
)
from ctrldoc.store.sqlite import SQLiteStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger(tmp_path: Path) -> VerdictLedger:
    store = SQLiteStore(tmp_path / "ledger.db")
    return VerdictLedger(store=store)


def _request(
    *,
    workspace_id: str = "ws-1",
    operation: str = "coverage",
    inputs: dict[str, object] | None = None,
    output: dict[str, object] | None = None,
    calibrated_confidence: float = 0.83,
    model_versions: dict[str, str] | None = None,
    paraphrase_votes: dict[str, int] | None = None,
    timestamp: str = "2026-05-24T12:00:00Z",
) -> LedgerAppendRequest:
    return LedgerAppendRequest(
        workspace_id=workspace_id,
        operation=operation,
        inputs=dict(inputs or {"target_doc_id": "d1", "source_doc_id": "d2"}),
        output=dict(output or {"per_claim": [], "summary": {}}),
        calibrated_confidence=calibrated_confidence,
        model_versions=dict(model_versions or {"nli": "deberta-v3-large-mnli"}),
        paraphrase_votes=dict(paraphrase_votes) if paraphrase_votes is not None else None,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_replay_tolerance_matches_spec_gate() -> None:
    """The replay-determinism gate from §6.5 is the ±0.02 tolerance."""
    assert pytest.approx(0.02) == REPLAY_TOLERANCE


# ---------------------------------------------------------------------------
# Append + read
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_append_returns_entry_with_assigned_id(tmp_path: Path) -> None:
    """`append` returns the persisted `LedgerEntry`, including its AUTOINCREMENT id."""
    ledger = _ledger(tmp_path)
    entry = ledger.append(_request())
    assert isinstance(entry, LedgerEntry)
    assert entry.id >= 1
    assert entry.workspace_id == "ws-1"
    assert entry.operation == "coverage"
    assert entry.calibrated_confidence == pytest.approx(0.83)


@pytest.mark.family_referential_integrity
def test_append_preserves_input_and_output_payloads(tmp_path: Path) -> None:
    """The JSON round-trip preserves the recorded `inputs` / `output` exactly."""
    ledger = _ledger(tmp_path)
    payload_inputs = {"target_doc_id": "d-target", "source_doc_id": "d-source", "k": 5}
    payload_output = {
        "per_claim": [{"target_claim_id": "c1", "verdict": "Covered"}],
        "summary": {"covered_rate": 1.0},
    }
    entry = ledger.append(_request(inputs=payload_inputs, output=payload_output))
    fetched = ledger.get(entry.id)
    assert fetched.inputs == payload_inputs
    assert fetched.output == payload_output


@pytest.mark.family_referential_integrity
def test_append_persists_optional_paraphrase_votes(tmp_path: Path) -> None:
    """Paraphrase-vote breakdown is optional and round-trips when supplied."""
    ledger = _ledger(tmp_path)
    votes = {"entailment": 3, "contradiction": 0, "neutral": 0}
    entry = ledger.append(_request(paraphrase_votes=votes))
    fetched = ledger.get(entry.id)
    assert fetched.paraphrase_votes == votes


@pytest.mark.family_referential_integrity
def test_append_paraphrase_votes_default_to_none(tmp_path: Path) -> None:
    """Unsupplied paraphrase votes persist as SQL NULL → Python `None`."""
    ledger = _ledger(tmp_path)
    entry = ledger.append(_request(paraphrase_votes=None))
    fetched = ledger.get(entry.id)
    assert fetched.paraphrase_votes is None


@pytest.mark.family_referential_integrity
def test_list_entries_returns_rows_in_append_order(tmp_path: Path) -> None:
    """`list_entries` orders by the AUTOINCREMENT id == insertion order."""
    ledger = _ledger(tmp_path)
    ids: list[int] = []
    for op, conf, ts in [
        ("coverage", 0.91, "2026-05-24T00:00:00Z"),
        ("compare", 0.72, "2026-05-24T00:00:01Z"),
        ("merge", 0.85, "2026-05-24T00:00:02Z"),
    ]:
        ids.append(
            ledger.append(_request(operation=op, calibrated_confidence=conf, timestamp=ts)).id
        )
    rows = ledger.list_entries()
    assert [row.id for row in rows] == ids
    assert [row.operation for row in rows] == ["coverage", "compare", "merge"]


@pytest.mark.family_referential_integrity
def test_list_entries_filters_by_workspace(tmp_path: Path) -> None:
    """`workspace_id` filter narrows the result set without affecting global order."""
    ledger = _ledger(tmp_path)
    ledger.append(_request(workspace_id="ws-a", operation="coverage"))
    ledger.append(_request(workspace_id="ws-b", operation="compare"))
    ledger.append(_request(workspace_id="ws-a", operation="merge"))
    rows = ledger.list_entries(workspace_id="ws-a")
    assert [row.operation for row in rows] == ["coverage", "merge"]
    assert all(row.workspace_id == "ws-a" for row in rows)


@pytest.mark.family_referential_integrity
def test_get_raises_for_missing_id(tmp_path: Path) -> None:
    """`get` is strict — an unknown id must be loud, not a silent `None`."""
    ledger = _ledger(tmp_path)
    with pytest.raises(LedgerEntryNotFoundError):
        ledger.get(999)


# ---------------------------------------------------------------------------
# Append-only contract
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_re_append_creates_a_new_row(tmp_path: Path) -> None:
    """Re-appending an "identical" payload allocates a fresh AUTOINCREMENT id."""
    ledger = _ledger(tmp_path)
    payload = _request()
    first = ledger.append(payload)
    second = ledger.append(payload)
    assert second.id != first.id
    assert ledger.get(first.id).id == first.id
    assert ledger.get(second.id).id == second.id


@pytest.mark.family_referential_integrity
def test_public_api_has_no_mutator(tmp_path: Path) -> None:
    """`VerdictLedger` exposes only append / read methods — no update / delete."""
    ledger = _ledger(tmp_path)
    public_methods = {name for name in dir(ledger) if not name.startswith("_")}
    forbidden = {"update", "delete", "remove", "clear", "drop"}
    assert public_methods.isdisjoint(
        forbidden
    ), f"VerdictLedger leaked mutator method(s): {public_methods & forbidden}"


# ---------------------------------------------------------------------------
# Replay determinism (the §6.5 release gate)
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_replay_recovers_persisted_confidence_within_tolerance(tmp_path: Path) -> None:
    """An idempotent replayer returns the same confidence — delta ≤ tolerance."""
    ledger = _ledger(tmp_path)
    entry = ledger.append(_request(calibrated_confidence=0.83))

    def replayer(inputs: dict[str, object]) -> float:
        # Honest replay: same inputs, exact same calibrated probability.
        assert "target_doc_id" in inputs
        return 0.83

    outcome = ledger.replay(entry.id, replayer=replayer)
    assert isinstance(outcome, ReplayOutcome)
    assert outcome.entry_id == entry.id
    assert outcome.persisted_confidence == pytest.approx(0.83)
    assert outcome.replayed_confidence == pytest.approx(0.83)
    assert outcome.delta == pytest.approx(0.0)
    assert outcome.is_deterministic
    assert outcome.tolerance == pytest.approx(REPLAY_TOLERANCE)


@pytest.mark.family_determinism
def test_replay_detects_drift_above_tolerance(tmp_path: Path) -> None:
    """A replay above the tolerance band fails the determinism contract."""
    ledger = _ledger(tmp_path)
    entry = ledger.append(_request(calibrated_confidence=0.50))

    # 0.05 drift is comfortably above the ±0.02 band.
    outcome = ledger.replay(entry.id, replayer=lambda _inputs: 0.55)
    assert outcome.delta == pytest.approx(0.05)
    assert not outcome.is_deterministic


@pytest.mark.family_determinism
def test_replay_tolerates_exact_boundary(tmp_path: Path) -> None:
    """A drift equal to the tolerance is the gate's boundary — still deterministic."""
    ledger = _ledger(tmp_path)
    entry = ledger.append(_request(calibrated_confidence=0.40))
    outcome = ledger.replay(entry.id, replayer=lambda _inputs: 0.40 + REPLAY_TOLERANCE)
    assert outcome.delta == pytest.approx(REPLAY_TOLERANCE)
    assert outcome.is_deterministic


@pytest.mark.family_determinism
def test_replay_passes_recorded_inputs_to_replayer(tmp_path: Path) -> None:
    """The replayer receives the persisted `inputs` dict unmodified."""
    ledger = _ledger(tmp_path)
    payload_inputs = {"workspace": "ws-1", "items": ["x", "y", "z"]}
    entry = ledger.append(_request(inputs=payload_inputs, calibrated_confidence=0.66))

    received: dict[str, object] = {}

    def replayer(inputs: dict[str, object]) -> float:
        received.update(inputs)
        return 0.66

    ledger.replay(entry.id, replayer=replayer)
    assert received == payload_inputs


@pytest.mark.family_determinism
def test_replay_raises_for_missing_id(tmp_path: Path) -> None:
    """Replaying an unknown id raises the same error class as `get`."""
    ledger = _ledger(tmp_path)
    with pytest.raises(LedgerEntryNotFoundError):
        ledger.replay(42, replayer=lambda _i: 0.0)


@pytest.mark.family_determinism
def test_is_replay_deterministic_helper_matches_outcome_flag(tmp_path: Path) -> None:
    """The stand-alone helper agrees with `ReplayOutcome.is_deterministic`."""
    ledger = _ledger(tmp_path)
    entry = ledger.append(_request(calibrated_confidence=0.70))
    outcome_within = ledger.replay(entry.id, replayer=lambda _i: 0.71)
    outcome_outside = ledger.replay(entry.id, replayer=lambda _i: 0.40)
    assert is_replay_deterministic(0.70, 0.71)
    assert outcome_within.is_deterministic
    assert not is_replay_deterministic(0.70, 0.40)
    assert not outcome_outside.is_deterministic
