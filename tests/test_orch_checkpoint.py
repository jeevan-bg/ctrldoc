"""Resumability — per-sub-task checkpoint store.

A playbook fan-out is a list of items with stable ids. After each
sub-task completes, its structured output is appended to
`runs/{run_id}/state.json` as a JSON object keyed by item id. A
crash at item 47/100 leaves 47 results on disk; resuming reads them
back and the playbook only fires for the remaining 53 ids.

Writes are atomic (write to `state.json.tmp` then `rename`) so a
crash mid-flush either preserves the previous good state or applies
the new one in full — never a torn file.

SPEC-REF: §4.7 (resumability), §8.6 family 12 (failure resilience)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from ctrldoc.orch.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointStore


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    confidence: float


# --- fresh store ---


def test_new_store_has_no_completions(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    assert store.completed() == {}


def test_state_path_under_run_id(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    assert store.state_path == tmp_path / "run-1" / "state.json"


def test_state_file_not_created_until_first_save(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    assert not store.state_path.exists()
    store.completed()  # cheap read; must not create the file either
    assert not store.state_path.exists()


# --- save and round-trip ---


def test_save_dict_result_persists_to_disk(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save("item-a", {"label": "ok", "confidence": 0.9})

    payload = json.loads(store.state_path.read_text())
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["run_id"] == "run-1"
    assert payload["results"] == {"item-a": {"label": "ok", "confidence": 0.9}}


def test_save_pydantic_result_serialises_via_model_dump(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save("item-a", _Result(label="ok", confidence=0.9))

    payload = json.loads(store.state_path.read_text())
    assert payload["results"]["item-a"] == {"label": "ok", "confidence": 0.9}


def test_completed_returns_results_from_memory_and_disk(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save("a", {"v": 1})
    store.save("b", {"v": 2})

    # Same store: in-memory view.
    assert store.completed() == {"a": {"v": 1}, "b": {"v": 2}}

    # Fresh store over the same path: reads from disk.
    store2 = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    assert store2.completed() == {"a": {"v": 1}, "b": {"v": 2}}


def test_save_is_idempotent_under_repeated_id(tmp_path: Path) -> None:
    """The latest write for a given id wins — playbooks that retry a
    failed sub-task overwrite the prior result rather than accumulating."""
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save("a", {"v": 1})
    store.save("a", {"v": 2})
    assert store.completed() == {"a": {"v": 2}}


# --- pending filter ---


def test_pending_filters_out_already_saved_items(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save("a", {"v": 1})
    store.save("c", {"v": 3})
    assert store.pending(["a", "b", "c", "d"]) == ["b", "d"]


def test_pending_preserves_input_order(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save("z", {"v": 0})
    # Order in the input list is preserved on output.
    assert store.pending(["m", "z", "a"]) == ["m", "a"]


# --- atomic write ---


def test_save_uses_tmp_then_rename_for_atomicity(tmp_path: Path) -> None:
    """After a successful save there is no leftover .tmp file."""
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save("a", {"v": 1})
    siblings = list(store.state_path.parent.iterdir())
    names = {p.name for p in siblings}
    assert names == {"state.json"}


def test_torn_tmp_file_does_not_disrupt_load(tmp_path: Path) -> None:
    """A leftover state.json.tmp from a previous crash must not be loaded."""
    state_dir = tmp_path / "run-1"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "run_id": "run-1",
                "results": {"a": {"v": 1}},
            }
        )
    )
    # Half-written tmp from a previous crash.
    (state_dir / "state.json.tmp").write_text("{not valid json")
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    assert store.completed() == {"a": {"v": 1}}


# --- schema versioning ---


def test_mismatched_schema_version_refuses_to_load(tmp_path: Path) -> None:
    state_dir = tmp_path / "run-1"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": "999",
                "run_id": "run-1",
                "results": {"a": {"v": 1}},
            }
        )
    )
    with pytest.raises(ValueError, match="schema_version"):
        CheckpointStore(runs_dir=tmp_path, run_id="run-1")


def test_run_id_mismatch_refuses_to_load(tmp_path: Path) -> None:
    """Loading a state file written by a different run is almost
    certainly a config error — fail loud."""
    state_dir = tmp_path / "run-1"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "run_id": "different-run",
                "results": {"a": {"v": 1}},
            }
        )
    )
    with pytest.raises(ValueError, match="run_id"):
        CheckpointStore(runs_dir=tmp_path, run_id="run-1")


# --- end-to-end resume ---


@pytest.mark.family_failure_resilience
def test_simulated_crash_at_item_47_resumes_from_48(tmp_path: Path) -> None:
    """The §4.7 resumability vignette: crash at 47/100 → resume from 48."""
    all_ids = [f"item-{i:03d}" for i in range(100)]

    # First run: save 47 results then "crash" (just stop saving).
    run_a = CheckpointStore(runs_dir=tmp_path, run_id="run-x")
    for item_id in all_ids[:47]:
        run_a.save(item_id, {"v": int(item_id.split("-")[-1])})
    # Simulated crash — drop the reference, don't call any cleanup.
    del run_a

    # Resume: a fresh store sees the 47 results and reports the right
    # pending list.
    run_b = CheckpointStore(runs_dir=tmp_path, run_id="run-x")
    pending = run_b.pending(all_ids)
    assert pending == all_ids[47:]
    assert len(run_b.completed()) == 47


# --- save accepts model-dump output ---


def test_save_round_trips_nested_lists_and_floats(tmp_path: Path) -> None:
    store = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    store.save(
        "complex",
        {
            "spans": [{"chunk_id": "c1", "char_start": 0, "char_end": 5}],
            "score": 0.875,
            "labels": ["a", "b"],
        },
    )
    fresh = CheckpointStore(runs_dir=tmp_path, run_id="run-1")
    assert fresh.completed()["complex"]["score"] == pytest.approx(0.875)
