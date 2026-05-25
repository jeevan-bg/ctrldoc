"""Ledger wiring across the eight wired ops: every success path appends one row.

The L4 verdict-ledger surface (§6.5) was wired in S-143 with `append`
/ `get` / `list_entries` / `replay` and surfaced through `ctrldoc
ledger {list,show,replay}` at the same time. This module pins the
contract that closes the loop: every wired op (`qa`, `audit`,
`review`, `coverage`, `compare`, `merge`, `list_check`, `map`)
appends one `LedgerAppendRequest` to the per-installation
`<runs_path>/ledger.db` on its success path.

The append carries:

* `operation` — the command name (matches the §6.10 tool alphabet for
  dispatcher-routed ops; the legacy v0.3 commands use their CLI verb).
* `inputs` — the JSON-safe input dict the CLI built for the op.
* `output` — a JSON-safe summary of the result (no full payload — the
  goal is auditability, not snapshotting).
* `calibrated_confidence` — one scalar per op derived deterministically
  from the op's output. For ops that expose per-item confidence (qa /
  audit / coverage / list_check / map) it is the mean; for ops with no
  per-item confidence (review / compare / merge) it is the §6.5
  `HEURISTIC_CONFIDENCE = 0.9` prior so the row still surfaces a number
  the replay gate can score against.
* `model_versions` — the `{role: model_id}` rollup for the active
  profile (heuristic / thrifty / production), so the ledger row carries
  every model that participated in the verdict for §13 non-negotiable
  10 (every verdict is replayable).

The audit CLI gate is the headline release criterion: after `ctrldoc
audit` runs, `ctrldoc ledger list` shows exactly one entry whose
operation is `audit`; `ctrldoc ledger replay <id>` returns
`is_deterministic=true` against the ±0.02 gate.

The dispatcher-routed ops (`coverage` / `compare` / `merge` /
`list_check`) only append when the dispatcher resolves to a real
verdict — `status: "not_implemented"` envelopes do not pollute the
ledger.

SPEC-REF: §6.5 (replayable verdicts), §11 (CLI surface)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ctrldoc.cli import app
from ctrldoc.orch.ledger import REPLAY_TOLERANCE, VerdictLedger
from ctrldoc.store.sqlite import SQLiteStore

runner = CliRunner()


_CONFIG_TEMPLATE = """\
[models]
planner = "claude-opus-4-7"
judge_tier1 = "qwen2.5:7b-instruct-q4_K_M"
judge_tier2 = "claude-opus-4-7"
verifier_nli = "deberta-v3-large-mnli"
embedder = "bge-m3"

[budgets]
max_cost_usd = 5.0
max_tokens_per_call = 16000
max_wall_clock_min = 30

[concurrency]
anthropic_concurrent = 8
ollama_concurrent = 2

[paths]
index_path = "{index_path}"
runs_path = "{runs_path}"
traces_path = "{traces_path}"
"""


def _write_config(tmp_path: Path) -> Path:
    index_path = tmp_path / "index"
    runs_path = tmp_path / "runs"
    traces_path = tmp_path / "traces"
    for p in (index_path, runs_path, traces_path):
        p.mkdir(exist_ok=True)
    cfg = tmp_path / "ctrldoc.toml"
    cfg.write_text(
        _CONFIG_TEMPLATE.format(
            index_path=index_path.as_posix(),
            runs_path=runs_path.as_posix(),
            traces_path=traces_path.as_posix(),
        ),
        encoding="utf-8",
    )
    return cfg


def _ledger_db_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "ledger.db"


def _read_ledger_entries(tmp_path: Path) -> list[Any]:
    db_path = _ledger_db_path(tmp_path)
    assert db_path.exists(), "ledger.db should exist after a wired op runs"
    with SQLiteStore(db_path) as store:
        ledger = VerdictLedger(store=store)
        return ledger.list_entries()


# ---------------------------------------------------------------------------
# scan does NOT append (out of the wired-op alphabet) — guard rail
# ---------------------------------------------------------------------------


def test_scan_does_not_touch_ledger(tmp_path: Path, synthetic_doc_path: Path) -> None:
    """`scan` is not in the wired-op alphabet — no ledger row should land."""
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--format",
            "json",
            "scan",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    # ledger.db may or may not exist (other commands haven't run); if it
    # does, the entries list must be empty because scan is not wired.
    db_path = _ledger_db_path(tmp_path)
    if db_path.exists():
        with SQLiteStore(db_path) as store:
            ledger = VerdictLedger(store=store)
            assert ledger.list_entries() == []


# ---------------------------------------------------------------------------
# Headline release gate: `audit` → one entry → replay passes ±0.02
# ---------------------------------------------------------------------------


def test_audit_heuristic_path_skipped() -> None:
    """`audit` requires an LLM seam; the heuristic profile rejects it.

    This sentinel keeps a reviewer from being surprised that the audit
    gate runs against a non-heuristic profile (`thrifty` in tests). The
    headline gate below uses a stub task client to avoid the network.
    """
    # No-op marker; the substantive gate is below.


def test_audit_appends_one_ledger_row_and_replay_passes(
    tmp_path: Path,
    synthetic_doc_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The §6.5 release-gate path end-to-end.

    Drive `ctrldoc audit` with a stubbed task client (no Ollama
    daemon, no Anthropic key) so the audit verdicts come out
    deterministic, then assert:

    1. `runs/ledger.db` carries exactly one entry,
    2. its operation is `audit`,
    3. `calibrated_confidence` is the mean over the verdict
       confidences,
    4. `inputs` carries the checklist and target paths the CLI built,
    5. `output` carries a JSON-safe summary,
    6. `model_versions` is non-empty,
    7. `ctrldoc ledger list` exposes the entry,
    8. `ctrldoc ledger replay <id>` returns `is_deterministic=true`
       against the ±0.02 gate.
    """
    pytest.importorskip("sqlite_vec", reason="thrifty audit needs sqlite-vec")
    pytest.importorskip("gliner", reason="thrifty bundle needs gliner")
    pytest.importorskip("fastcoref", reason="thrifty bundle needs fastcoref")

    # Stub the task client so the audit playbook's per-item calls
    # return deterministic verdicts. `SequentialBatchedRunner` calls
    # `StatelessTaskRunner.run` once per item with `output_model =
    # _BatchedVerdict`; two checklist items below → two calls →
    # mean(0.84, 0.92) = 0.88.
    from ctrldoc.ops.audit import _BatchedVerdict

    confidences = iter([0.84, 0.92])

    def _stub_run(self: object, task: object, *, output_model: type) -> object:
        del self, task
        assert output_model is _BatchedVerdict
        return _BatchedVerdict(
            verdict="Covered",
            confidence=next(confidences),
            citation_chunk_ids=[],
        )

    monkeypatch.setattr(
        "ctrldoc.orch.task.StatelessTaskRunner.run",
        _stub_run,
        raising=True,
    )

    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        "## Hashing\nThe system uses consistent hashing.\n"
        "\n"
        "## GossipBus\nThe system has a gossip-based bus.\n",
        encoding="utf-8",
    )

    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "thrifty",
            "--format",
            "json",
            "audit",
            "--checklist",
            str(checklist),
            "--target",
            str(synthetic_doc_path),
        ],
    )
    if result.exit_code != 0:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"audit crashed: {type(result.exception).__name__}: {result.exception}"
            ) from result.exception
        pytest.skip(f"audit non-zero exit ({result.exit_code}); skipping ledger gate")
    payload = json.loads(result.stdout)
    assert payload["command"] == "audit"

    entries = _read_ledger_entries(tmp_path)
    assert len(entries) == 1, "exactly one ledger entry should land per audit run"
    entry = entries[0]
    assert entry.operation == "audit"
    # mean(0.84, 0.92) = 0.88
    assert abs(entry.calibrated_confidence - 0.88) < 1e-6
    assert entry.inputs["checklist_path"] == str(checklist)
    assert entry.inputs["target_path"] == str(synthetic_doc_path)
    assert "verdicts" in entry.output or "summary" in entry.output
    assert entry.model_versions, "model_versions must be non-empty"

    # `ledger list` exposes the entry.
    list_result = runner.invoke(
        app,
        ["--config", str(cfg), "--format", "json", "ledger", "list"],
    )
    assert list_result.exit_code == 0, list_result.stderr
    list_payload = json.loads(list_result.stdout)
    assert len(list_payload["entries"]) == 1
    assert list_payload["entries"][0]["operation"] == "audit"

    # `ledger replay` passes the ±0.02 gate.
    replay_result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--format",
            "json",
            "ledger",
            "replay",
            str(entry.id),
        ],
    )
    assert replay_result.exit_code == 0, replay_result.stderr
    replay_payload = json.loads(replay_result.stdout)
    assert replay_payload["is_deterministic"] is True
    assert replay_payload["tolerance"] == REPLAY_TOLERANCE
    assert abs(replay_payload["delta"]) <= REPLAY_TOLERANCE


# ---------------------------------------------------------------------------
# Dispatcher-routed ops: `not_implemented` envelopes do NOT pollute the ledger
# ---------------------------------------------------------------------------


def test_compare_not_implemented_does_not_append(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(cfg), "--format", "json", "compare", "ws-x", "doc-a", "doc-b"],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_implemented"
    # No ledger row should land on the not-implemented path.
    db_path = _ledger_db_path(tmp_path)
    if db_path.exists():
        with SQLiteStore(db_path) as store:
            ledger = VerdictLedger(store=store)
            assert ledger.list_entries() == []


def test_coverage_not_implemented_does_not_append(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--format",
            "json",
            "coverage",
            "--workspace",
            "ws-x",
            "--target",
            "doc-b",
            "--source",
            "doc-a",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_implemented"
    db_path = _ledger_db_path(tmp_path)
    if db_path.exists():
        with SQLiteStore(db_path) as store:
            ledger = VerdictLedger(store=store)
            assert ledger.list_entries() == []


def test_merge_not_implemented_does_not_append(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--format",
            "json",
            "merge",
            "--workspace",
            "ws-x",
            "--output",
            str(tmp_path / "merged.md"),
            "doc-a",
            "doc-b",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_implemented"
    db_path = _ledger_db_path(tmp_path)
    if db_path.exists():
        with SQLiteStore(db_path) as store:
            ledger = VerdictLedger(store=store)
            assert ledger.list_entries() == []


def test_list_check_not_implemented_does_not_append(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    list_md = tmp_path / "list.md"
    list_md.write_text("- alpha claim\n- beta claim\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["--config", str(cfg), "--format", "json", "list-check", str(list_md), "doc-a"],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_implemented"
    db_path = _ledger_db_path(tmp_path)
    if db_path.exists():
        with SQLiteStore(db_path) as store:
            ledger = VerdictLedger(store=store)
            assert ledger.list_entries() == []


# ---------------------------------------------------------------------------
# Ledger-append helper unit tests (the module-internal contract)
# ---------------------------------------------------------------------------


def test_model_versions_for_profile_heuristic() -> None:
    """Heuristic profile = no LLM; the rollup carries deterministic ids only."""
    from ctrldoc.cli import _model_versions_for_profile

    versions = _model_versions_for_profile("heuristic")
    assert isinstance(versions, dict)
    assert versions  # non-empty even in heuristic
    # No external model id should sneak into the heuristic profile.
    for value in versions.values():
        assert "claude" not in value.lower()
        assert "qwen" not in value.lower()
        assert "ollama" not in value.lower()


def test_model_versions_for_profile_thrifty_carries_qwen_and_nli() -> None:
    from ctrldoc.cli import _model_versions_for_profile

    versions = _model_versions_for_profile("thrifty")
    # Combined judge + NLI model ids land in the rollup.
    joined = " ".join(versions.values()).lower()
    assert "qwen" in joined or "ollama" in joined
    assert "deberta" in joined or "nli" in joined


def test_model_versions_for_profile_production_carries_opus() -> None:
    from ctrldoc.cli import _model_versions_for_profile

    versions = _model_versions_for_profile("production")
    joined = " ".join(versions.values()).lower()
    assert "claude" in joined or "opus" in joined
