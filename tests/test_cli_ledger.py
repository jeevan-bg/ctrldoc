"""`ctrldoc ledger {list|show|replay}` CLI surface.

The ledger subcommand group surfaces the §6.5 replay contract through
the CLI. Each subcommand resolves the per-installation ledger DB under
`<runs_path>/ledger.db` (the same one-file-per-substrate pattern the
workspace sub-app uses) and dispatches to a `VerdictLedger` over a
`SQLiteStore`.

Because the v1 orchestrator paths that actually emit verdicts are not
yet wired, the CLI is exercised against a seeded ledger DB in these
tests: the fixture pre-pops one or more rows directly via
`VerdictLedger.append`, then drives `ledger list / show / replay`
through the typer runner. That keeps the surface tests independent of
the op-routing work that follows.

`replay` over the CLI uses a degenerate "identity" replayer that
echoes the persisted confidence back — enough to prove the round-trip
without pulling in every op's import graph. Real replay backends will
register through the L4 tool dispatcher in the MCP server work.

SPEC-REF: §6.5 (replayable verdicts), §11 (CLI surface)
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ctrldoc.cli import app
from ctrldoc.orch.ledger import LedgerAppendRequest, VerdictLedger
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


def _seed(
    db_path: Path,
    *,
    workspace_id: str = "ws-1",
    operation: str = "coverage",
    calibrated_confidence: float = 0.83,
    timestamp: str = "2026-05-24T12:00:00Z",
) -> int:
    """Append one row to the ledger DB the CLI will open; return its id."""
    with SQLiteStore(db_path) as store:
        ledger = VerdictLedger(store=store)
        entry = ledger.append(
            LedgerAppendRequest(
                workspace_id=workspace_id,
                operation=operation,
                inputs={"target_doc_id": "d1", "source_doc_id": "d2"},
                output={"per_claim": []},
                calibrated_confidence=calibrated_confidence,
                model_versions={"nli": "deberta-v3-large-mnli"},
                timestamp=timestamp,
            )
        )
    return entry.id


def _run(cfg: Path, *args: str, format_flag: str = "json") -> object:
    return runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--format",
            format_flag,
            "ledger",
            *args,
        ],
    )


# --- list ---


def test_ledger_list_empty_returns_no_entries(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "list")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "ledger list"
    assert payload["status"] == "ok"
    assert payload["entries"] == []


def test_ledger_list_returns_seeded_rows_in_append_order(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    db_path = _ledger_db_path(tmp_path)
    _seed(db_path, operation="coverage", calibrated_confidence=0.91)
    _seed(db_path, operation="compare", calibrated_confidence=0.72)
    result = _run(cfg, "list")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    ops = [row["operation"] for row in payload["entries"]]
    assert ops == ["coverage", "compare"]


def test_ledger_list_filters_by_workspace(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    db_path = _ledger_db_path(tmp_path)
    _seed(db_path, workspace_id="ws-a", operation="coverage")
    _seed(db_path, workspace_id="ws-b", operation="compare")
    result = _run(cfg, "list", "--workspace-id", "ws-a")
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert {row["workspace_id"] for row in payload["entries"]} == {"ws-a"}


# --- show ---


def test_ledger_show_returns_full_entry(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    db_path = _ledger_db_path(tmp_path)
    entry_id = _seed(db_path, operation="merge", calibrated_confidence=0.55)
    result = _run(cfg, "show", str(entry_id))
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "ledger show"
    assert payload["entry"]["id"] == entry_id
    assert payload["entry"]["operation"] == "merge"
    assert payload["entry"]["calibrated_confidence"] == 0.55
    assert payload["entry"]["inputs"] == {
        "target_doc_id": "d1",
        "source_doc_id": "d2",
    }


def test_ledger_show_missing_id_exits_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "show", "999")
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "not found" in result.stderr  # type: ignore[attr-defined]


# --- replay ---


def test_ledger_replay_identity_passes_determinism_gate(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    db_path = _ledger_db_path(tmp_path)
    entry_id = _seed(db_path, operation="coverage", calibrated_confidence=0.77)
    result = _run(cfg, "replay", str(entry_id))
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "ledger replay"
    assert payload["entry_id"] == entry_id
    assert payload["persisted_confidence"] == 0.77
    assert payload["replayed_confidence"] == 0.77
    assert payload["delta"] == 0.0
    assert payload["tolerance"] == 0.02
    assert payload["is_deterministic"] is True


def test_ledger_replay_missing_id_exits_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "replay", "999")
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "not found" in result.stderr  # type: ignore[attr-defined]


# --- markdown rendering ---


def test_ledger_list_markdown_renders_table(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    db_path = _ledger_db_path(tmp_path)
    _seed(db_path, operation="coverage", calibrated_confidence=0.91)
    _seed(db_path, operation="compare", calibrated_confidence=0.72)
    result = _run(cfg, "list", format_flag="markdown")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    out = result.stdout  # type: ignore[attr-defined]
    assert "ledger" in out
    assert "coverage" in out
    assert "compare" in out
