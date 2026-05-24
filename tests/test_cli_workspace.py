"""`ctrldoc workspace {create|add|list|info}` CLI surface.

The workspace subcommand group is the v1 entry into the L2.5 primitive.
Each subcommand resolves the persistent store under
`<runs_path>/workspaces.db` (the workspace-scope store; per-doc
indexes stay where they are at `<runs_path>/indexes/<doc_hash>.db`)
and dispatches to `WorkspaceManager`. Markdown + JSON output honor
the global `--format` flag exactly like the other commands.

SPEC-REF: §6.7 (workspace), §9 (CLI surface)
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ctrldoc.cli import app

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


def _run(cfg: Path, *args: str, format_flag: str = "json") -> object:
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--format",
            format_flag,
            "workspace",
            *args,
        ],
    )
    return result


# --- create ---


def test_workspace_create_emits_id_and_persists(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "create", "audit-2026")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "workspace create"
    assert payload["status"] == "ok"
    assert payload["name"] == "audit-2026"
    assert payload["id"].startswith("ws-")
    assert (tmp_path / "runs" / "workspaces.db").exists()


def test_workspace_create_duplicate_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _run(cfg, "create", "audit-2026")
    result = _run(cfg, "create", "audit-2026")
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "already exists" in result.stderr  # type: ignore[attr-defined]


def test_workspace_create_blank_name_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "create", "   ")
    assert result.exit_code == 2  # type: ignore[attr-defined]


# --- add ---


def test_workspace_add_appends_doc(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _run(cfg, "create", "audit-2026")
    result = _run(cfg, "add", "audit-2026", "doc-a")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["doc_ids"] == ["doc-a"]
    _run(cfg, "add", "audit-2026", "doc-b")
    info = _run(cfg, "info", "audit-2026")
    info_payload = json.loads(info.stdout)  # type: ignore[attr-defined]
    assert info_payload["doc_ids"] == ["doc-a", "doc-b"]


def test_workspace_add_missing_workspace_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "add", "missing", "doc-a")
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "not found" in result.stderr  # type: ignore[attr-defined]


# --- list ---


def test_workspace_list_empty_returns_empty_workspaces(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "list")
    assert result.exit_code == 0  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "workspace list"
    assert payload["workspaces"] == []


def test_workspace_list_renders_all(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _run(cfg, "create", "audit-2026")
    _run(cfg, "create", "spec-vs-impl")
    _run(cfg, "add", "spec-vs-impl", "doc-a")
    result = _run(cfg, "list")
    assert result.exit_code == 0  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    names = {w["name"] for w in payload["workspaces"]}
    assert names == {"audit-2026", "spec-vs-impl"}
    counts = {w["name"]: w["doc_count"] for w in payload["workspaces"]}
    assert counts == {"audit-2026": 0, "spec-vs-impl": 1}


# --- info ---


def test_workspace_info_exposes_shared_concept_count(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _run(cfg, "create", "audit-2026")
    _run(cfg, "add", "audit-2026", "doc-a")
    result = _run(cfg, "info", "audit-2026")
    assert result.exit_code == 0  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "workspace info"
    assert payload["name"] == "audit-2026"
    assert payload["doc_count"] == 1
    assert payload["concept_count"] == 0  # no concepts inserted yet
    assert payload["doc_ids"] == ["doc-a"]


def test_workspace_info_missing_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "info", "missing")
    assert result.exit_code == 2  # type: ignore[attr-defined]


# --- markdown default ---


def test_workspace_create_markdown_renders_header(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _run(cfg, "create", "audit-2026", format_flag="markdown")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    assert "# ctrldoc — workspace create" in result.stdout  # type: ignore[attr-defined]
    assert "audit-2026" in result.stdout  # type: ignore[attr-defined]


def test_workspace_info_markdown_renders_summary(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    _run(cfg, "create", "audit-2026")
    _run(cfg, "add", "audit-2026", "doc-a")
    result = _run(cfg, "info", "audit-2026", format_flag="markdown")
    assert result.exit_code == 0  # type: ignore[attr-defined]
    assert "# ctrldoc — workspace info" in result.stdout  # type: ignore[attr-defined]
    assert "audit-2026" in result.stdout  # type: ignore[attr-defined]
    assert "doc-a" in result.stdout  # type: ignore[attr-defined]
