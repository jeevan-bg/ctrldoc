"""CLI surface tests via `typer.testing.CliRunner` — hermetic, no LLM.

The ``ctrldoc`` typer app exposes six subcommands (one per use case).
This file covers:

  * ``--help`` rendering at the root and per subcommand.
  * The end-to-end ``ingest`` path running the deterministic L0
    pipeline against the synthetic gold doc and writing two
    artefacts (stats + canary signature).
  * Argument validation: blank queries / blank doc types / unknown
    audit kinds / missing files all exit with code 2.
  * The four LLM-backed stubs emit a structured JSON envelope with
    a ``next_step`` describing what production wiring is required.
  * ``scan`` runs the deterministic detector battery over an
    in-memory store.

SPEC-REF: §6, §12
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctrldoc.canary import load_baseline
from ctrldoc.cli import app

runner = CliRunner()


# --- root help ---


def test_root_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "qa", "audit", "review", "scan", "map"):
        assert command in result.stdout, f"subcommand {command!r} missing from --help"


def test_root_no_args_exits_with_help() -> None:
    """`no_args_is_help=True` makes a bare invocation print --help and exit 0."""
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)  # typer historically returned 0; newer returns 2
    assert "ingest" in result.stdout


# --- ingest end-to-end ---


def test_ingest_runs_against_synthetic_doc(
    tmp_path: Path,
    synthetic_doc_path: Path,
) -> None:
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "ingest",
            str(synthetic_doc_path),
            "--output-dir",
            str(out),
            "--doc-id",
            "aurora",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "ingest"
    assert payload["status"] == "ok"
    assert payload["chunks_indexed"] > 0

    stats = json.loads((out / "aurora__ingest_stats.json").read_text(encoding="utf-8"))
    assert stats["doc_id"] == "aurora"
    assert stats["chunks_indexed"] == payload["chunks_indexed"]

    baseline = load_baseline(out / "aurora__ingest_signature.json")
    assert baseline.doc_id == "aurora"
    assert baseline.playbook == "ingest"
    assert baseline.signature["chunk_ids"], "ingest signature has no chunk_ids"


def test_ingest_missing_file_exits_with_code_two(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["ingest", str(tmp_path / "does-not-exist.md"), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_ingest_directory_argument_exits_with_code_two(tmp_path: Path) -> None:
    """The CLI rejects a directory used in place of a file path."""
    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--output-dir", str(tmp_path / "runs")],
    )
    assert result.exit_code == 2
    assert "not a regular file" in result.stderr


def test_ingest_signature_matches_committed_baseline(
    tmp_path: Path,
    synthetic_doc_path: Path,
    repo_root: Path,
) -> None:
    """The CLI's ingest must produce the same signature as the canary
    baseline pinned in S-090 — proves the CLI wiring uses the same
    deterministic substrate (parser/coref/embedder/summarizer)."""
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "ingest",
            str(synthetic_doc_path),
            "--output-dir",
            str(out),
            "--doc-id",
            "aurora",
        ],
    )
    assert result.exit_code == 0

    committed = load_baseline(repo_root / "tests" / "canary" / "baselines" / "aurora__ingest.json")
    fresh = load_baseline(out / "aurora__ingest_signature.json")
    assert fresh.signature == committed.signature
    assert fresh.signature_hash == committed.signature_hash


# --- qa stub ---


def test_qa_help_describes_purpose() -> None:
    result = runner.invoke(app, ["qa", "--help"])
    assert result.exit_code == 0
    assert "trustworthy QA" in result.stdout or "QA" in result.stdout


def test_qa_blank_query_exits_with_code_two() -> None:
    result = runner.invoke(app, ["qa", "   "])
    assert result.exit_code == 2
    assert "blank" in result.stderr


def test_qa_emits_structured_stub_envelope() -> None:
    result = runner.invoke(app, ["qa", "What is Aurora?"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "qa"
    assert payload["status"] == "stub"
    assert payload["inputs"]["query"] == "What is Aurora?"
    assert "next_step" in payload
    # `anthropic_key_present` is a bool, regardless of value. Don't
    # assert on the value — the test must not depend on whether
    # ANTHROPIC_API_KEY is set in the host shell.
    assert isinstance(payload["anthropic_key_present"], bool)


# --- audit stub ---


def test_audit_requires_existing_checklist(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["audit", str(tmp_path / "missing.jsonl"), "--kind", "coverage"],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_audit_rejects_unknown_kind(tmp_path: Path) -> None:
    checklist = tmp_path / "items.jsonl"
    checklist.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["audit", str(checklist), "--kind", "bogus"])
    assert result.exit_code == 2
    assert "must be 'coverage' or 'quality'" in result.stderr


def test_audit_coverage_emits_stub(tmp_path: Path) -> None:
    checklist = tmp_path / "items.jsonl"
    checklist.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["audit", str(checklist), "--kind", "coverage"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "audit"
    assert payload["inputs"]["kind"] == "coverage"


def test_audit_quality_emits_stub(tmp_path: Path) -> None:
    checklist = tmp_path / "items.jsonl"
    checklist.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["audit", str(checklist), "--kind", "quality"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["inputs"]["kind"] == "quality"


# --- review stub ---


def test_review_blank_doc_type_exits_with_code_two() -> None:
    result = runner.invoke(app, ["review", "   "])
    assert result.exit_code == 2
    assert "blank" in result.stderr


def test_review_emits_stub_with_doc_type() -> None:
    result = runner.invoke(app, ["review", "Aurora kernel spec"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["inputs"]["doc_type"] == "Aurora kernel spec"


# --- scan ---


def test_scan_runs_deterministic_detectors() -> None:
    """An empty store yields an empty queue but the command exits OK."""
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "scan"
    assert payload["status"] == "ok"
    assert payload["findings"] == []


# --- map ---


def test_map_emits_stub_with_concepts_list() -> None:
    result = runner.invoke(app, ["map", "alpha", "beta"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["inputs"]["concepts"] == ["alpha", "beta"]


def test_map_with_no_concepts_emits_stub_with_empty_list() -> None:
    result = runner.invoke(app, ["map"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["inputs"]["concepts"] == []


# --- python -m entry point shape ---


def test_main_function_invokes_typer_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m ctrldoc` calls `main()` which calls `app()`. Verify the
    wiring without actually shelling out — `app()` raises SystemExit on
    --help, so we monkeypatch it to a sentinel."""
    from ctrldoc import cli as cli_module

    called: dict[str, bool] = {"yes": False}

    def fake_app() -> None:
        called["yes"] = True

    monkeypatch.setattr(cli_module, "app", fake_app)
    cli_module.main()
    assert called["yes"] is True
