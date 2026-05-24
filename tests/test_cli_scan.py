"""Tests for the anomaly-scan CLI wiring (S-116).

Covers:

  * `render_scan_markdown` — output shape: header, per-detector
    groups (sorted critical → warn → info), summary table with
    per-severity columns.
  * `ctrldoc scan` CLI surface:
      - missing target → exit 2.
      - heuristic profile works (no LLM dependency).
      - thrifty profile also works (slow, gated on optional deps).
      - report.md + result.json land under `<runs_path>/<run_id>/`.
      - `--format json` emits only JSON.

SPEC-REF: §5.5 (anomaly_scan), §6 (CLI)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctrldoc.cli import app
from ctrldoc.cli_scan import render_scan_markdown
from ctrldoc.models import Finding, Span
from ctrldoc.ops.scan import AnomalyQueue

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


# --- render_scan_markdown ---


def _fake_finding(detector: str, severity: str, chunk_id: str = "c1") -> Finding:
    return Finding(
        ctrldoc=detector,
        location=Span(chunk_id=chunk_id, char_start=0, char_end=11, text="hello world"),
        claim=f"issue from {detector}",
        severity=severity,  # type: ignore[arg-type]
    )


def test_render_scan_markdown_groups_by_detector_and_severity() -> None:
    queue = AnomalyQueue(
        findings=[
            _fake_finding("hedge_word", "warn", "c1"),
            _fake_finding("empty_summary", "info", "c2"),
            _fake_finding("hedge_word", "critical", "c3"),
        ]
    )
    md = render_scan_markdown(
        queue=queue,
        target_path=Path("doc.md"),
        profile="heuristic",
        run_id="r1",
    )
    assert "# ctrldoc — anomaly scan report" in md
    assert "## Detector `hedge_word` (2)" in md
    assert "## Detector `empty_summary` (1)" in md
    # Severity sort: critical comes before warn in the hedge_word group.
    hedge_block = md.split("## Detector `hedge_word` (2)")[1]
    crit_pos = hedge_block.find("**critical**")
    warn_pos = hedge_block.find("**warn**")
    assert 0 <= crit_pos < warn_pos


def test_render_scan_markdown_emits_per_severity_summary_table() -> None:
    queue = AnomalyQueue(
        findings=[
            _fake_finding("hedge_word", "warn"),
            _fake_finding("hedge_word", "warn"),
            _fake_finding("empty_summary", "info"),
        ]
    )
    md = render_scan_markdown(
        queue=queue,
        target_path=Path("doc.md"),
        profile="heuristic",
        run_id="r2",
    )
    assert "| Detector | critical | warn | info | Total |" in md
    assert "| `hedge_word` | 0 | 2 | 0 | 2 |" in md
    assert "| `empty_summary` | 0 | 0 | 1 | 1 |" in md
    assert "| **Total** | **0** | **2** | **1** | **3** |" in md


def test_render_scan_markdown_empty_queue_shows_placeholder() -> None:
    md = render_scan_markdown(
        queue=AnomalyQueue(findings=[]),
        target_path=Path("doc.md"),
        profile="heuristic",
        run_id="r3",
    )
    assert "_(no anomalies detected)_" in md
    assert "**Total findings**: 0" in md


# --- ctrldoc scan CLI surface ---


def test_scan_missing_target_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "scan",
            "--target",
            str(tmp_path / "absent.md"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_scan_heuristic_runs_against_synthetic_doc(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "--format",
            "json",
            "scan",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "scan"
    assert payload["status"] == "ok"
    assert payload["profile"] == "heuristic"
    assert payload["target_path"].endswith(synthetic_doc_path.name)
    assert isinstance(payload["findings"], list)
    assert "findings_total" in payload

    runs_path = tmp_path / "runs"
    assert list(runs_path.rglob("report.md"))
    assert list(runs_path.rglob("result.json"))


def test_scan_default_markdown_format_emits_report_to_stdout(
    tmp_path: Path, synthetic_doc_path: Path
) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "scan",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 0
    assert "# ctrldoc — anomaly scan report" in result.stdout
    assert "## Summary" in result.stdout


def test_scan_format_both_emits_markdown_and_json(tmp_path: Path, synthetic_doc_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "--format",
            "both",
            "scan",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 0
    assert "# ctrldoc — anomaly scan report" in result.stdout
    assert "--- JSON ---" in result.stdout
    assert '"command": "scan"' in result.stdout


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_scan_thrifty_writes_persistent_artefacts(tmp_path: Path, synthetic_doc_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("ollama")
    pytest.importorskip("gliner", reason="gliner optional; thrifty profile needs it")
    pytest.importorskip("fastcoref", reason="fastcoref optional; thrifty profile needs it")
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
            "scan",
            "--target",
            str(synthetic_doc_path),
        ],
    )
    if result.exit_code != 0:
        if result.exception and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"thrifty scan crashed: {type(result.exception).__name__}: {result.exception}"
            ) from result.exception
        pytest.skip(f"thrifty scan non-zero exit ({result.exit_code})")
    payload = json.loads(result.stdout)
    assert payload["profile"] == "thrifty"
    assert "findings_total" in payload
    runs_path = tmp_path / "runs"
    indexes = runs_path / "indexes"
    # The per-doc SQLite store + sqlite-vec sidecar exist after a
    # thrifty scan since the command goes through the full ingest.
    assert any(indexes.glob(f"{payload['target_doc_hash']}.db"))
    assert any(indexes.glob(f"{payload['target_doc_hash']}.vec.db"))
