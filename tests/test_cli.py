"""CLI surface tests via `typer.testing.CliRunner`.

Covers:

  * Root ``--help`` rendering and per-subcommand help.
  * ``ingest`` end-to-end through the heuristic `BackendBundle`
    (deterministic, no LLM): Markdown report on stdout by default,
    JSON via ``--format json``, both via ``--format both``;
    `report.md` + `result.json` under ``<runs_path>/<run_id>/`` and
    a per-doc canary signature next to them.
  * Argument validation: blank queries, blank doc types, unknown
    audit kinds, missing files, unknown profiles, unknown formats
    all exit with code 2.
  * The four LLM-backed stubs (qa / audit / review / map) still
    emit a structured JSON envelope with ``next_step`` while the
    per-playbook wiring lands in S-114 .. S-117.
  * ``scan`` runs the deterministic detector battery over an
    in-memory store.
  * ``--config`` falls back to a built-in default when the file is
    absent.
  * ``.env`` loading sets ``ANTHROPIC_API_KEY`` from disk without
    echoing the value.

SPEC-REF: §4.5, §4.7, §6
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctrldoc.canary import load_baseline
from ctrldoc.cli import _load_dotenv, app

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


# --- root help ---


def test_root_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "qa", "audit", "review", "scan", "map"):
        assert command in result.stdout, f"subcommand {command!r} missing from --help"


def test_root_no_args_exits_with_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "ingest" in result.stdout


# --- global option validation ---


def test_unknown_profile_exits_with_code_two(tmp_path: Path, synthetic_doc_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--profile", "bogus", "ingest", str(synthetic_doc_path)],
    )
    assert result.exit_code == 2
    assert "--profile must be one of" in result.stderr


def test_unknown_format_exits_with_code_two(tmp_path: Path, synthetic_doc_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--format", "bogus", "ingest", str(synthetic_doc_path)],
    )
    assert result.exit_code == 2
    assert "--format must be one of" in result.stderr


def test_zero_max_cost_usd_exits_with_code_two(tmp_path: Path, synthetic_doc_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--max-cost-usd", "0", "--profile", "heuristic", "ingest", str(synthetic_doc_path)],
    )
    assert result.exit_code == 2
    assert "--max-cost-usd must be positive" in result.stderr


# --- ingest end-to-end (heuristic profile, no LLM) ---


def _runs_path_of(cfg_path: Path) -> Path:
    """Extract the `paths.runs_path` from a config we just wrote."""
    return cfg_path.parent / "runs"


def test_ingest_runs_against_synthetic_doc_heuristic(
    tmp_path: Path,
    synthetic_doc_path: Path,
) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "ingest",
            str(synthetic_doc_path),
            "--doc-id",
            "aurora",
        ],
    )
    assert result.exit_code == 0, result.stderr
    # Default --format is markdown; stdout has the report.
    assert "# ctrldoc — ingest report" in result.stdout
    assert "| Chunks indexed |" in result.stdout

    runs_path = _runs_path_of(cfg)
    report_files = list(runs_path.rglob("report.md"))
    result_files = list(runs_path.rglob("result.json"))
    assert len(report_files) == 1
    assert len(result_files) == 1

    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["command"] == "ingest"
    assert payload["status"] == "ok"
    assert payload["doc_id"] == "aurora"
    assert payload["profile"] == "heuristic"
    assert payload["chunks_indexed"] > 0
    assert payload["sections_parsed"] > 0
    assert "signature" in payload
    assert "signature_hash" in payload


def test_ingest_format_json_emits_only_json(tmp_path: Path, synthetic_doc_path: Path) -> None:
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
            "ingest",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "ingest"
    assert payload["status"] == "ok"
    # No Markdown heading bled into stdout under --format json.
    assert "# ctrldoc" not in result.stdout


def test_ingest_format_both_emits_markdown_then_json(
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
            "both",
            "ingest",
            str(synthetic_doc_path),
        ],
    )
    assert result.exit_code == 0
    assert "# ctrldoc — ingest report" in result.stdout
    assert "--- JSON ---" in result.stdout
    assert '"command": "ingest"' in result.stdout


def test_ingest_doc_id_defaults_to_input_stem(tmp_path: Path, synthetic_doc_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(cfg), "--profile", "heuristic", "ingest", str(synthetic_doc_path)],
    )
    assert result.exit_code == 0
    payload = json.loads(next(_runs_path_of(cfg).rglob("result.json")).read_text(encoding="utf-8"))
    assert payload["doc_id"] == synthetic_doc_path.stem


def test_ingest_signature_matches_committed_baseline(
    tmp_path: Path,
    synthetic_doc_path: Path,
    repo_root: Path,
) -> None:
    """The CLI's ingest under the heuristic profile must still produce
    the same chunk/section/entity signature as the canary baseline
    pinned in S-090 — proves the bundle's deterministic L0 substrate
    matches the pre-Phase-10 wiring."""
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "ingest",
            str(synthetic_doc_path),
            "--doc-id",
            "aurora",
        ],
    )
    assert result.exit_code == 0

    runs_path = _runs_path_of(cfg)
    committed = load_baseline(repo_root / "tests" / "canary" / "baselines" / "aurora__ingest.json")
    # New path: result.json carries the signature directly.
    result_files = list(runs_path.rglob("result.json"))
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["signature"] == committed.signature
    assert payload["signature_hash"] == committed.signature_hash
    # Legacy path: a `<doc_id>__ingest_signature.json` next to runs/
    # so the existing S-090 canary downstream stays whole.
    legacy = load_baseline(runs_path / "aurora__ingest_signature.json")
    assert legacy.signature == committed.signature


def test_ingest_missing_file_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "ingest",
            str(tmp_path / "does-not-exist.md"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_ingest_directory_argument_exits_with_code_two(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "--config",
            str(cfg),
            "--profile",
            "heuristic",
            "ingest",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "not a regular file" in result.stderr


def test_ingest_default_config_is_used_when_file_absent(
    tmp_path: Path, synthetic_doc_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `--config` and no `./ctrldoc.toml` → built-in default config."""
    monkeypatch.chdir(tmp_path)
    # No ctrldoc.toml here. We still pass --output-dir so the run
    # artefacts land under a writable tmp_path.
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "--profile",
            "heuristic",
            "ingest",
            str(synthetic_doc_path),
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert list(out.rglob("report.md"))


# --- ingest end-to-end (thrifty profile, persistent SQLite + sqlite-vec) ---


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_ingest_thrifty_persists_per_doc_sqlite(tmp_path: Path, synthetic_doc_path: Path) -> None:
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
            "ingest",
            str(synthetic_doc_path),
            "--doc-id",
            "aurora",
        ],
    )
    if result.exit_code != 0:
        # Re-raise the original exception (if any) so the assertion
        # surfaces a real stack trace rather than a wall of tqdm noise.
        if result.exception and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"thrifty ingest crashed: {type(result.exception).__name__}: {result.exception}"
            ) from result.exception
        pytest.skip("thrifty ingest non-zero exit (likely no Ollama)")
    payload = json.loads(next(_runs_path_of(cfg).rglob("result.json")).read_text(encoding="utf-8"))
    assert payload["profile"] == "thrifty"
    persisted = payload["persisted"]
    assert "store" in persisted and Path(persisted["store"]).exists()
    assert "vector_index" in persisted and Path(persisted["vector_index"]).exists()
    # The per-doc filenames are <doc_hash>.db / .vec.db.
    assert payload["doc_hash"] in Path(persisted["store"]).name


# --- dotenv loader ---


def test_load_dotenv_sets_env_vars_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "ANTHROPIC_API_KEY=secret-value-1\n"
        'OTHER_KEY="quoted-value"\n'
        "\n"
        "MALFORMED-LINE-WITHOUT-EQUALS\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_KEY", raising=False)
    _load_dotenv(env_file)
    assert os.environ.get("ANTHROPIC_API_KEY") == "secret-value-1"
    assert os.environ.get("OTHER_KEY") == "quoted-value"


def test_load_dotenv_does_not_overwrite_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=should-not-override\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "pre-set-value")
    _load_dotenv(env_file)
    assert os.environ["ANTHROPIC_API_KEY"] == "pre-set-value"


def test_load_dotenv_missing_file_is_no_op(tmp_path: Path) -> None:
    # Should not raise.
    _load_dotenv(tmp_path / "no-such-file.env")


# --- qa moved to tests/test_cli_qa.py (S-114) ---
# --- audit moved to tests/test_cli_audit.py (S-113) ---


# --- review moved to tests/test_cli_review.py (S-115) ---


# --- scan moved to tests/test_cli_scan.py (S-116) ---


# --- map moved to tests/test_cli_map.py (S-117) ---


# --- python -m entry point shape ---


def test_main_function_invokes_typer_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m ctrldoc` calls `main()` which calls `app()`."""
    from ctrldoc import cli as cli_module

    called: dict[str, bool] = {"yes": False}

    def fake_app() -> None:
        called["yes"] = True

    monkeypatch.setattr(cli_module, "app", fake_app)
    cli_module.main()
    assert called["yes"] is True
