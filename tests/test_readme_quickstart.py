"""README quickstart — verified end-to-end against the bundled doc.

The README's "Quickstart" section walks a new user through three
commands:

  1. ``ctrldoc ingest tests/fixtures/synthetic/gold_doc.md --output-dir ./runs --doc-id aurora``
  2. ``ctrldoc scan``
  3. ``ctrldoc --help``

This file runs each command via `typer.testing.CliRunner` against
the synthetic doc bundled in the repo. A regression in the README's
flag names, file paths, or expected outputs will fail one of these
tests at the same commit the README drifted.

SPEC-REF: §6, §12
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctrldoc.cli import app

runner = CliRunner()


def _readme_text(repo_root: Path) -> str:
    return (repo_root / "README.md").read_text(encoding="utf-8")


def _extract_quickstart_commands(readme: str) -> list[list[str]]:
    """Pull every `ctrldoc …` invocation out of the Quickstart fenced block.

    The Quickstart section opens with a triple-backtick code fence
    immediately under the `## Quickstart` heading. We pull that one
    block, then collect every line that starts (after `#`-comment
    stripping) with `ctrldoc `.
    """
    # Find the Quickstart heading and the first fenced block after it.
    heading = re.search(r"^##\s+Quickstart", readme, flags=re.MULTILINE)
    assert heading is not None, "Quickstart heading not found in README"
    block = re.search(
        r"```bash\n(.*?)\n```",
        readme[heading.end() :],
        flags=re.DOTALL,
    )
    assert block is not None, "No ```bash block under Quickstart heading"
    body = block.group(1)
    # Join `\<newline>` continuations into one logical line.
    joined = body.replace("\\\n", " ")
    commands: list[list[str]] = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if tokens and tokens[0] == "ctrldoc":
            commands.append(tokens[1:])  # drop the `ctrldoc` prefix
    return commands


# --- one command per quickstart step ---


def test_readme_quickstart_section_exists(repo_root: Path) -> None:
    readme = _readme_text(repo_root)
    assert "## Quickstart" in readme


def test_readme_quickstart_step_1_ingest_succeeds(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Step 1: ingest the synthetic doc through the CLI."""
    result = runner.invoke(
        app,
        [
            "ingest",
            str(repo_root / "tests" / "fixtures" / "synthetic" / "gold_doc.md"),
            "--output-dir",
            str(tmp_path / "runs"),
            "--doc-id",
            "aurora",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "ingest"
    assert payload["status"] == "ok"
    assert payload["chunks_indexed"] > 0
    assert (tmp_path / "runs" / "aurora__ingest_stats.json").is_file()
    assert (tmp_path / "runs" / "aurora__ingest_signature.json").is_file()


def test_readme_quickstart_step_2_scan_succeeds() -> None:
    """Step 2: run the deterministic anomaly scan."""
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "scan"
    assert payload["status"] == "ok"
    assert isinstance(payload["findings"], list)


def test_readme_quickstart_step_3_help_lists_subcommands() -> None:
    """Step 3: `ctrldoc --help` advertises every UC subcommand."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("ingest", "qa", "audit", "review", "scan", "map"):
        assert sub in result.stdout


# --- README invariants ---


def test_readme_quickstart_extraction_recovers_each_step(repo_root: Path) -> None:
    """The extraction helper must pull at least three ctrldoc invocations."""
    commands = _extract_quickstart_commands(_readme_text(repo_root))
    assert len(commands) >= 3, f"only extracted {len(commands)} commands: {commands}"


def test_every_quickstart_command_runs_successfully(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Run every ctrldoc invocation parsed from the Quickstart block.

    `--output-dir ./runs` is rewritten to a temp dir so the test
    doesn't pollute the repo. Any non-zero exit code fails — keeping
    the README honest as the CLI surface evolves.
    """
    commands = _extract_quickstart_commands(_readme_text(repo_root))
    out_dir = tmp_path / "runs"
    for args in commands:
        # Redirect any `--output-dir ./runs` to the tmp_path so the
        # test is hermetic.
        rewritten = [str(out_dir) if arg == "./runs" else arg for arg in args]
        # Skip --help since CliRunner returns exit_code=0 but reads
        # from a different code path; we test it separately.
        if "--help" in rewritten:
            result = runner.invoke(app, rewritten)
            assert result.exit_code == 0
            continue
        result = runner.invoke(app, rewritten)
        assert result.exit_code == 0, (
            f"command {rewritten} failed with code {result.exit_code}\nstderr: {result.stderr}"
        )


# --- README content sanity ---


def test_readme_references_examples_directory(repo_root: Path) -> None:
    readme = _readme_text(repo_root)
    assert "examples/" in readme


def test_readme_references_synthetic_gold_doc_path(repo_root: Path) -> None:
    """The quickstart relies on the bundled doc; if its path changes
    the README needs an edit."""
    readme = _readme_text(repo_root)
    assert "tests/fixtures/synthetic/gold_doc.md" in readme


def test_readme_lists_every_uc_playbook(repo_root: Path) -> None:
    readme = _readme_text(repo_root)
    for marker in (
        "qa",
        "coverage_audit",
        "quality_audit",
        "analytical_review",
        "anomaly_scan",
        "relation_map",
    ):
        assert marker in readme, f"README missing reference to playbook {marker!r}"


@pytest.fixture
def repo_root_resolved(repo_root: Path) -> Path:
    """Re-export the conftest `repo_root` fixture under a name unique to
    this module — defensive aliasing so the test stays robust against
    future conftest refactors."""
    return repo_root
