"""v1.0.0 release artifacts gate.

The final v1 release slice tags `v1.0.0`. This test fails until every
release artifact is in place — version pinned, changelog entry,
migration guide, README rewrite, runnable v1 examples — so a future
regression that loses any one of them is caught immediately.

SPEC-REF: §16
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_VERSION = "1.0.0"

V1_EXAMPLES_DIR = REPO_ROOT / "examples" / "v1"
MIGRATION_GUIDE = REPO_ROOT / "MIGRATION_v0.3_to_v1.0.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
README = REPO_ROOT / "README.md"
ARCHITECTURE = REPO_ROOT / "docs" / "ARCHITECTURE.md"
SPEC_TRACE = REPO_ROOT / "docs" / "SPEC_TRACE.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_INIT = REPO_ROOT / "src" / "ctrldoc" / "__init__.py"


# --------------------------------------------------------------------------
# Version bump
# --------------------------------------------------------------------------


def test_pyproject_pins_v1_release_version() -> None:
    """`pyproject.toml` must declare the v1.0.0 version."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["version"] == EXPECTED_VERSION


def test_package_dunder_version_matches_pyproject() -> None:
    """`ctrldoc.__version__` must match the pyproject pin."""
    import ctrldoc

    assert ctrldoc.__version__ == EXPECTED_VERSION


# --------------------------------------------------------------------------
# CHANGELOG
# --------------------------------------------------------------------------


def test_changelog_has_v1_release_section() -> None:
    """CHANGELOG must carry a `## [1.0.0]` heading dated this year."""
    text = CHANGELOG.read_text(encoding="utf-8")
    pattern = re.compile(r"^## \[1\.0\.0\]\s+—\s+\d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)
    assert (
        pattern.search(text) is not None
    ), "CHANGELOG.md must declare a `## [1.0.0] — YYYY-MM-DD` release section"


def test_changelog_v1_section_mentions_substrate_surface() -> None:
    """The v1 release notes must call out the universal-substrate surface."""
    text = CHANGELOG.read_text(encoding="utf-8")
    # Pull the v1.0.0 section through to the next `## [` heading.
    match = re.search(r"^## \[1\.0\.0\].*?(?=^## \[)", text, re.MULTILINE | re.DOTALL)
    assert match is not None, "missing v1.0.0 CHANGELOG section"
    section = match.group(0).lower()
    for keyword in (
        "workspace",
        "mcp",
        "coverage",
        "calibration",
        "ledger",
    ):
        assert keyword in section, (
            f"v1.0.0 CHANGELOG section must mention '{keyword}' — "
            "the user-visible v1 surface is incomplete in the release notes"
        )


# --------------------------------------------------------------------------
# Migration guide
# --------------------------------------------------------------------------


def test_migration_guide_exists() -> None:
    assert (
        MIGRATION_GUIDE.exists()
    ), f"migration guide missing at {MIGRATION_GUIDE.relative_to(REPO_ROOT)}"


def test_migration_guide_covers_breaking_changes() -> None:
    """Migration guide must walk readers through every v0.3 → v1 break."""
    text = MIGRATION_GUIDE.read_text(encoding="utf-8").lower()
    required_topics = (
        "schema_version",  # storage schema bump 0.1.0 → 0.2.0
        "re-ingest",  # no in-place migration path
        "playbooks",  # package removed
        "ops",  # symbols re-homed under ctrldoc.ops.*
        "workspace",  # new L2.5 primitive
        "mcp",  # new MCP server surface
    )
    for topic in required_topics:
        assert topic in text, (
            f"migration guide must mention '{topic}' so callers know "
            "what to do when upgrading from v0.3"
        )


# --------------------------------------------------------------------------
# README rewrite — must reflect the v1 surface
# --------------------------------------------------------------------------


def test_readme_advertises_v1_release_status() -> None:
    text = README.read_text(encoding="utf-8")
    assert (
        "v1.0.0" in text or "1.0.0" in text
    ), "README must advertise the v1.0.0 release in its Status section"


def test_readme_lists_universal_substrate_operations() -> None:
    """README must name the v1 ops the user can actually run."""
    text = README.read_text(encoding="utf-8").lower()
    for op in ("workspace", "coverage", "compare", "merge", "mcp"):
        assert op in text, (
            f"README must mention the v1 '{op}' operation in the user-facing "
            "surface — readers landing on the page see the v0.3 playbooks "
            "otherwise"
        )


# --------------------------------------------------------------------------
# examples/v1/ runnable walkthroughs
# --------------------------------------------------------------------------


def _v1_example_scripts() -> list[Path]:
    if not V1_EXAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in V1_EXAMPLES_DIR.glob("*.py") if p.name != "__init__.py")


def test_v1_examples_dir_exists_with_runnable_walkthroughs() -> None:
    """`examples/v1/` must ship at least three runnable scripts."""
    scripts = _v1_example_scripts()
    assert len(scripts) >= 3, (
        f"examples/v1/ must ship ≥ 3 runnable v1 walkthroughs "
        f"(found {len(scripts)}: {[p.name for p in scripts]})"
    )


def test_v1_examples_import_only_v1_substrate_surface() -> None:
    """Every v1 example must import from `ctrldoc.ops.*`, never v0.3 playbooks."""
    scripts = _v1_example_scripts()
    assert scripts, "no v1 example scripts to scan"
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
        # The deprecated package was removed in S-146; no v1 example may
        # reach for it.
        for module in imported_modules:
            assert "playbooks" not in module, (
                f"{path.name}: imports deprecated module {module!r}; "
                "v1 examples must drive the universal substrate "
                "(ctrldoc.ops.*, ctrldoc.mcp, ctrldoc.orch.ledger)"
            )
        # Every example must actually exercise *some* v1 surface.
        joined = " ".join(imported_modules)
        assert any(
            marker in joined
            for marker in (
                "ctrldoc.ops",
                "ctrldoc.mcp",
                "ctrldoc.orch",
                "ctrldoc.extract",
                "ctrldoc.retrieval",
            )
        ), (
            f"{path.name} does not import any v1 substrate module; "
            "if the example is non-functional it does not belong in examples/v1/"
        )


@pytest.mark.parametrize(
    "script_path",
    _v1_example_scripts() or [pytest.param(None, id="(no scripts)")],
    ids=lambda p: p.name if isinstance(p, Path) else "(no scripts)",
)
def test_v1_examples_execute_cleanly(script_path: Path | None) -> None:
    """Each v1 example must run from a clean clone with zero LLM credentials."""
    if script_path is None:
        pytest.fail("no v1 example scripts to execute")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"v1 example {script_path.name} exited non-zero:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    # Examples are walkthroughs — they must print something to stdout so
    # a reader sees the result. A silent example is a broken example.
    assert result.stdout.strip(), (
        f"v1 example {script_path.name} produced no stdout output; "
        "examples must surface their result to the reader"
    )


# --------------------------------------------------------------------------
# SPEC_TRACE row for the release slice
# --------------------------------------------------------------------------


def test_spec_trace_carries_release_slice_row() -> None:
    """SPEC_TRACE must record S-147 as covered against §16."""
    text = SPEC_TRACE.read_text(encoding="utf-8")
    assert "S-147" in text, "docs/SPEC_TRACE.md must carry a row for the v1 release slice"


# --------------------------------------------------------------------------
# Architecture refresh
# --------------------------------------------------------------------------


def test_architecture_doc_covers_v1_layers() -> None:
    """ARCHITECTURE.md must describe the v1 L1.5 / L2.5 / MCP additions."""
    text = ARCHITECTURE.read_text(encoding="utf-8").lower()
    for marker in ("claim graph", "workspace", "mcp"):
        assert marker in text, (
            f"docs/ARCHITECTURE.md must describe the v1 '{marker}' surface — "
            "the doc otherwise still reads as v0.3"
        )


# --------------------------------------------------------------------------
# Real-doc smoke (release gate)
# --------------------------------------------------------------------------


def test_real_doc_smoke_script_is_executable_release_gate() -> None:
    """The §16 real-doc smoke script must remain importable and executable.

    The full hermetic run is exercised in tests/test_real_doc_smoke.py;
    this test is the cheap release-gate guard that catches a vanished
    script or a broken module-level import without paying the ~20-second
    cost of running the whole corpus.
    """
    smoke_module = "ctrldoc.eval.real_doc_smoke"
    result = subprocess.run(
        [sys.executable, "-c", f"import {smoke_module}; print('ok')"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, f"real-doc smoke module fails to import:\n{result.stderr}"
    assert result.stdout.strip() == "ok"


# --------------------------------------------------------------------------
# ECE release gate
# --------------------------------------------------------------------------


def test_calibration_ece_release_gate_constant_is_pinned() -> None:
    """The §6.5 release gate `ECE ≤ 0.05` must remain encoded in code."""
    from ctrldoc.extract.isotonic_calibration import (
        CALIBRATION_ECE_THRESHOLD,
        ece_within_release_gate,
    )

    assert CALIBRATION_ECE_THRESHOLD == 0.05
    assert ece_within_release_gate(0.04) is True
    assert ece_within_release_gate(0.05) is True
    assert ece_within_release_gate(0.06) is False
