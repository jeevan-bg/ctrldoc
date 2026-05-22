"""Static checks that the build toolchain is wired correctly.

These tests lock the declared toolchain to the contract the rest of the
build relies on: a strict-mypy environment, a configured ruff lint set,
and a pytest marker for every test family in SPEC §8.6.

SPEC-REF: §12 (Build Order), §4.7 (cross-cutting), §8.6 (test families)
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


REQUIRED_DEV_TOOLS = (
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "hypothesis",
    "ruff",
    "mypy",
    "pre-commit",
)


def test_required_python_version(pyproject: dict) -> None:
    requires = pyproject["project"]["requires-python"]
    assert requires.startswith(">=3.11"), f"unexpected requires-python: {requires!r}"


def test_required_dev_dependencies_declared(pyproject: dict) -> None:
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
    declared = {dep.split(">=")[0].split("==")[0].split("[")[0].strip() for dep in dev_deps}
    missing = sorted(set(REQUIRED_DEV_TOOLS) - declared)
    assert not missing, f"missing dev dependencies in pyproject: {missing}"


def test_mypy_is_strict(pyproject: dict) -> None:
    mypy_cfg = pyproject["tool"]["mypy"]
    assert mypy_cfg.get("strict") is True, "mypy must run in strict mode"
    assert mypy_cfg["python_version"] == "3.11"


def test_ruff_lint_set_present(pyproject: dict) -> None:
    ruff_lint = pyproject["tool"]["ruff"]["lint"]
    selected = set(ruff_lint["select"])
    for code in ("E", "F", "W", "I", "B"):
        assert code in selected, f"ruff lint set missing {code!r}"


SPEC_FAMILY_MARKERS = tuple(
    f"family_{name}"
    for name in (
        "ingest_completeness",
        "niah",
        "synthetic_gold",
        "reachability",
        "negative_refusal",
        "referential_integrity",
        "robustness",
        "adversarial",
        "verifier_calibration",
        "determinism",
        "performance_cost",
        "failure_resilience",
        "incremental_update",
        "concurrency",
    )
)


def test_every_spec_family_has_pytest_marker(pyproject: dict) -> None:
    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    marker_names = {m.split(":", 1)[0].strip() for m in markers}
    missing = [m for m in SPEC_FAMILY_MARKERS if m not in marker_names]
    assert not missing, f"pytest markers missing for SPEC §8.6 families: {missing}"
    assert len(SPEC_FAMILY_MARKERS) == 14, "SPEC §8.6 defines exactly 14 families"


def test_leak_scan_script_executable() -> None:
    script = REPO_ROOT / "scripts" / "leak_scan.sh"
    assert script.exists(), "scripts/leak_scan.sh is missing"
    assert os.access(script, os.X_OK), "scripts/leak_scan.sh must be executable"


def test_ctrldoc_package_layout() -> None:
    src_pkg = REPO_ROOT / "src" / "ctrldoc" / "__init__.py"
    assert src_pkg.exists(), "src/ctrldoc/__init__.py missing"
    import ctrldoc

    assert hasattr(ctrldoc, "__version__")
