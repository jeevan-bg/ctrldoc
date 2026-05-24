"""eval_v1 baseline scripts — wiring smoke for the five v1 eval substrates.

The `scripts/eval_v1_*.py` modules each load one of the v1 eval JSONL
fixtures, drive a deterministic stub through that substrate's
`EvalRunner`, and emit a single-line JSON summary on stdout. The
summary line is the contract `scripts/run_v1_smoke.sh` aggregates;
this test file pins:

1. Each baseline `main(argv=None)` exits 0 on the shipped fixture and
   prints exactly one JSON line carrying `set_name`, `cases`,
   `passed`, and an `aggregate` dict including the substrate's
   primary metric.
2. The stub for each substrate is a degenerate Protocol implementation
   — its job is to exercise harness wiring end-to-end, not to clear
   any release gate. A baseline that suddenly starts clearing the
   gate would mean the harness has been silently mocked.
3. `scripts/run_v1_smoke.sh` invokes all five baselines and exits 0
   when every one of them prints a parseable JSON line.

The shell aggregator is exercised via `subprocess` so the test pins
the integration contract that downstream slices (S-125+) and the
release smoke will rely on.

SPEC-REF: §6 (universal substrate), §14 (eval substrates)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

_BASELINE_MODULES = (
    ("eval_v1_claim_extraction", "claim_extraction"),
    ("eval_v1_cross_doc_coverage", "cross_doc_coverage"),
    ("eval_v1_compare", "compare"),
    ("eval_v1_merge", "merge"),
    ("eval_v1_calibration", "calibration"),
)


def _run_script(module_name: str) -> tuple[int, str, str]:
    """Run `python scripts/<module_name>.py` and return (rc, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / f"{module_name}.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.mark.parametrize(("module_name", "set_name"), _BASELINE_MODULES)
def test_baseline_script_emits_one_json_line(module_name: str, set_name: str) -> None:
    rc, stdout, stderr = _run_script(module_name)
    assert rc == 0, f"{module_name} exited {rc}; stderr=\n{stderr}"
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert (
        len(lines) == 1
    ), f"{module_name} must print exactly one JSON summary line; got {len(lines)} lines"
    payload = json.loads(lines[0])
    assert payload["set_name"] == set_name
    assert isinstance(payload["cases"], int)
    assert payload["cases"] >= 1
    assert isinstance(payload["passed"], bool)
    assert isinstance(payload["aggregate"], dict)
    # Every harness aggregate ships pass_rate and score.
    assert "pass_rate" in payload["aggregate"]
    assert "score" in payload["aggregate"]


def test_baseline_scripts_report_below_release_gate() -> None:
    """Trivial stubs must NOT clear the substrate's release gate.

    A baseline that claims `passed=True` would mean the harness is
    being short-circuited or the stub is doing real work. Both are
    silent regressions in the eval substrate. This test guards
    against that by requiring every shipped baseline to fail its
    gate on the fixture data.
    """
    for module_name, _set_name in _BASELINE_MODULES:
        rc, stdout, _ = _run_script(module_name)
        assert rc == 0
        payload = json.loads(stdout.splitlines()[0])
        assert payload["passed"] is False, (
            f"{module_name} baseline unexpectedly passed — "
            "the stub Protocol implementation may have grown real behaviour"
        )


def test_run_v1_smoke_exits_zero_and_summarizes_all_five() -> None:
    smoke = SCRIPTS_DIR / "run_v1_smoke.sh"
    proc = subprocess.run(
        ["bash", str(smoke)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert (
        proc.returncode == 0
    ), f"run_v1_smoke.sh exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    # Every substrate's set_name must appear in the aggregator's output.
    for _module_name, set_name in _BASELINE_MODULES:
        assert (
            set_name in proc.stdout
        ), f"smoke aggregator stdout missing substrate {set_name!r}; got:\n{proc.stdout}"


def test_run_v1_smoke_fails_when_a_baseline_missing(tmp_path: Path) -> None:
    """If a baseline script is removed, the aggregator must exit non-zero.

    This pins the contract that the smoke runner is a wiring check —
    silent skips would defeat its purpose. We exercise it by writing
    a copy of the shell script that points at a nonexistent baseline
    and confirming the failure surfaces.
    """
    smoke_src = (SCRIPTS_DIR / "run_v1_smoke.sh").read_text(encoding="utf-8")
    broken_smoke = tmp_path / "broken_smoke.sh"
    broken_smoke.write_text(
        smoke_src.replace("eval_v1_claim_extraction.py", "eval_v1_missing_substrate.py"),
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(broken_smoke)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode != 0, (
        "run_v1_smoke.sh must exit non-zero when a baseline script is missing; "
        f"got rc=0 with stdout:\n{proc.stdout}"
    )
