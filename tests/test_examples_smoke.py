"""Smoke tests for every file in `examples/`.

Each example is a runnable Python script. The smoke test imports the
module, calls its `main()` entry point, and asserts the stdout is
valid JSON whose top-level structure matches the playbook's
expected payload shape. This keeps the examples honest as the
underlying playbook APIs evolve.

SPEC-REF: §6, §12
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _import_example(filename: str) -> Any:
    """Import `examples/<filename>` as a module under a synthetic name."""
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_example_{filename[:-3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Make the module discoverable to itself.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_and_capture(filename: str) -> dict[str, Any]:
    """Run the example's `main()` and parse its stdout as JSON."""
    module = _import_example(filename)
    buf = io.StringIO()
    with redirect_stdout(buf):
        module.main()
    raw = buf.getvalue().strip()
    return json.loads(raw)


# --- per-example smokes ---


def test_qa_example_returns_verified_answer() -> None:
    payload = _run_and_capture("01_qa.py")
    assert payload["answer"]
    assert payload["claims"]
    assert payload["claims"][0]["verified"] is True
    assert payload["claims"][0]["citations"]


def test_coverage_audit_example_runs_three_clusters() -> None:
    payload = _run_and_capture("02_coverage_audit.py")
    assert payload["batched_calls"] == 3
    verdicts = {v["item_id"]: v["verdict"] for v in payload["verdicts"]}
    assert verdicts["r-hashing"] == "Covered"
    assert verdicts["r-failover"] == "Covered"
    assert verdicts["r-license"] == "NotCovered"


def test_quality_audit_example_delegates_to_coverage() -> None:
    payload = _run_and_capture("03_quality_audit.py")
    assert payload["doc_type"] == "L0 kernel spec"
    # HeuristicCriteriaGenerator emits four base axes.
    assert len(payload["criteria"]) == 4
    # Every criterion is in a distinct topic_key → four batched calls.
    assert payload["batched_calls"] == 4


def test_analytical_review_example_aggregates_findings() -> None:
    payload = _run_and_capture("04_analytical_review.py")
    # Five canonical lenses → five findings → synthesised narrative.
    assert len(payload["findings"]) == 5
    lens_names = {f["lens"] for f in payload["findings"]}
    assert {"assumptions", "boundary_cases", "consistency", "ambiguity", "scope_gaps"} == lens_names
    assert payload["narrative"]["headline"]


def test_anomaly_scan_example_detects_hedge_words_and_blank_summaries() -> None:
    payload = _run_and_capture("05_anomaly_scan.py")
    detectors = {f["detector"] for f in payload["findings"]}
    assert "hedge_word" in detectors
    assert "empty_summary" in detectors
    # The blank-summary section "sec/sec" is the one flagged.
    blank_findings = [f for f in payload["findings"] if f["detector"] == "empty_summary"]
    assert any(f["chunk_id"] == "sec/sec" for f in blank_findings)


def test_relation_map_example_emits_three_edges() -> None:
    payload = _run_and_capture("06_relation_map.py")
    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 3
    types = {e["type"] for e in payload["edges"]}
    assert types == {"depends_on", "refines"}


# --- meta: every example follows the same shape ---


@pytest.mark.parametrize(
    "filename",
    [
        "01_qa.py",
        "02_coverage_audit.py",
        "03_quality_audit.py",
        "04_analytical_review.py",
        "05_anomaly_scan.py",
        "06_relation_map.py",
    ],
)
def test_example_main_returns_valid_json(filename: str) -> None:
    """Every example's `main()` must print a single JSON object."""
    payload = _run_and_capture(filename)
    assert isinstance(payload, dict), f"{filename} did not emit a JSON object"


def test_readme_lists_every_example() -> None:
    """The README must reference every runnable file in `examples/`."""
    readme = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    for filename in EXAMPLES_DIR.glob("*.py"):
        assert filename.name in readme, f"README missing reference to {filename.name}"
