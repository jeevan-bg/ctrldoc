"""Baseline measurement for the calibration eval substrate.

Drives a degenerate `CalibrationScorer` (returns a uniform softmax
1/3, 1/3, 1/3 for every premise/hypothesis pair) through
`CalibrationEvalRunner` against `tests/eval/calibration_eval.jsonl`
and prints a one-line JSON summary on stdout. The uniform-distribution
baseline exercises the substrate end-to-end: the per-case argmax
tiebreak rule resolves to `entailment` (documented in `NLIScore`),
so label accuracy equals the entailment prior; per-case ECE is the
gap between bin accuracy and the constant top-confidence 1/3, well
above the 0.05 release gate.

The baseline is a wiring check, not a contender for the §6.5 release
gate. A baseline that suddenly clears either gate means the harness
has been silently mocked.

SPEC-REF: §6.5 (probabilistic edges + calibration), §14
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ctrldoc.eval.calibration import (
    CALIBRATION_ACCURACY_THRESHOLD,
    CALIBRATION_ECE_THRESHOLD,
    CalibrationEvalCase,
    CalibrationEvalRunner,
    NLIScore,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "tests" / "eval" / "calibration_eval.jsonl"
SET_NAME = "calibration"

_UNIFORM = NLIScore(
    entailment=1.0 / 3.0,
    contradiction=1.0 / 3.0,
    neutral=1.0 / 3.0,
)


class _UniformScorer:
    """Degenerate baseline — returns the uniform softmax for every pair."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        del premise, hypothesis
        return _UNIFORM


def main(argv: list[str] | None = None) -> int:
    del argv
    cases = load_jsonl_cases(EVAL_PATH, case_model=CalibrationEvalCase)
    runner = CalibrationEvalRunner(scorer=_UniformScorer())
    # `run_eval` only supports `metric >= threshold` gates; the §6.5
    # ECE gate is inverted (`<= 0.05`), so we evaluate it after the
    # harness aggregates and combine both into the reported `passed`.
    report = run_eval(
        set_name=SET_NAME,
        cases=cases,
        runner=runner,
        thresholds={"label_accuracy": CALIBRATION_ACCURACY_THRESHOLD},
    )
    aggregate_ece = report.aggregate.get("expected_calibration_error", float("inf"))
    ece_gate_passed = aggregate_ece <= CALIBRATION_ECE_THRESHOLD
    passed = report.passed and ece_gate_passed
    summary = {
        "set_name": SET_NAME,
        "cases": len(cases),
        "passed": passed,
        "thresholds": {
            "label_accuracy": CALIBRATION_ACCURACY_THRESHOLD,
            "expected_calibration_error": CALIBRATION_ECE_THRESHOLD,
        },
        "aggregate": report.aggregate,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
