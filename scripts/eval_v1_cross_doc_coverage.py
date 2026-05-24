"""Baseline measurement for the cross_doc_coverage eval substrate.

Drives a degenerate `CrossDocCoverageVerifier` (always answers
`Covered` for every target claim) through `CrossDocCoverageEvalRunner`
against `tests/eval/cross_doc_coverage_eval.jsonl` and prints a
one-line JSON summary on stdout. The baseline is a wiring check —
its job is to exercise the substrate end-to-end, not to clear the
§6.6 release gate. A baseline that suddenly clears the gate means
the harness has been silently mocked.

SPEC-REF: §6.6 (optimal-transport coverage), §14
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.cross_doc_coverage import (
    CROSS_DOC_COVERAGE_THRESHOLD,
    CoverageVerdictLiteral,
    CrossDocCoverageEvalCase,
    CrossDocCoverageEvalRunner,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "tests" / "eval" / "cross_doc_coverage_eval.jsonl"
SET_NAME = "cross_doc_coverage"


class _AlwaysCoveredVerifier:
    """Degenerate baseline — answers Covered for every target claim."""

    def verdicts(
        self,
        *,
        source: list[ClaimTuple],
        target: list[ClaimTuple],
    ) -> list[CoverageVerdictLiteral]:
        del source
        return ["Covered"] * len(target)


def main(argv: list[str] | None = None) -> int:
    del argv
    cases = load_jsonl_cases(EVAL_PATH, case_model=CrossDocCoverageEvalCase)
    runner = CrossDocCoverageEvalRunner(verifier=_AlwaysCoveredVerifier())
    report = run_eval(
        set_name=SET_NAME,
        cases=cases,
        runner=runner,
        thresholds={"accuracy": CROSS_DOC_COVERAGE_THRESHOLD},
    )
    summary = {
        "set_name": SET_NAME,
        "cases": len(cases),
        "passed": report.passed,
        "thresholds": {"accuracy": CROSS_DOC_COVERAGE_THRESHOLD},
        "aggregate": report.aggregate,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
