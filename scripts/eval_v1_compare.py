"""Baseline measurement for the compare eval substrate.

Drives a degenerate `CompareVerifier` (always answers `Gap` for every
cluster) through `CompareEvalRunner` against `tests/eval/compare_eval.jsonl`
and prints a one-line JSON summary on stdout. The baseline is a wiring
check — its job is to exercise the substrate end-to-end, not to clear
the §6.6 release gate. A baseline that suddenly clears the gate means
the harness has been silently mocked.

SPEC-REF: §6.6 (compare = per-concept-cluster strengths/weaknesses), §14
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ctrldoc.eval.compare import (
    COMPARE_VERDICT_THRESHOLD,
    CompareEvalCase,
    CompareEvalRunner,
    CompareVerdictLiteral,
    ConceptComparisonInput,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "tests" / "eval" / "compare_eval.jsonl"
SET_NAME = "compare"


class _AlwaysGapVerifier:
    """Degenerate baseline — answers Gap for every cluster."""

    def verdicts(
        self,
        *,
        clusters: list[ConceptComparisonInput],
    ) -> list[CompareVerdictLiteral]:
        return ["Gap"] * len(clusters)


def main(argv: list[str] | None = None) -> int:
    del argv
    cases = load_jsonl_cases(EVAL_PATH, case_model=CompareEvalCase)
    runner = CompareEvalRunner(verifier=_AlwaysGapVerifier())
    report = run_eval(
        set_name=SET_NAME,
        cases=cases,
        runner=runner,
        thresholds={"accuracy": COMPARE_VERDICT_THRESHOLD},
    )
    summary = {
        "set_name": SET_NAME,
        "cases": len(cases),
        "passed": report.passed,
        "thresholds": {"accuracy": COMPARE_VERDICT_THRESHOLD},
        "aggregate": report.aggregate,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
