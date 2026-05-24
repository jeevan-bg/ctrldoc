"""Baseline measurement for the claim_extraction eval substrate.

Drives a degenerate `ClaimExtractor` (always returns `[]`) through
`ClaimExtractionEvalRunner` against `tests/eval/claim_extraction_eval.jsonl`
and prints a one-line JSON summary on stdout. The baseline is a
wiring check — its job is to exercise the substrate end-to-end, not
to clear the §14 release gate. A baseline that suddenly clears the
gate means the harness has been silently mocked.

Exit code is 0 on a clean run regardless of threshold passage; the
aggregator (`scripts/run_v1_smoke.sh`) keys off the JSON line, not
the exit code, for substrate-level pass/fail signal.

SPEC-REF: §6.2 (universal claim tuple), §14 (claim_F1 gate)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ctrldoc.eval.claim_extraction import (
    CLAIM_F1_THRESHOLD,
    ClaimExtractionEvalCase,
    ClaimExtractionEvalRunner,
    ClaimTuple,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "tests" / "eval" / "claim_extraction_eval.jsonl"
SET_NAME = "claim_extraction"


class _EmptyExtractor:
    """Degenerate baseline — returns no tuples for any sentence."""

    def extract(self, sentence: str) -> list[ClaimTuple]:
        return []


def main(argv: list[str] | None = None) -> int:
    del argv  # baselines take no flags today
    cases = load_jsonl_cases(EVAL_PATH, case_model=ClaimExtractionEvalCase)
    runner = ClaimExtractionEvalRunner(extractor=_EmptyExtractor())
    report = run_eval(
        set_name=SET_NAME,
        cases=cases,
        runner=runner,
        thresholds={"f1": CLAIM_F1_THRESHOLD},
    )
    summary = {
        "set_name": SET_NAME,
        "cases": len(cases),
        "passed": report.passed,
        "thresholds": {"f1": CLAIM_F1_THRESHOLD},
        "aggregate": report.aggregate,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
