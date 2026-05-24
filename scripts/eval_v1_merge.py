"""Baseline measurement for the merge eval substrate.

Drives a degenerate `Merger` (puts every input claim into its own
singleton cluster, picking the input's own claim tuple as the
representative) through `MergeEvalRunner` against
`tests/eval/merge_eval.jsonl` and prints a one-line JSON summary on
stdout. The all-singletons baseline trivially satisfies the §6.6
loss invariant — every input claim id appears in exactly one output
cluster — so the substrate's hard gate is exercised, while the soft
pairwise-accuracy gate stays below 0.85 (most fixtures have
non-trivial merging gold). The baseline is a wiring check, not a
contender for the release gate.

SPEC-REF: §6.6 (merge = partition + Galois join with loss invariant), §14
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.eval.merge import (
    MERGE_PARTITION_THRESHOLD,
    InputClaim,
    MergedCluster,
    MergeEvalCase,
    MergeEvalRunner,
    MergeOutput,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "tests" / "eval" / "merge_eval.jsonl"
SET_NAME = "merge"


class _AllSingletonsMerger:
    """Degenerate baseline — every input claim → its own singleton cluster.

    Satisfies the §6.6 loss invariant by construction (each input id
    appears in exactly one cluster). The cluster representative is
    the input's own claim, so the representative-match metric is
    informative but uninteresting.
    """

    def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
        clusters = [
            MergedCluster(
                id=f"cluster-{ic.id}",
                member_claim_ids=[ic.id],
                strongest_claim=ic.claim,
            )
            for ic in input_claims
        ]
        return MergeOutput(clusters=clusters)


def main(argv: list[str] | None = None) -> int:
    del argv
    cases = load_jsonl_cases(EVAL_PATH, case_model=MergeEvalCase)
    runner = MergeEvalRunner(merger=_AllSingletonsMerger())
    report = run_eval(
        set_name=SET_NAME,
        cases=cases,
        runner=runner,
        thresholds={"pairwise_accuracy": MERGE_PARTITION_THRESHOLD},
    )
    summary = {
        "set_name": SET_NAME,
        "cases": len(cases),
        "passed": report.passed,
        "thresholds": {"pairwise_accuracy": MERGE_PARTITION_THRESHOLD},
        "aggregate": report.aggregate,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
