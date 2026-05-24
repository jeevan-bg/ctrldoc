"""v1 coverage via the §6.6 optimal-transport reduction.

`coverage(source, target, scorer)` answers "does target B cover
source A?" — per-target-claim verdicts (`Covered` / `Missing`) plus
calibrated confidence. The reduction is identical to `list_check`
(which grades a list against a doc) and uses the same min-cost-flow
engine `compare` and `merge` ride on.

This walkthrough drives the operation with a deterministic synthetic
NLI scorer (exact-string-match entailment with a soft fallback) so it
runs hermetically with no model dependency. Swap the scorer for any
`NLIScorer` Protocol implementation — DeBERTa NLI, a local Qwen
judge, an Anthropic-mediated tool call — to drive it for real.

Run:

    python examples/v1/02_coverage_transport.py

SPEC-REF: §6.6
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.ops.coverage import CoverageConfig, coverage


def _claim(subject: str, predicate: str, obj: str) -> ClaimTuple:
    """Build a minimal asserted-affirmative claim tuple."""
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity="affirmative",
        modality="asserted",
    )


@dataclass
class _ExactMatchScorer:
    """Synthetic NLI: exact-string-match entailment, soft neutral elsewhere.

    Real backends (DeBERTa NLI, an LLM judge, the calibrated wrapper
    from `ctrldoc.extract.isotonic_calibration`) plug in via the same
    `NLIScorer` Protocol shape.
    """

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        if premise.strip().lower() == hypothesis.strip().lower():
            return NLIScore(entailment=0.97, contradiction=0.01, neutral=0.02)
        return NLIScore(entailment=0.10, contradiction=0.05, neutral=0.85)


def main() -> None:
    # The source doc says the system retries and logs failures.
    source = [
        _claim("the system", "retries", "transient failures"),
        _claim("the system", "logs", "every failure to disk"),
    ]
    # The target checklist asks whether the system retries, logs, and
    # encrypts. We expect Covered / Covered / Missing — the source
    # never mentions encryption.
    target = [
        _claim("the system", "retries", "transient failures"),
        _claim("the system", "logs", "every failure to disk"),
        _claim("the system", "encrypts", "data at rest"),
    ]

    result = coverage(
        source=source,
        target=target,
        scorer=_ExactMatchScorer(),
        config=CoverageConfig(entailment_threshold=0.5),
    )

    print(
        json.dumps(
            {
                "verdicts": list(result.verdicts),
                "scorer_calls": result.scorer_calls,
                "cost_contract": "exactly |sources| * |targets| = "
                f"{len(source)} * {len(target)} = {len(source) * len(target)}",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
