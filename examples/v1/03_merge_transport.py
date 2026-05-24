"""v1 merge via the §6.6 optimal-transport reduction.

`merge(input_claims, scorer)` collapses N input claims (typically
across multiple docs) into K equivalence-class clusters, respecting
the §13 loss invariant: every input claim ID maps to exactly one
output cluster, no input is lost, no input duplicates. Each cluster's
strongest representative is selected via the Galois GLB (`claim_meet`)
across its members.

This walkthrough mixes obvious paraphrases (Galois-resolved at zero
NLI cost) with one Galois-incomparable pair (escalated to bidirectional
NLI). A synthetic scorer keeps the example hermetic — no LLM, no
network. Swap it for any `NLIScorer` Protocol implementation to drive
the operation for real.

Run:

    python examples/v1/03_merge_transport.py

SPEC-REF: §6.6
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.merge import InputClaim
from ctrldoc.ops.merge import MergeConfig, merge


def _claim(subject: str, predicate: str, obj: str, qualifier: str = "") -> ClaimTuple:
    """Build a minimal asserted-affirmative claim tuple."""
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity="affirmative",
        modality="asserted",
        qualifier=qualifier,
    )


@dataclass
class _ParaphraseScorer:
    """Synthetic NLI that recognises one paraphrase pair.

    Real backends plug in via the same `NLIScorer` Protocol shape.
    """

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        a = premise.strip().lower()
        b = hypothesis.strip().lower()
        # Recognise "the api uses bearer tokens" ≡ "the api uses jwt".
        markers = {"bearer", "jwt"}
        if markers.issubset(a.split() + b.split()):
            return NLIScore(entailment=0.88, contradiction=0.04, neutral=0.08)
        if a == b:
            return NLIScore(entailment=0.97, contradiction=0.01, neutral=0.02)
        return NLIScore(entailment=0.10, contradiction=0.05, neutral=0.85)


def main() -> None:
    # Three docs' worth of claims mixed together.
    # - Doc A and Doc B agree the API uses bearer tokens (Galois-equivalent).
    # - Doc C says "JWT" — a paraphrase of bearer tokens (NLI fallback).
    # - Doc A and Doc B both say the API rate-limits requests, with
    #   different qualifiers; the empty qualifier subsumes the scoped
    #   one at the Galois floor.
    # - Doc C contributes one unique claim about logging.
    inputs = [
        InputClaim(
            id="a-1",
            doc_id="doc-a",
            doc_type="spec",
            claim=_claim("the api", "uses", "bearer tokens"),
        ),
        InputClaim(
            id="b-1",
            doc_id="doc-b",
            doc_type="spec",
            claim=_claim("the api", "uses", "bearer tokens"),
        ),
        InputClaim(
            id="c-1",
            doc_id="doc-c",
            doc_type="rfc",
            claim=_claim("the api", "uses", "jwt"),
        ),
        InputClaim(
            id="a-2",
            doc_id="doc-a",
            doc_type="spec",
            claim=_claim("the api", "rate-limits", "requests"),
        ),
        InputClaim(
            id="b-2",
            doc_id="doc-b",
            doc_type="spec",
            claim=_claim("the api", "rate-limits", "requests", qualifier="per ip address"),
        ),
        InputClaim(
            id="c-2",
            doc_id="doc-c",
            doc_type="rfc",
            claim=_claim("the api", "logs", "every failed request"),
        ),
    ]

    result = merge(
        input_claims=inputs,
        scorer=_ParaphraseScorer(),
        config=MergeConfig(equivalence_threshold=0.7),
    )

    # Loss invariant: every input id appears in exactly one cluster.
    flat = [mid for c in result.output.clusters for mid in c.member_claim_ids]
    assert sorted(flat) == sorted(
        ic.id for ic in inputs
    ), "loss invariant violated — merge should be lossless"

    print(
        json.dumps(
            {
                "n_inputs": len(inputs),
                "n_clusters": len(result.output.clusters),
                "scorer_calls": result.scorer_calls,
                "clusters": [
                    {
                        "id": c.id,
                        "members": list(c.member_claim_ids),
                        "strongest": (
                            f"{c.strongest_claim.subject} "
                            f"{c.strongest_claim.predicate} "
                            f"{c.strongest_claim.object}"
                            + (
                                f" [{c.strongest_claim.qualifier}]"
                                if c.strongest_claim.qualifier
                                else ""
                            )
                        ),
                    }
                    for c in result.output.clusters
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
