"""Family-9 invariants: verifier calibration.

A labelled synthetic dataset exercises `ClaimVerifier` end-to-end
with the heuristic NLI + judge references and a stub retriever. The
test asserts SPEC §8.6 family-9 calibration targets:

  - False-positive rate (claim marked verified when truth says
    refuse) ≤ 2%.
  - False-negative rate (claim refused when truth says verify) ≤ 5%.

The targets are tight on purpose: a regression in the verifier
contract (e.g. flipping the AND to an OR between NLI and judge, or
defaulting thresholds outside their spec range) will surface here
before the production cross-encoder/judge wirings land.

SPEC-REF: §4.4, §8.6 family 9
"""

from __future__ import annotations

import random

import pytest

from ctrldoc.models import Span
from ctrldoc.verify.claim_verifier import (
    ClaimVerifier,
    RetrievalDepth,
    RetrievedEvidence,
)
from ctrldoc.verify.judge import HeuristicLLMJudge
from ctrldoc.verify.nli import HeuristicNLIChecker

_DOMAIN_WORDS = [
    "aurora",
    "replication",
    "hashing",
    "consistent",
    "shard",
    "ring",
    "gossip",
    "cluster",
    "membership",
    "virtual",
    "node",
    "cache",
    "store",
    "eviction",
    "quorum",
    "lease",
    "replica",
    "primary",
    "follower",
    "partition",
    "heartbeat",
    "failure",
    "detector",
]


def _make_positive(rng: random.Random, idx: int) -> tuple[str, str]:
    """Claim is a strict subset of evidence — should verify."""
    base = rng.sample(_DOMAIN_WORDS, k=4)
    claim = f"Aurora supports {' '.join(base)}."
    extra = rng.sample(_DOMAIN_WORDS, k=6)
    evidence = (
        f"In section {idx} the system documents that aurora supports "
        f"{' '.join(base)}. It also covers {' '.join(extra)}."
    )
    return claim, evidence


def _make_negative(rng: random.Random, idx: int) -> tuple[str, str]:
    """Claim and evidence share at most ~25% of their tokens — should refuse."""
    claim_words = rng.sample(_DOMAIN_WORDS, k=4)
    # Sample evidence words from the remainder so token overlap is bounded.
    remaining = [w for w in _DOMAIN_WORDS if w not in claim_words]
    evidence_words = rng.sample(remaining, k=8)
    claim = f"The {' '.join(claim_words)} subsystem ships next quarter."
    evidence = (
        f"Section {idx} describes operational practices around "
        f"{' '.join(evidence_words)}. No release dates are mentioned."
    )
    return claim, evidence


class _DatasetRetriever:
    """Stub retriever that maps each claim to its labelled evidence."""

    def __init__(self, by_claim: dict[str, str]) -> None:
        self._by_claim = by_claim

    def retrieve(self, claim_text: str, *, depth: RetrievalDepth) -> RetrievedEvidence:
        # Same evidence at both depths for the calibration test — we are
        # measuring the verifier gate, not the retriever's broadening
        # behaviour (which has its own tests in test_claim_verifier).
        text = self._by_claim.get(claim_text, "")
        return RetrievedEvidence(
            text=text,
            citations=[Span(chunk_id="c-fixed", char_start=0, char_end=len(text), text=text)],
        )


@pytest.fixture
def calibration_set() -> tuple[list[tuple[str, str, bool]], _DatasetRetriever]:
    rng = random.Random(0xCA1)  # deterministic
    pairs: list[tuple[str, str, bool]] = []
    by_claim: dict[str, str] = {}
    for i in range(50):
        claim, evidence = _make_positive(rng, i)
        pairs.append((claim, evidence, True))
        by_claim[claim] = evidence
    for i in range(50):
        claim, evidence = _make_negative(rng, i)
        pairs.append((claim, evidence, False))
        by_claim[claim] = evidence
    return pairs, _DatasetRetriever(by_claim)


@pytest.mark.family_verifier_calibration
def test_calibration_meets_spec_targets(
    calibration_set: tuple[list[tuple[str, str, bool]], _DatasetRetriever],
) -> None:
    pairs, retriever = calibration_set
    verifier = ClaimVerifier(
        nli=HeuristicNLIChecker(entailment_threshold=0.999),
        judge=HeuristicLLMJudge(pass_threshold=0.5),
        retriever=retriever,
    )

    positives = [p for p in pairs if p[2]]
    negatives = [p for p in pairs if not p[2]]
    assert positives and negatives, "calibration set must have both classes"

    false_negatives = 0
    for claim, _evidence, _label in positives:
        result = verifier.verify(claim)
        if not result.verified:
            false_negatives += 1

    false_positives = 0
    for claim, _evidence, _label in negatives:
        result = verifier.verify(claim)
        if result.verified:
            false_positives += 1

    fp_rate = false_positives / len(negatives)
    fn_rate = false_negatives / len(positives)
    assert fp_rate <= 0.02, (
        f"FP rate {fp_rate:.3f} exceeds §8.6 family-9 cap of 0.02 "
        f"({false_positives}/{len(negatives)})"
    )
    assert fn_rate <= 0.05, (
        f"FN rate {fn_rate:.3f} exceeds §8.6 family-9 cap of 0.05 "
        f"({false_negatives}/{len(positives)})"
    )


@pytest.mark.family_verifier_calibration
def test_confidence_bins_match_accuracy(
    calibration_set: tuple[list[tuple[str, str, bool]], _DatasetRetriever],
) -> None:
    """High-confidence verified claims should never be on a wrong label.

    Specifically: for any verified claim with confidence ≥ 0.9 the
    ground-truth label must be `True`. This catches a regression that
    would let high-confidence false positives through.
    """
    pairs, retriever = calibration_set
    verifier = ClaimVerifier(
        nli=HeuristicNLIChecker(entailment_threshold=0.999),
        judge=HeuristicLLMJudge(pass_threshold=0.5),
        retriever=retriever,
    )
    for claim, _evidence, label in pairs:
        result = verifier.verify(claim)
        if result.verified and result.confidence >= 0.9:
            assert label is True, (
                f"high-confidence verification on a negative claim: "
                f"{claim!r} (confidence={result.confidence:.3f})"
            )
