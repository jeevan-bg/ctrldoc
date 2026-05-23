"""LLM-judge — protocol and heuristic reference.

`LLMJudge.judge(claim, evidence)` decides whether the evidence
supports the claim and returns a `JudgeResult` (boolean pass +
confidence + short reasoning). The MVP ships a deterministic
token-overlap heuristic so the verifier and downstream tests can
exercise the contract without an LLM call.

SPEC-REF: §4.4 (verifier step 3 — LLM-as-judge)
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.models import UnitInterval


class JudgeResult(BaseModel):
    """One judgement: `passed` ∈ {True, False}, confidence ∈ [0, 1]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    confidence: UnitInterval
    reasoning: str


@runtime_checkable
class LLMJudge(Protocol):
    """Claim + evidence → pass/fail + confidence + reasoning."""

    def judge(self, claim: str, evidence: str) -> JudgeResult: ...


class HeuristicLLMJudge:
    """Token-overlap baseline. Deterministic, dependency-free.

    `confidence` is the fraction of (lower-cased) claim tokens that
    appear in the evidence. `passed` is `True` when `confidence`
    clears `pass_threshold`. `reasoning` summarises the score.
    """

    def __init__(self, *, pass_threshold: float = 0.5) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0, 1]")
        self._threshold = pass_threshold

    def judge(self, claim: str, evidence: str) -> JudgeResult:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            return JudgeResult(
                passed=False,
                confidence=0.0,
                reasoning="empty claim — nothing to judge",
            )
        evidence_tokens = _tokenize(evidence)
        if not evidence_tokens:
            return JudgeResult(
                passed=False,
                confidence=0.0,
                reasoning="empty evidence — cannot support the claim",
            )
        overlap = sum(1 for token in claim_tokens if token in evidence_tokens)
        confidence = overlap / len(claim_tokens)
        passed = confidence >= self._threshold
        reasoning = (
            f"{overlap}/{len(claim_tokens)} claim tokens overlap with evidence "
            f"(threshold={self._threshold:.2f})"
        )
        return JudgeResult(passed=passed, confidence=confidence, reasoning=reasoning)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


__all__ = ["HeuristicLLMJudge", "JudgeResult", "LLMJudge"]
