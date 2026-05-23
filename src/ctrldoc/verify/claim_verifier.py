"""Claim verifier — refusal + one repair pass.

`ClaimVerifier.verify(text)` retrieves evidence (normal depth first,
broad depth on a failed first pass) and runs NLI + judge against it.
Both must pass for `Claim.verified=True`; otherwise the claim comes
back with `verified=False` but still carries the (broader) citations
so the caller can see what the verifier looked at.

SPEC-REF: §4.4 (verifier — refusal + one repair pass)
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.models import Claim, Span
from ctrldoc.verify.judge import JudgeResult, LLMJudge
from ctrldoc.verify.nli import NLIChecker, NLIResult

RetrievalDepth = Literal["normal", "broad"]


class RetrievedEvidence(BaseModel):
    """Evidence text plus the citation spans it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    citations: list[Span]


@runtime_checkable
class Retriever(Protocol):
    """Claim text → bundled evidence at the requested depth."""

    def retrieve(self, claim_text: str, *, depth: RetrievalDepth) -> RetrievedEvidence: ...


class ClaimVerifier:
    """NLI + LLM-judge pipeline with one repair pass on failure."""

    def __init__(
        self,
        *,
        nli: NLIChecker,
        judge: LLMJudge,
        retriever: Retriever,
    ) -> None:
        self._nli = nli
        self._judge = judge
        self._retriever = retriever

    def verify(self, claim_text: str) -> Claim:
        body = claim_text.strip()
        if not body:
            return Claim(
                text=claim_text,
                citations=[],
                verified=False,
                confidence=0.0,
                nli_score=0.0,
                judge_score=0.0,
            )

        evidence, nli_result, judge_result = self._attempt(body, depth="normal")
        verified = _both_pass(nli_result, judge_result)

        if not verified:
            evidence, nli_result, judge_result = self._attempt(body, depth="broad")
            verified = _both_pass(nli_result, judge_result)

        confidence = min(nli_result.score, judge_result.confidence) if verified else 0.0
        return Claim(
            text=body,
            citations=list(evidence.citations),
            verified=verified,
            confidence=confidence,
            nli_score=nli_result.score,
            judge_score=judge_result.confidence,
        )

    def _attempt(
        self,
        body: str,
        *,
        depth: RetrievalDepth,
    ) -> tuple[RetrievedEvidence, NLIResult, JudgeResult]:
        evidence = self._retriever.retrieve(body, depth=depth)
        nli_result = self._nli.check(evidence.text, body)
        judge_result = self._judge.judge(body, evidence.text)
        return evidence, nli_result, judge_result


def _both_pass(nli_result: NLIResult, judge_result: JudgeResult) -> bool:
    return nli_result.label == "entailment" and judge_result.passed


__all__ = ["ClaimVerifier", "RetrievalDepth", "RetrievedEvidence", "Retriever"]
