"""Contract tests for the claim verifier with repair pass.

`ClaimVerifier.verify(text)` retrieves evidence, runs NLI + judge,
and on a first-pass failure broadens retrieval for one repair pass.
Both checks must pass at the same depth for the claim to be marked
verified; otherwise the returned `Claim` is refused (`verified=False`)
and citations are still populated from whatever the retriever found.

SPEC-REF: §4.4 (verifier — refusal + one repair pass)
"""

from __future__ import annotations

import pytest

from ctrldoc.models import Span
from ctrldoc.verify.claim_verifier import (
    ClaimVerifier,
    RetrievalDepth,
    RetrievedEvidence,
    Retriever,
)
from ctrldoc.verify.judge import JudgeResult
from ctrldoc.verify.nli import NLIResult


class _StubRetriever:
    """Records every retrieve call and returns the configured evidence."""

    def __init__(self, *, normal: RetrievedEvidence, broad: RetrievedEvidence) -> None:
        self._by_depth = {"normal": normal, "broad": broad}
        self.calls: list[tuple[str, RetrievalDepth]] = []

    def retrieve(self, claim_text: str, *, depth: RetrievalDepth) -> RetrievedEvidence:
        self.calls.append((claim_text, depth))
        return self._by_depth[depth]


class _StubNLI:
    def __init__(self, results: list[NLIResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        self.calls.append((premise, hypothesis))
        return self._results.pop(0) if self._results else NLIResult(label="neutral", score=0.0)


class _StubJudge:
    def __init__(self, results: list[JudgeResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []

    def judge(self, claim: str, evidence: str) -> JudgeResult:
        self.calls.append((claim, evidence))
        return (
            self._results.pop(0)
            if self._results
            else JudgeResult(passed=False, confidence=0.0, reasoning="exhausted stub")
        )


def _evidence(text: str, *, chunk_id: str = "c1") -> RetrievedEvidence:
    return RetrievedEvidence(
        text=text,
        citations=[Span(chunk_id=chunk_id, char_start=0, char_end=len(text), text=text)],
    )


def _verifier(
    *,
    retriever: Retriever,
    nli_results: list[NLIResult],
    judge_results: list[JudgeResult],
) -> tuple[ClaimVerifier, _StubNLI, _StubJudge]:
    nli = _StubNLI(nli_results)
    judge = _StubJudge(judge_results)
    return ClaimVerifier(nli=nli, judge=judge, retriever=retriever), nli, judge


# --- first-pass success ---


def test_first_pass_success_marks_verified() -> None:
    retriever = _StubRetriever(
        normal=_evidence("Aurora uses consistent hashing across nodes."),
        broad=_evidence("never reached"),
    )
    verifier, nli, judge = _verifier(
        retriever=retriever,
        nli_results=[NLIResult(label="entailment", score=0.95)],
        judge_results=[JudgeResult(passed=True, confidence=0.9, reasoning="ok")],
    )
    claim = verifier.verify("Aurora uses consistent hashing.")
    assert claim.verified is True
    assert claim.confidence == pytest.approx(0.9)  # min(nli, judge)
    assert claim.nli_score == pytest.approx(0.95)
    assert claim.judge_score == pytest.approx(0.9)
    assert claim.text == "Aurora uses consistent hashing."
    assert len(claim.citations) == 1
    # Repair pass was never invoked.
    assert [c[1] for c in retriever.calls] == ["normal"]
    assert len(nli.calls) == 1
    assert len(judge.calls) == 1


# --- repair pass recovery ---


def test_repair_pass_can_rescue_failed_claim() -> None:
    retriever = _StubRetriever(
        normal=_evidence("too-narrow context."),
        broad=_evidence("Aurora uses consistent hashing across nodes. Detailed evidence."),
    )
    verifier, nli, judge = _verifier(
        retriever=retriever,
        nli_results=[
            NLIResult(label="neutral", score=0.4),
            NLIResult(label="entailment", score=0.9),
        ],
        judge_results=[
            JudgeResult(passed=False, confidence=0.4, reasoning="weak"),
            JudgeResult(passed=True, confidence=0.85, reasoning="solid"),
        ],
    )
    claim = verifier.verify("Aurora uses consistent hashing.")
    assert claim.verified is True
    assert claim.confidence == pytest.approx(0.85)
    # Retriever called twice: normal first, then broad.
    assert [c[1] for c in retriever.calls] == ["normal", "broad"]
    assert len(nli.calls) == 2
    assert len(judge.calls) == 2
    # Citations come from the broad retrieve (the one that succeeded).
    assert claim.citations[0].text.startswith("Aurora uses consistent hashing")


# --- double failure ---


def test_both_passes_failing_marks_refused() -> None:
    retriever = _StubRetriever(
        normal=_evidence("unrelated normal."),
        broad=_evidence("unrelated broad."),
    )
    verifier, _, _ = _verifier(
        retriever=retriever,
        nli_results=[
            NLIResult(label="neutral", score=0.1),
            NLIResult(label="neutral", score=0.2),
        ],
        judge_results=[
            JudgeResult(passed=False, confidence=0.1, reasoning="no support"),
            JudgeResult(passed=False, confidence=0.2, reasoning="still no support"),
        ],
    )
    claim = verifier.verify("Aurora supports transactions.")
    assert claim.verified is False
    assert claim.confidence == pytest.approx(0.0)
    assert claim.nli_score == pytest.approx(0.2)
    assert claim.judge_score == pytest.approx(0.2)
    # Citations still come from the second (broader) retrieve so the
    # caller can see what the verifier was looking at.
    assert claim.citations[0].text == "unrelated broad."


# --- mixed failure modes ---


def test_nli_pass_judge_fail_triggers_repair() -> None:
    retriever = _StubRetriever(
        normal=_evidence("partial."),
        broad=_evidence("complete."),
    )
    verifier, _, _ = _verifier(
        retriever=retriever,
        nli_results=[
            NLIResult(label="entailment", score=0.99),
            NLIResult(label="entailment", score=0.99),
        ],
        judge_results=[
            JudgeResult(passed=False, confidence=0.3, reasoning="judge says no"),
            JudgeResult(passed=True, confidence=0.9, reasoning="judge says yes"),
        ],
    )
    claim = verifier.verify("claim")
    assert claim.verified is True
    assert [c[1] for c in retriever.calls] == ["normal", "broad"]


def test_judge_pass_nli_fail_triggers_repair() -> None:
    retriever = _StubRetriever(
        normal=_evidence("partial."),
        broad=_evidence("complete."),
    )
    verifier, _, _ = _verifier(
        retriever=retriever,
        nli_results=[
            NLIResult(label="neutral", score=0.4),
            NLIResult(label="entailment", score=0.95),
        ],
        judge_results=[
            JudgeResult(passed=True, confidence=0.9, reasoning="ok"),
            JudgeResult(passed=True, confidence=0.9, reasoning="ok"),
        ],
    )
    claim = verifier.verify("claim")
    assert claim.verified is True


# --- empty input ---


def test_empty_claim_short_circuits_refusal() -> None:
    retriever = _StubRetriever(normal=_evidence("x"), broad=_evidence("y"))
    verifier, nli, judge = _verifier(
        retriever=retriever,
        nli_results=[],
        judge_results=[],
    )
    claim = verifier.verify("")
    assert claim.verified is False
    assert claim.confidence == pytest.approx(0.0)
    assert claim.citations == []
    # No downstream calls were made.
    assert retriever.calls == []
    assert nli.calls == []
    assert judge.calls == []
