"""Contract tests for the escalating LLM-judge composer.

`EscalatingLLMJudge` calls a tier-1 judge first. If its confidence
clears `escalation_threshold` the tier-1 result is returned as-is;
otherwise the tier-2 (more expensive) judge is invoked and its
result is returned. The composer itself satisfies the `LLMJudge`
protocol so callers can compose escalation chains.

SPEC-REF: §4.4 (tier-2 escalation)
"""

from __future__ import annotations

import pytest

from ctrldoc.verify.judge import JudgeResult, LLMJudge
from ctrldoc.verify.judge_escalating import EscalatingLLMJudge


class _StubJudge:
    """LLMJudge stub that records calls and returns a fixed result."""

    def __init__(self, result: JudgeResult) -> None:
        self._result = result
        self.call_count = 0
        self.last_args: tuple[str, str] | None = None

    def judge(self, claim: str, evidence: str) -> JudgeResult:
        self.call_count += 1
        self.last_args = (claim, evidence)
        return self._result


def test_satisfies_protocol() -> None:
    composer = EscalatingLLMJudge(
        tier1=_StubJudge(JudgeResult(passed=True, confidence=0.9, reasoning="t1")),
        tier2=_StubJudge(JudgeResult(passed=True, confidence=0.95, reasoning="t2")),
    )
    assert isinstance(composer, LLMJudge)


def test_returns_tier1_when_confidence_above_threshold() -> None:
    tier1 = _StubJudge(JudgeResult(passed=True, confidence=0.9, reasoning="t1"))
    tier2 = _StubJudge(JudgeResult(passed=True, confidence=0.95, reasoning="t2"))
    composer = EscalatingLLMJudge(tier1=tier1, tier2=tier2, escalation_threshold=0.7)
    out = composer.judge("c", "e")
    assert out.reasoning == "t1"
    assert tier1.call_count == 1
    assert tier2.call_count == 0


def test_escalates_to_tier2_when_confidence_below_threshold() -> None:
    tier1 = _StubJudge(JudgeResult(passed=True, confidence=0.4, reasoning="t1"))
    tier2 = _StubJudge(JudgeResult(passed=False, confidence=0.85, reasoning="t2"))
    composer = EscalatingLLMJudge(tier1=tier1, tier2=tier2, escalation_threshold=0.7)
    out = composer.judge("c", "e")
    assert out.reasoning == "t2"
    assert out.passed is False
    assert tier1.call_count == 1
    assert tier2.call_count == 1


def test_escalation_at_threshold_keeps_tier1() -> None:
    """`confidence == threshold` should NOT escalate (≥ is the gate)."""
    tier1 = _StubJudge(JudgeResult(passed=True, confidence=0.7, reasoning="t1"))
    tier2 = _StubJudge(JudgeResult(passed=True, confidence=0.95, reasoning="t2"))
    composer = EscalatingLLMJudge(tier1=tier1, tier2=tier2, escalation_threshold=0.7)
    out = composer.judge("c", "e")
    assert out.reasoning == "t1"
    assert tier2.call_count == 0


def test_explicit_should_escalate_predicate_overrides_threshold() -> None:
    """Caller can supply a predicate (e.g. NLI/judge-disagreement) to force escalation."""
    tier1 = _StubJudge(JudgeResult(passed=True, confidence=0.99, reasoning="t1"))
    tier2 = _StubJudge(JudgeResult(passed=False, confidence=0.95, reasoning="t2"))
    composer = EscalatingLLMJudge(
        tier1=tier1,
        tier2=tier2,
        should_escalate=lambda result: not result.passed or result.passed,  # always escalate
    )
    out = composer.judge("c", "e")
    assert out.reasoning == "t2"
    assert tier2.call_count == 1


def test_predicate_returning_false_keeps_tier1() -> None:
    tier1 = _StubJudge(JudgeResult(passed=True, confidence=0.1, reasoning="t1"))
    tier2 = _StubJudge(JudgeResult(passed=True, confidence=0.95, reasoning="t2"))
    composer = EscalatingLLMJudge(
        tier1=tier1,
        tier2=tier2,
        should_escalate=lambda _result: False,
    )
    out = composer.judge("c", "e")
    assert out.reasoning == "t1"
    assert tier2.call_count == 0


def test_tier1_receives_full_inputs() -> None:
    tier1 = _StubJudge(JudgeResult(passed=True, confidence=0.9, reasoning="t1"))
    tier2 = _StubJudge(JudgeResult(passed=True, confidence=0.95, reasoning="t2"))
    composer = EscalatingLLMJudge(tier1=tier1, tier2=tier2)
    composer.judge("the claim", "the evidence")
    assert tier1.last_args == ("the claim", "the evidence")


def test_tier2_receives_same_inputs_when_escalated() -> None:
    tier1 = _StubJudge(JudgeResult(passed=False, confidence=0.2, reasoning="t1"))
    tier2 = _StubJudge(JudgeResult(passed=False, confidence=0.95, reasoning="t2"))
    composer = EscalatingLLMJudge(tier1=tier1, tier2=tier2, escalation_threshold=0.7)
    composer.judge("the claim", "the evidence")
    assert tier2.last_args == ("the claim", "the evidence")


def test_invalid_threshold_rejected() -> None:
    judge_a = _StubJudge(JudgeResult(passed=True, confidence=0.5, reasoning=""))
    judge_b = _StubJudge(JudgeResult(passed=True, confidence=0.5, reasoning=""))
    with pytest.raises(ValueError):
        EscalatingLLMJudge(tier1=judge_a, tier2=judge_b, escalation_threshold=-0.01)
    with pytest.raises(ValueError):
        EscalatingLLMJudge(tier1=judge_a, tier2=judge_b, escalation_threshold=1.01)
