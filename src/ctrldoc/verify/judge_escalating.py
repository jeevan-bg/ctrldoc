"""Tier-2 escalation composer for the LLM-judge.

`EscalatingLLMJudge(tier1, tier2)` calls tier-1 first; if its
confidence falls below the escalation threshold (or a caller-supplied
predicate fires), it forwards the call to the more expensive tier-2
judge and returns that result. The composer itself satisfies the
`LLMJudge` protocol so escalation chains can nest.

SPEC-REF: §4.4 (verifier — escalate to Opus if disagree)
"""

from __future__ import annotations

from collections.abc import Callable

from ctrldoc.verify.judge import JudgeResult, LLMJudge


class EscalatingLLMJudge:
    """Two-tier judge with confidence- or predicate-driven escalation."""

    def __init__(
        self,
        *,
        tier1: LLMJudge,
        tier2: LLMJudge,
        escalation_threshold: float = 0.7,
        should_escalate: Callable[[JudgeResult], bool] | None = None,
    ) -> None:
        if not 0.0 <= escalation_threshold <= 1.0:
            raise ValueError("escalation_threshold must be in [0, 1]")
        self._tier1 = tier1
        self._tier2 = tier2
        self._threshold = escalation_threshold
        self._predicate = should_escalate

    def judge(self, claim: str, evidence: str) -> JudgeResult:
        tier1 = self._tier1.judge(claim, evidence)
        if self._predicate is not None:
            escalate = self._predicate(tier1)
        else:
            escalate = tier1.confidence < self._threshold
        if escalate:
            return self._tier2.judge(claim, evidence)
        return tier1


__all__ = ["EscalatingLLMJudge"]
