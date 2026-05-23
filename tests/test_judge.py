"""Contract tests for the verifier's LLM-judge.

`LLMJudge.judge(claim, evidence)` returns a binary `pass` decision
plus a confidence score and a short reasoning string. The MVP ships
a deterministic heuristic that reuses NLI-style token overlap; the
production Qwen2.5-7B backend is queued as S-052b (needs Ollama).

SPEC-REF: §4.4 (verifier step 3 — LLM-as-judge)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.verify.judge import (
    HeuristicLLMJudge,
    JudgeResult,
    LLMJudge,
)


def test_heuristic_satisfies_protocol() -> None:
    assert isinstance(HeuristicLLMJudge(), LLMJudge)


# --- JudgeResult ---


def test_result_is_frozen() -> None:
    r = JudgeResult(passed=True, confidence=0.8, reasoning="ok")
    with pytest.raises(ValidationError):
        r.passed = False  # type: ignore[misc]


def test_result_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        JudgeResult(passed=True, confidence=1.5, reasoning="bad")
    with pytest.raises(ValidationError):
        JudgeResult(passed=True, confidence=-0.1, reasoning="bad")


def test_result_round_trip() -> None:
    r = JudgeResult(passed=False, confidence=0.3, reasoning="insufficient overlap")
    assert JudgeResult.model_validate(r.model_dump()) == r


# --- HeuristicLLMJudge ---


def test_full_overlap_passes() -> None:
    result = HeuristicLLMJudge().judge(
        claim="Aurora uses consistent hashing.",
        evidence="Aurora uses consistent hashing for routing across nodes.",
    )
    assert result.passed is True
    assert result.confidence == pytest.approx(1.0)


def test_zero_overlap_fails() -> None:
    result = HeuristicLLMJudge().judge(
        claim="Aurora supports transactions.",
        evidence="Completely unrelated text about caching.",
    )
    assert result.passed is False
    assert result.confidence == pytest.approx(0.0)


def test_partial_overlap_passes_or_fails_based_on_threshold() -> None:
    judge = HeuristicLLMJudge(pass_threshold=0.5)
    result = judge.judge(
        claim="alpha beta gamma delta",
        evidence="alpha beta gamma",
    )
    # 3/4 hypothesis tokens overlap → above 0.5 threshold → passed.
    assert result.passed is True
    assert result.confidence == pytest.approx(3 / 4)


def test_empty_claim_fails_with_zero_confidence() -> None:
    result = HeuristicLLMJudge().judge(claim="", evidence="evidence text")
    assert result.passed is False
    assert result.confidence == pytest.approx(0.0)


def test_empty_evidence_fails_with_zero_confidence() -> None:
    result = HeuristicLLMJudge().judge(claim="some claim", evidence="")
    assert result.passed is False
    assert result.confidence == pytest.approx(0.0)


def test_reasoning_non_empty_on_pass() -> None:
    result = HeuristicLLMJudge().judge(
        claim="Aurora uses consistent hashing.",
        evidence="Aurora uses consistent hashing.",
    )
    assert result.reasoning.strip() != ""


def test_reasoning_non_empty_on_fail() -> None:
    result = HeuristicLLMJudge().judge(claim="x", evidence="y")
    assert result.reasoning.strip() != ""


def test_case_insensitive() -> None:
    result = HeuristicLLMJudge().judge(
        claim="Alpha Beta Gamma",
        evidence="alpha beta gamma",
    )
    assert result.passed is True


def test_determinism() -> None:
    judge = HeuristicLLMJudge()
    args = ("alpha beta gamma", "alpha beta gamma")
    assert judge.judge(*args) == judge.judge(*args)


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        HeuristicLLMJudge(pass_threshold=-0.01)
    with pytest.raises(ValueError):
        HeuristicLLMJudge(pass_threshold=1.01)
