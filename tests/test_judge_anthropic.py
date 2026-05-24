"""Unit tests for the Anthropic-backed LLM-judge — no network calls.

Verifies request shape, JSON-only response parsing, and validation
of the typed JudgeResult. The real Anthropic API is never called.

SPEC-REF: §4.4 (tier-2 LLM-judge)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from ctrldoc.verify.judge import LLMJudge
from ctrldoc.verify.judge_anthropic import AnthropicLLMJudge


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Message:
    content: list[_TextBlock]


class _StubMessages:
    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text
        self.last_call: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _Message:
        self.last_call = kwargs
        return _Message(content=[_TextBlock(text=self._reply_text)])


class _StubClient:
    def __init__(self, reply_text: str) -> None:
        self.messages = _StubMessages(reply_text)


def _reply(passed: bool, confidence: float, reasoning: str) -> str:
    return json.dumps({"passed": passed, "confidence": confidence, "reasoning": reasoning})


def test_satisfies_protocol() -> None:
    client = _StubClient(_reply(True, 0.9, "ok"))
    assert isinstance(AnthropicLLMJudge(client=client), LLMJudge)  # type: ignore[arg-type]


def test_parses_judge_result() -> None:
    client = _StubClient(_reply(True, 0.85, "evidence supports the claim"))
    result = AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]
    assert result.passed is True
    assert result.confidence == pytest.approx(0.85)
    assert result.reasoning == "evidence supports the claim"


def test_request_carries_claim_and_evidence() -> None:
    client = _StubClient(_reply(True, 1.0, "ok"))
    AnthropicLLMJudge(client=client).judge("the claim", "the evidence")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    user_content = call["messages"][0]["content"]
    assert "the claim" in user_content
    assert "the evidence" in user_content


def test_system_prompt_constrains_json() -> None:
    client = _StubClient(_reply(False, 0.0, ""))
    AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    system = call["system"].lower()
    assert "json" in system
    assert "passed" in system
    assert "confidence" in system


def test_empty_claim_returns_fail_without_api_call() -> None:
    client = _StubClient("should-not-be-used")
    result = AnthropicLLMJudge(client=client).judge("", "evidence")  # type: ignore[arg-type]
    assert result.passed is False
    assert result.confidence == pytest.approx(0.0)
    assert client.messages.last_call is None


def test_empty_evidence_returns_fail_without_api_call() -> None:
    client = _StubClient("should-not-be-used")
    result = AnthropicLLMJudge(client=client).judge("claim", "")  # type: ignore[arg-type]
    assert result.passed is False
    assert client.messages.last_call is None


def test_non_json_response_raises() -> None:
    client = _StubClient("not json")
    with pytest.raises(ValueError):
        AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]


def test_invalid_schema_response_raises() -> None:
    client = _StubClient(json.dumps({"wrong": "shape"}))
    with pytest.raises(ValueError):
        AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]


def test_confidence_clamped_when_model_overshoots() -> None:
    """The model can return scores outside [0,1]; clamp rather than crash."""
    client = _StubClient(_reply(True, 1.5, "overshoot"))
    result = AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]
    assert result.confidence == pytest.approx(1.0)


def test_confidence_clamped_when_model_undershoots() -> None:
    client = _StubClient(_reply(False, -0.2, "undershoot"))
    result = AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]
    assert result.confidence == pytest.approx(0.0)


def test_custom_model_passed_through() -> None:
    client = _StubClient(_reply(True, 0.5, ""))
    AnthropicLLMJudge(client=client, model="claude-haiku-4-5").judge("c", "e")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    assert call["model"] == "claude-haiku-4-5"


def test_language_tagged_code_fence_is_tolerated() -> None:
    """Anthropic sometimes returns ` ```json ... ``` ` despite the prompt."""
    fenced = "```json\n" + _reply(True, 0.8, "fenced") + "\n```"
    client = _StubClient(fenced)
    result = AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]
    assert result.passed is True
    assert result.confidence == pytest.approx(0.8)


def test_bare_code_fence_is_tolerated() -> None:
    fenced = "```\n" + _reply(False, 0.2, "bare") + "\n```"
    client = _StubClient(fenced)
    result = AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]
    assert result.passed is False
    assert result.confidence == pytest.approx(0.2)


def test_refusal_inside_fence_still_raises() -> None:
    """A refusal wrapped in fences must surface as a parse error."""
    client = _StubClient("```\nI cannot answer that.\n```")
    with pytest.raises(ValueError):
        AnthropicLLMJudge(client=client).judge("c", "e")  # type: ignore[arg-type]
