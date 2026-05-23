"""Unit tests for the Anthropic-backed claim decomposer — no network.

Verifies request shape, JSON-only response parsing, and rejection
of malformed output. The real Anthropic API is never called.

SPEC-REF: §4.4
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from ctrldoc.verify.claim_decomposer import ClaimDecomposer
from ctrldoc.verify.claim_decomposer_anthropic import AnthropicClaimDecomposer


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


def test_satisfies_protocol() -> None:
    client = _StubClient(json.dumps({"claims": []}))
    assert isinstance(AnthropicClaimDecomposer(client=client), ClaimDecomposer)  # type: ignore[arg-type]


def test_parses_json_claims_list() -> None:
    client = _StubClient(json.dumps({"claims": ["Claim one.", "Claim two."]}))
    out = AnthropicClaimDecomposer(client=client).decompose("Some answer.")  # type: ignore[arg-type]
    assert out == ["Claim one.", "Claim two."]


def test_request_passes_answer_in_user_message() -> None:
    client = _StubClient(json.dumps({"claims": []}))
    AnthropicClaimDecomposer(client=client).decompose("The body to decompose.")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    assert call["messages"] == [{"role": "user", "content": "The body to decompose."}]


def test_request_system_prompt_constrains_output_to_json() -> None:
    client = _StubClient(json.dumps({"claims": []}))
    AnthropicClaimDecomposer(client=client).decompose("answer")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    assert "json" in call["system"].lower()
    assert "claim" in call["system"].lower()


def test_empty_answer_skips_api_call() -> None:
    client = _StubClient("should-not-be-used")
    out = AnthropicClaimDecomposer(client=client).decompose("")  # type: ignore[arg-type]
    assert out == []
    assert client.messages.last_call is None


def test_non_json_response_raises() -> None:
    client = _StubClient("not json")
    with pytest.raises(ValueError):
        AnthropicClaimDecomposer(client=client).decompose("answer")  # type: ignore[arg-type]


def test_invalid_schema_response_raises() -> None:
    client = _StubClient(json.dumps({"wrong_key": ["x"]}))
    with pytest.raises(ValueError):
        AnthropicClaimDecomposer(client=client).decompose("answer")  # type: ignore[arg-type]


def test_non_string_claims_filtered_out() -> None:
    client = _StubClient(json.dumps({"claims": ["valid", 42, None, "also valid"]}))
    out = AnthropicClaimDecomposer(client=client).decompose("answer")  # type: ignore[arg-type]
    assert out == ["valid", "also valid"]


def test_strips_whitespace_around_claims() -> None:
    client = _StubClient(json.dumps({"claims": ["  trimmed  ", "kept"]}))
    out = AnthropicClaimDecomposer(client=client).decompose("answer")  # type: ignore[arg-type]
    assert out == ["trimmed", "kept"]


def test_custom_model_passed_through() -> None:
    client = _StubClient(json.dumps({"claims": []}))
    AnthropicClaimDecomposer(client=client, model="claude-opus-4-7").decompose("a")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    assert call["model"] == "claude-opus-4-7"
