"""Unit tests for the Anthropic-backed summariser — no network calls.

The real Anthropic API is not exercised in these tests (cost + flake).
Instead a stub client implements the same `messages.create` surface
the wrapper consumes, so we verify the request shape and response
parsing without leaving the process.

SPEC-REF: §4.1 (ingest step 7)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ctrldoc.ingest.summarizer import Summarizer
from ctrldoc.ingest.summarizer_anthropic import AnthropicSummarizer


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
    summariser = AnthropicSummarizer(client=_StubClient(""))  # type: ignore[arg-type]
    assert isinstance(summariser, Summarizer)


def test_summarize_passes_body_to_anthropic_messages() -> None:
    client = _StubClient("Two-sentence summary returned.")
    summariser = AnthropicSummarizer(client=client)  # type: ignore[arg-type]
    out = summariser.summarize("Body text to summarise.")
    assert out == "Two-sentence summary returned."
    call = client.messages.last_call
    assert call is not None
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 96
    assert "summarise" in call["system"].lower() or "summarize" in call["system"].lower()
    assert call["messages"] == [{"role": "user", "content": "Body text to summarise."}]


def test_summarize_empty_input_skips_api_call() -> None:
    client = _StubClient("should not be called")
    summariser = AnthropicSummarizer(client=client)  # type: ignore[arg-type]
    assert summariser.summarize("") == ""
    assert client.messages.last_call is None


def test_summarize_strips_response() -> None:
    client = _StubClient("  Trimmed summary.  \n")
    summariser = AnthropicSummarizer(client=client)  # type: ignore[arg-type]
    assert summariser.summarize("body") == "Trimmed summary."


def test_summarize_concatenates_multiple_text_blocks() -> None:
    class _MultiBlockClient:
        def __init__(self) -> None:
            self.messages = _MultiBlockMessages()

    class _MultiBlockMessages:
        last_call: dict[str, Any] | None = None

        def create(self, **kwargs: Any) -> _Message:
            self.last_call = kwargs
            return _Message(content=[_TextBlock(text="part a "), _TextBlock(text="part b")])

    client = _MultiBlockClient()
    summariser = AnthropicSummarizer(client=client)  # type: ignore[arg-type]
    assert summariser.summarize("body") == "part a part b"


def test_summarize_custom_model_and_max_tokens() -> None:
    client = _StubClient("ok")
    summariser = AnthropicSummarizer(
        client=client,  # type: ignore[arg-type]
        model="claude-opus-4-7",
        max_output_tokens=200,
    )
    summariser.summarize("body")
    call = client.messages.last_call
    assert call is not None
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 200
