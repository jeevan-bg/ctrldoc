"""Anthropic-backed TaskClient — no network.

The wrapper attaches `cache_control: ephemeral` to the system block
so the Anthropic prompt cache keys on the byte-stable prefix produced
by `CacheablePrefix.render()`. Stub `messages.create` lets us inspect
the request shape and confirm the response text is returned verbatim.

SPEC-REF: §3.1 (pillar 2 — shared prompt cache), §4.5 (orchestrator)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.task import (
    StatelessTaskRunner,
    TaskClient,
    TaskInput,
)
from ctrldoc.orch.task_anthropic import AnthropicTaskClient


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
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Message:
        self.calls.append(kwargs)
        return _Message(content=[_TextBlock(text=self._reply_text)])


class _StubClient:
    def __init__(self, reply_text: str) -> None:
        self.messages = _StubMessages(reply_text)


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are a strict structured-output engine.",
        doc_skeleton="# §1\n\nbody",
        entity_glossary="- **e/1** [concept]",
    )


# --- protocol conformance ---


def test_satisfies_task_client_protocol() -> None:
    client = AnthropicTaskClient(client=_StubClient("x"))  # type: ignore[arg-type]
    assert isinstance(client, TaskClient)


# --- request shape ---


def test_system_block_has_cache_control_ephemeral() -> None:
    stub = _StubClient("response-text")
    AnthropicTaskClient(client=stub).call(system="SYS", user="USR")  # type: ignore[arg-type]
    call = stub.messages.calls[0]
    system = call["system"]
    assert isinstance(system, list) and system, "system must be a non-empty list of blocks"
    # The last block of the cacheable prefix carries the marker.
    assert system[-1]["cache_control"] == {"type": "ephemeral"}


def test_system_block_carries_the_system_string_verbatim() -> None:
    stub = _StubClient("ok")
    AnthropicTaskClient(client=stub).call(system="SYS-VERBATIM", user="u")  # type: ignore[arg-type]
    rendered = "".join(block["text"] for block in stub.messages.calls[0]["system"])
    assert "SYS-VERBATIM" in rendered


def test_user_message_carries_user_string_verbatim() -> None:
    stub = _StubClient("ok")
    AnthropicTaskClient(client=stub).call(system="s", user="USER-PAYLOAD")  # type: ignore[arg-type]
    assert stub.messages.calls[0]["messages"] == [{"role": "user", "content": "USER-PAYLOAD"}]


def test_identical_systems_produce_byte_identical_request_blocks() -> None:
    """Cache stability: two calls with the same system string emit
    byte-identical system blocks (so the Anthropic cache keys on them
    identically)."""
    stub = _StubClient("ok")
    backend = AnthropicTaskClient(client=stub)  # type: ignore[arg-type]
    backend.call(system="prefix", user="a")
    backend.call(system="prefix", user="b")
    assert stub.messages.calls[0]["system"] == stub.messages.calls[1]["system"]


# --- response surface ---


def test_call_returns_text_concatenated_from_response_blocks() -> None:
    @dataclass
    class _Msg:
        content: list[_TextBlock]

    class _MultiBlockMessages(_StubMessages):
        def create(self, **kwargs: Any) -> _Msg:  # type: ignore[override]
            self.calls.append(kwargs)
            return _Msg(content=[_TextBlock(text="part-A"), _TextBlock(text="part-B")])

    class _MBClient:
        def __init__(self) -> None:
            self.messages = _MultiBlockMessages("")

    client = _MBClient()
    out = AnthropicTaskClient(client=client).call(system="s", user="u")  # type: ignore[arg-type]
    assert out == "part-Apart-B"


# --- configuration ---


def test_custom_model_and_max_tokens_forwarded_to_api() -> None:
    stub = _StubClient("ok")
    AnthropicTaskClient(
        client=stub,  # type: ignore[arg-type]
        model="claude-opus-4-7",
        max_output_tokens=128,
    ).call(system="s", user="u")
    call = stub.messages.calls[0]
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 128


def test_lazy_client_construction_does_not_import_when_stub_provided() -> None:
    """If a stub client is supplied the wrapper must not touch the
    anthropic package — keeps the unit-test path hermetic."""
    stub = _StubClient("ok")
    backend = AnthropicTaskClient(client=stub)  # type: ignore[arg-type]
    backend.call(system="s", user="u")
    # _client is the stub, not an Anthropic() instance.
    assert backend._client is stub  # type: ignore[attr-defined]


# --- end-to-end with StatelessTaskRunner ---


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    confidence: float


def test_runner_with_anthropic_client_round_trips_a_verdict() -> None:
    stub = _StubClient('{"label": "verified", "confidence": 0.8}')
    backend = AnthropicTaskClient(client=stub)  # type: ignore[arg-type]
    runner = StatelessTaskRunner(client=backend)
    task = TaskInput(
        prefix=_prefix(),
        evidence_pack="Aurora uses consistent hashing.",
        task_input="Aurora supports consistent hashing.",
    )
    result = runner.run(task, output_model=_Verdict)
    assert result.label == "verified"
    assert result.confidence == pytest.approx(0.8)
    # The system text passed to Anthropic is the rendered prefix verbatim.
    rendered = "".join(block["text"] for block in stub.messages.calls[0]["system"])
    assert rendered == task.prefix.render()
