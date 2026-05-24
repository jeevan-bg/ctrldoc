"""Unit tests for the Anthropic-backed planner — no network calls.

A stub client implements the same `messages.create` surface the
wrapper consumes, so we verify the request shape (cache-control on
the skeleton + glossary, system prompt content) and the response
parsing without leaving the process.

SPEC-REF: §4.3, §3.1 (cacheable prefix)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.retrieval.dsl import Plan
from ctrldoc.retrieval.planner import Planner
from ctrldoc.retrieval.planner_anthropic import AnthropicPlanner


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


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are a careful retrieval planner.",
        doc_skeleton="# A\n\nbody\n",
        entity_glossary="- **ent/x** [concept]\n",
    )


def _reply(steps: list[dict[str, Any]]) -> str:
    return json.dumps({"steps": steps})


def test_satisfies_protocol() -> None:
    assert isinstance(AnthropicPlanner(client=_StubClient(_reply([]))), Planner)  # type: ignore[arg-type]


def test_plan_round_trips_a_two_step_plan() -> None:
    reply = _reply(
        [
            {"op": "search", "query": "Aurora consistency", "view": "dense", "k": 8},
            {"op": "expand", "section_id": "sec/intro"},
        ]
    )
    client = _StubClient(reply)
    plan = AnthropicPlanner(client=client).plan(_prefix(), "Aurora consistency?")  # type: ignore[arg-type]
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 2
    assert plan.steps[0].op == "search"
    assert plan.steps[1].op == "expand"


def test_request_sends_cache_control_on_prefix_blocks() -> None:
    client = _StubClient(_reply([]))
    AnthropicPlanner(client=client).plan(_prefix(), "q")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    system = call["system"]
    assert isinstance(system, list)
    # Last cacheable block must carry cache_control.
    cache_marked = [block for block in system if block.get("cache_control") is not None]
    assert cache_marked, "expected at least one cache-controlled prefix block"


def test_request_includes_skeleton_and_glossary_in_system() -> None:
    prefix = _prefix()
    client = _StubClient(_reply([]))
    AnthropicPlanner(client=client).plan(prefix, "q")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    rendered = json.dumps(call["system"])
    assert "Document skeleton" in rendered
    assert "Entity glossary" in rendered
    assert "ent/x" in rendered


def test_request_user_message_carries_query() -> None:
    client = _StubClient(_reply([]))
    AnthropicPlanner(client=client).plan(_prefix(), "what is X?")  # type: ignore[arg-type]
    call = client.messages.last_call
    assert call is not None
    assert call["messages"] == [{"role": "user", "content": "what is X?"}]


def test_response_must_be_valid_json() -> None:
    client = _StubClient("not json at all")
    with pytest.raises(ValueError):
        AnthropicPlanner(client=client).plan(_prefix(), "q")  # type: ignore[arg-type]


def test_response_must_validate_against_plan_schema() -> None:
    client = _StubClient(_reply([{"op": "imaginary", "x": 1}]))
    with pytest.raises(ValueError):
        AnthropicPlanner(client=client).plan(_prefix(), "q")  # type: ignore[arg-type]


def test_empty_query_short_circuits_without_api_call() -> None:
    client = _StubClient("should not be called")
    plan = AnthropicPlanner(client=client).plan(_prefix(), "")  # type: ignore[arg-type]
    assert plan.steps == []
    assert client.messages.last_call is None


def test_custom_model_and_max_tokens() -> None:
    client = _StubClient(_reply([]))
    AnthropicPlanner(
        client=client,  # type: ignore[arg-type]
        model="claude-opus-4-7",
        max_output_tokens=512,
    ).plan(_prefix(), "q")
    call = client.messages.last_call
    assert call is not None
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 512


def test_language_tagged_code_fence_is_tolerated() -> None:
    """The planner round-trips ` ```json ... ``` ` wrappers."""
    fenced = "```json\n" + _reply([{"op": "expand", "section_id": "sec/x"}]) + "\n```"
    client = _StubClient(fenced)
    plan = AnthropicPlanner(client=client).plan(_prefix(), "q")  # type: ignore[arg-type]
    assert len(plan.steps) == 1
    assert plan.steps[0].op == "expand"


def test_bare_code_fence_is_tolerated() -> None:
    fenced = "```\n" + _reply([]) + "\n```"
    client = _StubClient(fenced)
    plan = AnthropicPlanner(client=client).plan(_prefix(), "q")  # type: ignore[arg-type]
    assert plan.steps == []


def test_refusal_inside_fence_still_raises() -> None:
    client = _StubClient("```\nI cannot answer that.\n```")
    with pytest.raises(ValueError):
        AnthropicPlanner(client=client).plan(_prefix(), "q")  # type: ignore[arg-type]
