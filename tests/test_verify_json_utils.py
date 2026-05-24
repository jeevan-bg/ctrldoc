"""Unit tests for the shared `strip_code_fence` JSON-prelude helper.

The helper sits in `ctrldoc.verify.json_utils` and is shared by every
LLM-backed module that parses model output as JSON (Ollama judge,
Anthropic judge, Anthropic claim decomposer, Anthropic retrieval
planner). The four shapes the helper must round-trip are:

    1. Already-clean JSON (no fence) — pass through unchanged.
    2. Fenced JSON with a language tag (` ```json `).
    3. Fenced JSON with a trailing fence on its own line.
    4. Refusal / non-JSON prose surrounded by fences — body returned
       so the caller's `json.loads` can raise the correct error.

SPEC-REF: §6.5
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.family_determinism


def test_no_fence_returns_input_unchanged() -> None:
    from ctrldoc.verify.json_utils import strip_code_fence

    payload = '{"passed": true, "confidence": 0.9, "reasoning": "yes"}'
    assert strip_code_fence(payload) == payload


def test_strips_language_tagged_fence() -> None:
    from ctrldoc.verify.json_utils import strip_code_fence

    fenced = '```json\n{"passed": true, "confidence": 0.8, "reasoning": "yes"}\n```'
    assert strip_code_fence(fenced) == '{"passed": true, "confidence": 0.8, "reasoning": "yes"}'


def test_strips_bare_fence() -> None:
    from ctrldoc.verify.json_utils import strip_code_fence

    fenced = '```\n{"claims": ["one", "two"]}\n```'
    assert strip_code_fence(fenced) == '{"claims": ["one", "two"]}'


def test_strips_trailing_fence_only_when_present() -> None:
    """A leading fence with no trailing fence still yields the body."""
    from ctrldoc.verify.json_utils import strip_code_fence

    fenced = '```json\n{"steps": []}'
    assert strip_code_fence(fenced) == '{"steps": []}'


def test_refusal_inside_fence_returns_prose_for_caller_to_reject() -> None:
    """Refusals wrapped in fences still surface so json.loads raises."""
    from ctrldoc.verify.json_utils import strip_code_fence

    fenced = "```\nI cannot answer that.\n```"
    assert strip_code_fence(fenced) == "I cannot answer that."


def test_idempotent_on_repeated_application() -> None:
    from ctrldoc.verify.json_utils import strip_code_fence

    fenced = '```json\n{"x": 1}\n```'
    once = strip_code_fence(fenced)
    twice = strip_code_fence(once)
    assert once == twice == '{"x": 1}'


def test_handles_leading_and_trailing_whitespace() -> None:
    from ctrldoc.verify.json_utils import strip_code_fence

    fenced = '  ```json\n{"x": 1}\n```  '
    assert strip_code_fence(fenced) == '{"x": 1}'


def test_empty_string_returns_empty() -> None:
    from ctrldoc.verify.json_utils import strip_code_fence

    assert strip_code_fence("") == ""


def test_fence_only_returns_empty_body() -> None:
    """A model that emits ` ``` ` with no body yields the empty body."""
    from ctrldoc.verify.json_utils import strip_code_fence

    assert strip_code_fence("```\n```") == ""
