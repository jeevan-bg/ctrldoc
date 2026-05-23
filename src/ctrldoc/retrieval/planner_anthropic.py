"""Anthropic-backed retrieval planner.

The planner reads the cacheable prefix from `CacheablePrefix` and
attaches `cache_control` to the skeleton + glossary blocks so the
Anthropic prompt cache keys on the same byte prefix across every
planner call in a session (SPEC §3.1 pillar 2).

The model is instructed to return a single JSON object that
validates straight against the retrieval `Plan` schema.

SPEC-REF: §4.3, §3.1
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.retrieval.dsl import Plan

if TYPE_CHECKING:
    from anthropic import Anthropic


_PLANNER_INSTRUCTIONS = (
    "You are a retrieval planner. Read the document skeleton and entity "
    "glossary above, then translate the user's question into a JSON object "
    "with one field `steps`, a list of retrieval-DSL steps. Each step is "
    "one of:\n"
    '  {"op": "search", "query": <str>, "view": <dense|lexical|entity>, "k": <int>}\n'
    '  {"op": "expand", "section_id": <str>}\n'
    '  {"op": "neighbors", "entity_id": <str>, "hops": <int>}\n'
    "Return only the JSON object. Do not include code fences, prose, or any "
    "text outside the JSON."
)


class AnthropicPlanner:
    """Planner backed by Anthropic Messages with prompt-cache markers."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_output_tokens: int = 1024,
        client: Anthropic | None = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = client

    def _ensure_client(self) -> Anthropic:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def plan(self, prefix: CacheablePrefix, query: str) -> Plan:
        if not query.strip():
            return Plan(steps=[])
        client = self._ensure_client()
        system_blocks = _build_system_blocks(prefix)
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_output_tokens,
            system=cast(Any, system_blocks),
            messages=[{"role": "user", "content": query}],
        )
        return _parse_plan(_extract_text(message))


def _build_system_blocks(prefix: CacheablePrefix) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if prefix.system_prompt:
        blocks.append({"type": "text", "text": prefix.system_prompt})
    if prefix.doc_skeleton:
        blocks.append({"type": "text", "text": "# Document skeleton\n\n" + prefix.doc_skeleton})
    if prefix.entity_glossary:
        blocks.append({"type": "text", "text": "# Entity glossary\n\n" + prefix.entity_glossary})
    blocks.append({"type": "text", "text": _PLANNER_INSTRUCTIONS})
    # Cache-control the last block of the stable prefix so the cache keys on
    # everything before the per-call user message.
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return blocks


def _extract_text(message: object) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _parse_plan(text: str) -> Plan:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"planner returned non-JSON output: {text[:80]!r}") from exc
    try:
        return Plan.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"planner output did not validate as Plan: {exc}") from exc


__all__ = ["AnthropicPlanner"]
