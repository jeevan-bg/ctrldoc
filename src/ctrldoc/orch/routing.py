"""Tiered routing — pick a `TaskClient` per task kind.

The orchestrator splits work across two tiers per SPEC §4.5:

  - **local** (Qwen2.5-7B via Ollama): claim decomposition, simple
    judging, easy scans.
  - **opus** (Claude Opus via Anthropic): planning, hard / escalated
    judges, final synthesis.

`TaskClientRouter` is a passive lookup. It holds one sub-client per
tier and returns the right one given either a `Tier` or a `TaskKind`.
The policy that maps kind → tier lives in `default_tier_for` (or a
callable override at construction time). Routing decisions stay
explicit in the playbook code — the router never makes calls itself.

SPEC-REF: §4.5
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ctrldoc.orch.task import TaskClient

Tier = Literal["local", "opus"]
TaskKind = Literal[
    "decompose",
    "judge_simple",
    "judge_hard",
    "plan",
    "scan",
    "synthesize",
]

_DEFAULT_TIER_FOR_KIND: dict[TaskKind, Tier] = {
    "decompose": "local",
    "judge_simple": "local",
    "scan": "local",
    "judge_hard": "opus",
    "plan": "opus",
    "synthesize": "opus",
}


def default_tier_for(kind: TaskKind) -> Tier:
    """Map a task kind to its default tier per §4.5."""
    try:
        return _DEFAULT_TIER_FOR_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"unknown task kind: {kind!r}") from exc


class TaskClientRouter:
    """Holds one `TaskClient` per tier and dispatches lookups."""

    def __init__(
        self,
        *,
        local: TaskClient,
        opus: TaskClient,
        policy: Callable[[TaskKind], Tier] | None = None,
    ) -> None:
        self._clients: dict[Tier, TaskClient] = {"local": local, "opus": opus}
        self._policy = policy or default_tier_for

    def for_tier(self, tier: Tier) -> TaskClient:
        try:
            return self._clients[tier]
        except KeyError as exc:
            raise ValueError(f"unknown tier: {tier!r}") from exc

    def for_kind(self, kind: TaskKind) -> TaskClient:
        return self.for_tier(self._policy(kind))


__all__ = [
    "TaskClientRouter",
    "TaskKind",
    "Tier",
    "default_tier_for",
]
