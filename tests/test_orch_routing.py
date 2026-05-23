"""Tiered routing — picks local vs Opus per task kind.

SPEC §4.5 splits work across tiers: local 7B for decomposition,
simple judging, and easy scans; Opus for planning, hard judges,
and synthesis. The router is a passive lookup — it never makes
calls itself — so the orchestrator/playbook code stays the place
where tier policy lives.

SPEC-REF: §4.5 (orchestrator — tiered routing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.routing import (
    TaskClientRouter,
    TaskKind,
    Tier,
    default_tier_for,
)
from ctrldoc.orch.task import StatelessTaskRunner, TaskInput


@dataclass
class _StubClient:
    """Records every call for inspection."""

    label: str
    response: str = "{}"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


# --- default policy ---


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("decompose", "local"),
        ("judge_simple", "local"),
        ("scan", "local"),
        ("plan", "opus"),
        ("judge_hard", "opus"),
        ("synthesize", "opus"),
    ],
)
def test_default_tier_for_matches_spec_policy(kind: TaskKind, expected: Tier) -> None:
    assert default_tier_for(kind) == expected


def test_default_tier_for_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown task kind"):
        default_tier_for("invent_a_kind")  # type: ignore[arg-type]


# --- router lookup ---


def test_for_tier_returns_registered_subclient() -> None:
    local = _StubClient("local")
    opus = _StubClient("opus")
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]
    assert router.for_tier("local") is local
    assert router.for_tier("opus") is opus


def test_for_tier_unknown_tier_raises() -> None:
    router = TaskClientRouter(local=_StubClient("l"), opus=_StubClient("o"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown tier"):
        router.for_tier("bogus")  # type: ignore[arg-type]


def test_for_kind_uses_default_policy() -> None:
    local = _StubClient("local")
    opus = _StubClient("opus")
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]
    assert router.for_kind("decompose") is local
    assert router.for_kind("plan") is opus


def test_for_kind_accepts_a_policy_override() -> None:
    local = _StubClient("local")
    opus = _StubClient("opus")
    # Force everything to opus regardless of kind.
    router = TaskClientRouter(  # type: ignore[arg-type]
        local=local,
        opus=opus,
        policy=lambda kind: "opus",
    )
    assert router.for_kind("decompose") is opus
    assert router.for_kind("plan") is opus


def test_for_kind_policy_can_return_unknown_tier_and_raises_at_dispatch() -> None:
    local = _StubClient("local")
    opus = _StubClient("opus")
    router = TaskClientRouter(  # type: ignore[arg-type]
        local=local,
        opus=opus,
        policy=lambda kind: "bogus",  # type: ignore[return-value]
    )
    with pytest.raises(ValueError, match="unknown tier"):
        router.for_kind("plan")


# --- end-to-end with StatelessTaskRunner ---


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="sys",
        doc_skeleton="# §1",
        entity_glossary="- **e/1** [concept]",
    )


def test_runner_routes_to_local_for_judge_simple() -> None:
    local = _StubClient("local", response='{"label": "from-local"}')
    opus = _StubClient("opus", response='{"label": "from-opus"}')
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]

    runner = StatelessTaskRunner(client=router.for_kind("judge_simple"))
    task = TaskInput(prefix=_prefix(), evidence_pack="ev", task_input="t")
    result = runner.run(task, output_model=_Verdict)

    assert result.label == "from-local"
    assert len(local.calls) == 1
    assert opus.calls == []


def test_runner_routes_to_opus_for_synthesize() -> None:
    local = _StubClient("local", response='{"label": "from-local"}')
    opus = _StubClient("opus", response='{"label": "from-opus"}')
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]

    runner = StatelessTaskRunner(client=router.for_kind("synthesize"))
    task = TaskInput(prefix=_prefix(), evidence_pack="ev", task_input="t")
    result = runner.run(task, output_model=_Verdict)

    assert result.label == "from-opus"
    assert len(opus.calls) == 1
    assert local.calls == []


def test_two_calls_with_different_kinds_go_to_different_clients() -> None:
    """One router, one run per tier — no cross-contamination."""
    local = _StubClient("local", response='{"label": "ok"}')
    opus = _StubClient("opus", response='{"label": "ok"}')
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]

    task = TaskInput(prefix=_prefix(), evidence_pack="ev", task_input="t")

    StatelessTaskRunner(client=router.for_kind("scan")).run(task, output_model=_Verdict)
    StatelessTaskRunner(client=router.for_kind("plan")).run(task, output_model=_Verdict)

    assert len(local.calls) == 1
    assert len(opus.calls) == 1


# --- router is passive ---


def test_router_does_not_call_subclients_on_lookup() -> None:
    local = _StubClient("local")
    opus = _StubClient("opus")
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]
    router.for_tier("local")
    router.for_kind("plan")
    # Neither lookup should invoke a sub-client.
    assert local.calls == []
    assert opus.calls == []


def test_required_tiers_must_be_provided() -> None:
    """The constructor enforces that both spec-mandated tiers are present —
    a routing decision that silently dropped a tier would be a hidden bug."""
    with pytest.raises(TypeError):
        TaskClientRouter(local=_StubClient("l"))  # type: ignore[call-arg]


def test_subclients_used_through_router_remain_independent() -> None:
    """A side-channel sanity check: routing one call doesn't perturb the
    other sub-client's state."""
    local = _StubClient("local", response='{"label": "ok"}')
    opus = _StubClient("opus", response='{"label": "ok"}')
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]
    task = TaskInput(prefix=_prefix(), evidence_pack="ev", task_input="t-A")
    StatelessTaskRunner(client=router.for_kind("decompose")).run(task, output_model=_Verdict)
    # Opus client's call log is untouched.
    assert opus.calls == []
    # Local client received exactly one call with the expected user payload.
    assert local.calls[0][1].endswith("</task>")
    assert any("t-A" in arg for arg in local.calls[0])


# --- type-hint ergonomics ---


def test_router_subclient_satisfies_task_client_protocol() -> None:
    """Whatever the router hands back must conform to TaskClient — otherwise
    runners can't accept it."""
    from ctrldoc.orch.task import TaskClient

    local = _StubClient("local")
    opus = _StubClient("opus")
    router = TaskClientRouter(local=local, opus=opus)  # type: ignore[arg-type]
    assert isinstance(router.for_tier("local"), TaskClient)
    assert isinstance(router.for_kind("plan"), TaskClient)


# --- exhaustiveness ---


def test_default_policy_covers_every_documented_kind() -> None:
    """Lock in the §4.5 tier table — adding a kind without a tier should fail."""
    from typing import get_args

    for kind in get_args(TaskKind):
        # Should not raise.
        default_tier_for(kind)


_ = Any  # keep import for forward use by readers of this file
