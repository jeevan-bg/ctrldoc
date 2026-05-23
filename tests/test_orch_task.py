"""Contract tests for the stateless task primitive.

`StatelessTaskRunner.run(task, output_model)` makes a single fresh
API call per invocation. Each call sees exactly the cacheable prefix,
the evidence pack, and the task-specific input — no other state. The
runner accepts a Pydantic `output_model` and returns a validated
instance; malformed JSON or schema mismatches surface as
`TaskOutputError`.

SPEC-REF: §3.1 (pillar 1 — stateless tasks), §4.5 (orchestrator)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel, ConfigDict

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.task import (
    StatelessTaskRunner,
    TaskInput,
    TaskOutputError,
)


class _Verdict(BaseModel):
    """Example structured output the runner is asked to produce."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence: float


@dataclass
class _StubClient:
    """Records every call and replays scripted responses in order."""

    responses: list[str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("stub exhausted")
        return self.responses.pop(0)


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are a verifier.",
        doc_skeleton="# §1 Aurora\nIntroduces consistent hashing.",
        entity_glossary="- **aurora** [system]",
    )


def _task(task_input: str = "verify: aurora supports consistent hashing.") -> TaskInput:
    return TaskInput(
        prefix=_prefix(),
        evidence_pack="Aurora uses consistent hashing across nodes.",
        task_input=task_input,
    )


# --- happy path ---


def test_run_returns_validated_output() -> None:
    client = _StubClient(responses=[json.dumps({"label": "verified", "confidence": 0.9})])
    runner = StatelessTaskRunner(client=client)

    result = runner.run(_task(), output_model=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.label == "verified"
    assert result.confidence == pytest.approx(0.9)
    assert len(client.calls) == 1


# --- statelessness ---


def test_runner_is_stateless_across_calls() -> None:
    """Two runs see independent client invocations — no leakage."""
    client = _StubClient(
        responses=[
            json.dumps({"label": "verified", "confidence": 0.9}),
            json.dumps({"label": "refused", "confidence": 0.1}),
        ]
    )
    runner = StatelessTaskRunner(client=client)

    r1 = runner.run(_task("claim-A"), output_model=_Verdict)
    r2 = runner.run(_task("claim-B"), output_model=_Verdict)

    assert r1.label == "verified"
    assert r2.label == "refused"
    assert len(client.calls) == 2
    # The user-message tail for the second call carries claim-B, not
    # claim-A — i.e. no implicit context bleed from the first call.
    assert "claim-A" in client.calls[0][1]
    assert "claim-A" not in client.calls[1][1]
    assert "claim-B" in client.calls[1][1]


def test_two_runner_instances_share_no_state() -> None:
    client = _StubClient(
        responses=[
            json.dumps({"label": "x", "confidence": 1.0}),
            json.dumps({"label": "y", "confidence": 0.0}),
        ]
    )
    a = StatelessTaskRunner(client=client)
    b = StatelessTaskRunner(client=client)
    ra = a.run(_task("from-a"), output_model=_Verdict)
    rb = b.run(_task("from-b"), output_model=_Verdict)
    assert ra.label == "x" and rb.label == "y"


# --- prompt layout ---


def test_run_composes_prompt_with_prefix_evidence_and_task() -> None:
    client = _StubClient(responses=[json.dumps({"label": "ok", "confidence": 0.5})])
    runner = StatelessTaskRunner(client=client)
    prefix = _prefix()
    task = TaskInput(
        prefix=prefix,
        evidence_pack="EVIDENCE-MARKER",
        task_input="TASK-MARKER",
    )
    runner.run(task, output_model=_Verdict)

    system_arg, user_arg = client.calls[0]
    # The system prompt sent to the client is the rendered cacheable
    # prefix verbatim — byte-stable for caching.
    assert system_arg == prefix.render()
    # The user message carries both evidence and task tails, each in
    # a labelled block so downstream parsing is unambiguous.
    assert "EVIDENCE-MARKER" in user_arg
    assert "TASK-MARKER" in user_arg
    assert user_arg.index("EVIDENCE-MARKER") < user_arg.index("TASK-MARKER")


def test_identical_inputs_produce_identical_client_args() -> None:
    """Cache-stability: same TaskInput → byte-identical system+user."""
    client = _StubClient(
        responses=[
            json.dumps({"label": "ok", "confidence": 0.5}),
            json.dumps({"label": "ok", "confidence": 0.5}),
        ]
    )
    runner = StatelessTaskRunner(client=client)
    task = _task("identical")
    runner.run(task, output_model=_Verdict)
    runner.run(task, output_model=_Verdict)
    assert client.calls[0] == client.calls[1]


# --- error surfaces ---


def test_malformed_json_raises_task_output_error() -> None:
    client = _StubClient(responses=["not-json"])
    runner = StatelessTaskRunner(client=client)
    with pytest.raises(TaskOutputError):
        runner.run(_task(), output_model=_Verdict)


def test_schema_validation_failure_raises_task_output_error() -> None:
    # Valid JSON, wrong shape (missing confidence; extra field).
    client = _StubClient(responses=[json.dumps({"label": "ok", "extra": True})])
    runner = StatelessTaskRunner(client=client)
    with pytest.raises(TaskOutputError):
        runner.run(_task(), output_model=_Verdict)


def test_response_with_fenced_json_block_is_accepted() -> None:
    """A model that wraps JSON in ```json fences shouldn't break the runner."""
    payload = json.dumps({"label": "verified", "confidence": 0.8})
    client = _StubClient(responses=[f"```json\n{payload}\n```"])
    runner = StatelessTaskRunner(client=client)
    result = runner.run(_task(), output_model=_Verdict)
    assert result.label == "verified"


# --- input invariants ---


def test_task_input_is_frozen() -> None:
    task = _task()
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError on frozen model
        task.task_input = "mutated"  # type: ignore[misc]


def test_empty_evidence_pack_is_permitted() -> None:
    """Some sub-tasks (planning, synthesis) carry no evidence — the
    runner must not require it."""
    client = _StubClient(responses=[json.dumps({"label": "ok", "confidence": 0.5})])
    runner = StatelessTaskRunner(client=client)
    task = TaskInput(
        prefix=_prefix(),
        evidence_pack="",
        task_input="plan something",
    )
    result = runner.run(task, output_model=_Verdict)
    assert result.label == "ok"
