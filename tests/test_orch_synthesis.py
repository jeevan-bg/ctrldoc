"""Synthesis primitive — reduce over structured findings.

A playbook's terminal step takes the JSON results from a stateless
fan-out and feeds them to a single Opus call that emits the final
synthesised report. Per §3.1 pillar 1 the synthesis call never sees
the raw document — only the cacheable prefix plus the structured
findings JSON. The runner is intentionally one-shot (no fan-out, no
batching) so cost stays predictable.

SPEC-REF: §3.1 (pillar 1), §4.5 (orchestrator — synthesis)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.synthesis import (
    SynthesisInput,
    SynthesisRunner,
)
from ctrldoc.orch.task import TaskOutputError


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    headline: str
    sections: list[str]


@dataclass
class _StubClient:
    response: str = "{}"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are a synthesis writer.",
        doc_skeleton="# §1\n\nbody",
        entity_glossary="- **e/1** [concept]",
    )


def _findings() -> list[dict[str, object]]:
    return [
        {"id": "f-1", "claim": "Aurora uses consistent hashing.", "severity": "info"},
        {"id": "f-2", "claim": "Sections lack rollback guidance.", "severity": "warn"},
    ]


# --- happy path ---


def test_synthesis_round_trips_a_report() -> None:
    client = _StubClient(
        response=json.dumps({"headline": "Two findings", "sections": ["f-1", "f-2"]})
    )
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(
        prefix=_prefix(),
        findings=_findings(),
        instruction="Write a short report covering each finding.",
    )
    report = runner.run(task, output_model=_Report)
    assert report.headline == "Two findings"
    assert report.sections == ["f-1", "f-2"]
    # Exactly one API call — synthesis is reduce, not map.
    assert len(client.calls) == 1


# --- prompt layout ---


def test_system_message_is_the_rendered_prefix() -> None:
    """Cache stability: synthesis shares the prefix with fan-out calls
    in the same session."""
    prefix = _prefix()
    client = _StubClient(response=json.dumps({"headline": "x", "sections": []}))
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(
        prefix=prefix,
        findings=_findings(),
        instruction="summarize",
    )
    runner.run(task, output_model=_Report)
    assert client.calls[0][0] == prefix.render()


def test_user_message_carries_findings_json_and_instruction() -> None:
    client = _StubClient(response=json.dumps({"headline": "x", "sections": []}))
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(
        prefix=_prefix(),
        findings=[{"id": "f-1", "claim": "MARKER-CLAIM"}],
        instruction="INSTRUCTION-MARKER",
    )
    runner.run(task, output_model=_Report)
    user = client.calls[0][1]
    assert "MARKER-CLAIM" in user
    assert "INSTRUCTION-MARKER" in user
    # Findings appear before the instruction so the model reads the
    # data first and the ask last.
    assert user.index("MARKER-CLAIM") < user.index("INSTRUCTION-MARKER")


def test_findings_serialisation_preserves_input_order() -> None:
    client = _StubClient(response=json.dumps({"headline": "x", "sections": []}))
    runner = SynthesisRunner(client=client)
    findings = [{"id": f"f-{i}", "claim": f"claim-{i}"} for i in range(5)]
    task = SynthesisInput(prefix=_prefix(), findings=findings, instruction="x")
    runner.run(task, output_model=_Report)
    user = client.calls[0][1]
    positions = [user.index(f"claim-{i}") for i in range(5)]
    assert positions == sorted(positions), "findings appear in input order"


# --- byte-stable cacheability ---


def test_identical_inputs_produce_byte_identical_client_args() -> None:
    client = _StubClient(response=json.dumps({"headline": "x", "sections": []}))
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(
        prefix=_prefix(),
        findings=_findings(),
        instruction="instr",
    )
    runner.run(task, output_model=_Report)
    runner.run(task, output_model=_Report)
    assert client.calls[0] == client.calls[1]


# --- edge cases ---


def test_empty_findings_still_makes_one_call() -> None:
    """A reduce over zero findings is still a valid synthesis call —
    the model may emit an 'empty report'. Don't short-circuit."""
    client = _StubClient(response=json.dumps({"headline": "no findings", "sections": []}))
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(
        prefix=_prefix(),
        findings=[],
        instruction="summarize",
    )
    report = runner.run(task, output_model=_Report)
    assert report.headline == "no findings"
    assert len(client.calls) == 1


def test_empty_instruction_rejected_at_input_construction() -> None:
    """A synthesis call needs an explicit ask; blank instructions would
    silently produce undefined behaviour."""
    with pytest.raises(ValidationError):
        SynthesisInput(prefix=_prefix(), findings=_findings(), instruction="   ")


# --- error surfaces ---


def test_malformed_json_response_raises_task_output_error() -> None:
    client = _StubClient(response="not-json")
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(prefix=_prefix(), findings=_findings(), instruction="x")
    with pytest.raises(TaskOutputError):
        runner.run(task, output_model=_Report)


def test_schema_validation_failure_raises_task_output_error() -> None:
    # Valid JSON, wrong shape (missing sections; extra field).
    client = _StubClient(response=json.dumps({"headline": "x", "extra": True}))
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(prefix=_prefix(), findings=_findings(), instruction="x")
    with pytest.raises(TaskOutputError):
        runner.run(task, output_model=_Report)


def test_fenced_response_is_accepted() -> None:
    payload = json.dumps({"headline": "ok", "sections": []})
    client = _StubClient(response=f"```json\n{payload}\n```")
    runner = SynthesisRunner(client=client)
    task = SynthesisInput(prefix=_prefix(), findings=_findings(), instruction="x")
    report = runner.run(task, output_model=_Report)
    assert report.headline == "ok"


# --- input invariants ---


def test_synthesis_input_is_frozen() -> None:
    task = SynthesisInput(prefix=_prefix(), findings=_findings(), instruction="x")
    with pytest.raises(ValidationError):
        task.instruction = "mutated"  # type: ignore[misc]


def test_findings_must_be_a_list() -> None:
    with pytest.raises(ValidationError):
        SynthesisInput(prefix=_prefix(), findings={"not": "a list"}, instruction="x")  # type: ignore[arg-type]
