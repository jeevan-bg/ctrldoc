"""Contract tests for the batched task runner.

When K sub-tasks share the same `{prefix, evidence_pack}`, batching
collapses K separate API calls (K prefix-cost + K tail) into one
(1 prefix + K tail). The runner asks the model to emit a JSON
array of `{id, output}` entries; results are looked up by `id` so
the model is allowed to reorder. Missing, extra, or duplicate `id`s
all raise `TaskOutputError`.

SPEC-REF: §4.5 (orchestrator — batching)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.batch import (
    BatchedTaskInput,
    BatchedTaskRunner,
    BatchItem,
)
from ctrldoc.orch.task import TaskOutputError


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    confidence: float


@dataclass
class _StubClient:
    response: str = "[]"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="batch judge",
        doc_skeleton="# §1",
        entity_glossary="- **e/1** [concept]",
    )


def _items(ids: list[str]) -> list[BatchItem]:
    return [BatchItem(id=i, task_input=f"task for {i}") for i in ids]


def _batch_response(items: list[tuple[str, dict[str, object]]]) -> str:
    return json.dumps([{"id": i, "output": out} for i, out in items])


# --- happy path ---


def test_batched_run_returns_outputs_in_input_order() -> None:
    response = _batch_response(
        [
            ("a", {"label": "ok", "confidence": 0.9}),
            ("b", {"label": "ok", "confidence": 0.8}),
            ("c", {"label": "ok", "confidence": 0.7}),
        ]
    )
    client = _StubClient(response=response)
    runner = BatchedTaskRunner(client=client)

    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a", "b", "c"]))
    results = runner.run(task, output_model=_Verdict)

    assert [r.confidence for r in results] == pytest.approx([0.9, 0.8, 0.7])
    # K items, one API call.
    assert len(client.calls) == 1


def test_batched_run_uses_id_lookup_not_position() -> None:
    """Model is allowed to reorder; the runner reassembles by id."""
    response = _batch_response(
        [
            ("b", {"label": "ok", "confidence": 0.5}),
            ("a", {"label": "ok", "confidence": 0.1}),
        ]
    )
    client = _StubClient(response=response)
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a", "b"]))

    results = runner.run(task, output_model=_Verdict)

    # Output order matches input order even though the model reordered.
    assert results[0].confidence == pytest.approx(0.1)
    assert results[1].confidence == pytest.approx(0.5)


def test_batched_run_only_calls_client_once_per_batch() -> None:
    response = _batch_response(
        [
            ("a", {"label": "ok", "confidence": 0.5}),
            ("b", {"label": "ok", "confidence": 0.5}),
        ]
    )
    client = _StubClient(response=response)
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a", "b"]))
    runner.run(task, output_model=_Verdict)
    assert len(client.calls) == 1


# --- empty input ---


def test_empty_batch_short_circuits_without_client_call() -> None:
    client = _StubClient(response="[]")
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=[])
    results = runner.run(task, output_model=_Verdict)
    assert results == []
    assert client.calls == []


# --- error surfaces ---


def test_missing_id_in_response_raises_task_output_error() -> None:
    response = _batch_response([("a", {"label": "ok", "confidence": 0.5})])
    client = _StubClient(response=response)
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a", "b"]))
    with pytest.raises(TaskOutputError, match="missing"):
        runner.run(task, output_model=_Verdict)


def test_extra_id_in_response_raises_task_output_error() -> None:
    response = _batch_response(
        [
            ("a", {"label": "ok", "confidence": 0.5}),
            ("b", {"label": "ok", "confidence": 0.5}),
            ("ghost", {"label": "ok", "confidence": 0.5}),
        ]
    )
    client = _StubClient(response=response)
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a", "b"]))
    with pytest.raises(TaskOutputError, match="unexpected"):
        runner.run(task, output_model=_Verdict)


def test_duplicate_id_in_response_raises_task_output_error() -> None:
    """Two entries claiming the same id — ambiguous, must fail loud."""
    response = _batch_response(
        [
            ("a", {"label": "ok", "confidence": 0.5}),
            ("a", {"label": "ok", "confidence": 0.9}),
        ]
    )
    client = _StubClient(response=response)
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a"]))
    with pytest.raises(TaskOutputError, match="duplicate"):
        runner.run(task, output_model=_Verdict)


def test_response_must_be_a_json_array() -> None:
    client = _StubClient(response='{"not": "an array"}')
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a"]))
    with pytest.raises(TaskOutputError):
        runner.run(task, output_model=_Verdict)


def test_malformed_json_raises_task_output_error() -> None:
    client = _StubClient(response="not-json")
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a"]))
    with pytest.raises(TaskOutputError):
        runner.run(task, output_model=_Verdict)


def test_per_item_schema_failure_raises_task_output_error() -> None:
    """One bad output in the array → the whole batch fails."""
    response = json.dumps(
        [
            {"id": "a", "output": {"label": "ok", "confidence": 0.5}},
            {"id": "b", "output": {"label": "ok"}},  # missing confidence
        ]
    )
    client = _StubClient(response=response)
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a", "b"]))
    with pytest.raises(TaskOutputError):
        runner.run(task, output_model=_Verdict)


# --- input invariants ---


def test_duplicate_input_ids_rejected_at_construction() -> None:
    """Two input items sharing an id would make the lookup ambiguous —
    fail at construction so playbook code can't accidentally batch with
    duplicate keys."""
    with pytest.raises(ValidationError):
        BatchedTaskInput(
            prefix=_prefix(),
            evidence_pack="ev",
            items=[
                BatchItem(id="a", task_input="t1"),
                BatchItem(id="a", task_input="t2"),
            ],
        )


def test_batched_task_input_is_frozen() -> None:
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a"]))
    with pytest.raises(ValidationError):
        task.evidence_pack = "mutated"  # type: ignore[misc]


# --- prompt layout ---


def test_user_message_includes_evidence_and_every_item_id() -> None:
    client = _StubClient(
        response=_batch_response(
            [
                ("a", {"label": "ok", "confidence": 0.5}),
                ("b", {"label": "ok", "confidence": 0.5}),
            ]
        )
    )
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(
        prefix=_prefix(),
        evidence_pack="EVIDENCE-MARKER",
        items=[
            BatchItem(id="a", task_input="TASK-A"),
            BatchItem(id="b", task_input="TASK-B"),
        ],
    )
    runner.run(task, output_model=_Verdict)
    _, user_arg = client.calls[0]
    assert "EVIDENCE-MARKER" in user_arg
    assert "TASK-A" in user_arg
    assert "TASK-B" in user_arg
    # The ids the model must echo back appear in the prompt.
    assert '"a"' in user_arg
    assert '"b"' in user_arg


def test_system_message_is_the_rendered_prefix() -> None:
    """Cache stability: the system carries the prefix verbatim — identical
    across batched and non-batched calls in the same session."""
    prefix = _prefix()
    client = _StubClient(response=_batch_response([("a", {"label": "ok", "confidence": 0.5})]))
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=prefix, evidence_pack="ev", items=_items(["a"]))
    runner.run(task, output_model=_Verdict)
    system_arg, _ = client.calls[0]
    assert system_arg == prefix.render()


# --- fenced response tolerated ---


def test_fenced_json_array_response_is_accepted() -> None:
    inner = _batch_response([("a", {"label": "ok", "confidence": 0.5})])
    client = _StubClient(response=f"```json\n{inner}\n```")
    runner = BatchedTaskRunner(client=client)
    task = BatchedTaskInput(prefix=_prefix(), evidence_pack="ev", items=_items(["a"]))
    results = runner.run(task, output_model=_Verdict)
    assert results[0].confidence == pytest.approx(0.5)
