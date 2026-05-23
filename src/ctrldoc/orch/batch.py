"""Batched task primitive — one API call covers K items sharing evidence.

When K sub-tasks share the same `{prefix, evidence_pack}` (e.g. one
checklist evaluated over one section's evidence pack), it's wasteful
to issue K separate calls. `BatchedTaskRunner` collapses them into a
single call: the model receives the evidence once and a JSON-typed
list of K item descriptors, and must return an equal-length list of
`{id, output}` entries. The runner reassembles outputs by `id`, so
the model is free to reorder.

`output_model` is applied to each entry's `output`. Any missing,
extra, or duplicate ids — or any per-item schema failure — raises
`TaskOutputError`; partial batches are never returned silently.

SPEC-REF: §4.5 (orchestrator — batching)
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.task import TaskClient, TaskOutputError, _extract_json

T = TypeVar("T", bound=BaseModel)


class BatchItem(BaseModel):
    """One item in a batched task. `id` is the lookup key for the response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    task_input: str


class BatchedTaskInput(BaseModel):
    """Inputs for one batched call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix: CacheablePrefix
    evidence_pack: str
    items: list[BatchItem]

    @model_validator(mode="after")
    def _unique_ids(self) -> BatchedTaskInput:
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError(f"duplicate batch item id: {item.id!r}")
            seen.add(item.id)
        return self


class BatchedTaskRunner:
    """One API call per batch; results returned in input order."""

    def __init__(self, *, client: TaskClient) -> None:
        self._client = client

    def run(self, task: BatchedTaskInput, *, output_model: type[T]) -> list[T]:
        if not task.items:
            return []
        system = task.prefix.render()
        user = self._build_user_message(task)
        raw = self._client.call(system=system, user=user)
        return self._parse(raw, task.items, output_model)

    @staticmethod
    def _build_user_message(task: BatchedTaskInput) -> str:
        items_payload = json.dumps(
            [{"id": item.id, "task": item.task_input} for item in task.items],
            ensure_ascii=False,
        )
        return (
            f"<evidence>\n{task.evidence_pack}\n</evidence>\n\n"
            f"<items>\n{items_payload}\n</items>\n\n"
            "Return a JSON array. For each input id emit one entry "
            '{"id": <id>, "output": <object>} — exactly once, no extras.'
        )

    @staticmethod
    def _parse(
        raw: str,
        items: list[BatchItem],
        output_model: type[T],
    ) -> list[T]:
        payload = _extract_json(raw).strip()
        try:
            data: Any = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TaskOutputError(f"response is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise TaskOutputError("response must be a JSON array")

        by_id: dict[str, Any] = {}
        for entry in data:
            if not isinstance(entry, dict):
                raise TaskOutputError(f"batch entry must be an object: {entry!r}")
            entry_id = entry.get("id")
            if not isinstance(entry_id, str):
                raise TaskOutputError(f"batch entry missing string id: {entry!r}")
            if entry_id in by_id:
                raise TaskOutputError(f"duplicate id in response: {entry_id!r}")
            by_id[entry_id] = entry.get("output")

        expected = {item.id for item in items}
        unexpected = set(by_id) - expected
        if unexpected:
            raise TaskOutputError(f"unexpected ids in response: {sorted(unexpected)}")
        missing = expected - set(by_id)
        if missing:
            raise TaskOutputError(f"missing ids in response: {sorted(missing)}")

        results: list[T] = []
        for item in items:
            try:
                results.append(output_model.model_validate(by_id[item.id]))
            except ValidationError as exc:
                raise TaskOutputError(
                    f"batch entry {item.id!r} does not match schema: {exc}"
                ) from exc
        return results


__all__ = [
    "BatchItem",
    "BatchedTaskInput",
    "BatchedTaskRunner",
]
