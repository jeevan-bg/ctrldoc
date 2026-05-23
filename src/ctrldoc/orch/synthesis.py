"""Synthesis primitive — reduce over structured findings.

The last step of every playbook collapses a list of structured
findings into a single synthesised report. Per §3.1 pillar 1, the
synthesis call sees only the cacheable prefix and the findings JSON
— never the raw document — so the model can't drift from the
distilled evidence even when the underlying doc is huge.

The runner is intentionally one-shot: no fan-out, no batching. It
shares the prompt-cache prefix with the upstream map calls in the
same session, so the only per-call cost is the synthesis tail.

SPEC-REF: §3.1 (pillar 1), §4.5 (orchestrator)
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.orch.task import TaskClient, TaskOutputError, _extract_json

T = TypeVar("T", bound=BaseModel)


class SynthesisInput(BaseModel):
    """Inputs for one synthesis (reduce) call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix: CacheablePrefix
    findings: list[dict[str, Any]]
    instruction: str

    @field_validator("instruction")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value


class SynthesisRunner:
    """Reduce structured findings into one validated structured output."""

    def __init__(self, *, client: TaskClient) -> None:
        self._client = client

    def run(self, task: SynthesisInput, *, output_model: type[T]) -> T:
        system = task.prefix.render()
        user = self._build_user_message(task)
        raw = self._client.call(system=system, user=user)
        return self._parse(raw, output_model)

    @staticmethod
    def _build_user_message(task: SynthesisInput) -> str:
        findings_payload = json.dumps(task.findings, ensure_ascii=False, sort_keys=False)
        return f"<findings>\n{findings_payload}\n</findings>\n\n<task>\n{task.instruction}\n</task>"

    @staticmethod
    def _parse(raw: str, output_model: type[T]) -> T:
        payload = _extract_json(raw).strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TaskOutputError(f"synthesis response is not valid JSON: {exc}") from exc
        try:
            return output_model.model_validate(data)
        except ValidationError as exc:
            raise TaskOutputError(f"synthesis response does not match schema: {exc}") from exc


__all__ = [
    "SynthesisInput",
    "SynthesisRunner",
]
