"""Stateless task primitive.

`StatelessTaskRunner.run(task, output_model)` is the only orchestrator
seam that touches an LLM. One call → one fresh API hit; the runner
itself carries no per-call state. The system message is the rendered
cacheable prefix (so the Anthropic prompt cache keys on the same
byte sequence across N parallel sub-tasks) and the user message is
the evidence pack plus the task-specific tail.

Outputs are structured: callers pass a Pydantic model and receive
a validated instance. Malformed JSON and schema mismatches both raise
`TaskOutputError` with the underlying problem in the message.

SPEC-REF: §3.1 (pillar 1 — stateless tasks), §4.5 (orchestrator)
"""

from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from ctrldoc.assembler import CacheablePrefix

T = TypeVar("T", bound=BaseModel)


class TaskOutputError(ValueError):
    """Raised when the LLM response can't be parsed/validated."""


class TaskInput(BaseModel):
    """Inputs for a single sub-task invocation.

    Everything the LLM sees lives on this record. The runner does not
    fold in any additional context, run-id, or counter — keeping the
    serialised prompt byte-stable across identical inputs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix: CacheablePrefix
    evidence_pack: str
    task_input: str


@runtime_checkable
class TaskClient(Protocol):
    """LLM-agnostic transport. Returns the raw text the model emitted."""

    def call(self, *, system: str, user: str) -> str: ...


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Strip a single ```...``` fence if present; otherwise return raw."""
    match = _FENCED_JSON_RE.search(raw)
    return match.group(1) if match else raw


class StatelessTaskRunner:
    """Holds a transport client; carries no per-call state.

    The runner is safe to reuse across many `.run()` invocations and
    across threads — it stores nothing between calls. Any shared state
    (HTTP pools, rate-limit tokens) lives on the injected `client`.
    """

    def __init__(self, *, client: TaskClient) -> None:
        self._client = client

    def run(self, task: TaskInput, *, output_model: type[T]) -> T:
        system = task.prefix.render()
        user = self._build_user_message(task)
        raw = self._client.call(system=system, user=user)
        return self._parse(raw, output_model)

    @staticmethod
    def _build_user_message(task: TaskInput) -> str:
        return (
            f"<evidence>\n{task.evidence_pack}\n</evidence>\n\n<task>\n{task.task_input}\n</task>"
        )

    @staticmethod
    def _parse(raw: str, output_model: type[T]) -> T:
        payload = _extract_json(raw).strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TaskOutputError(f"response is not valid JSON: {exc}") from exc
        try:
            return output_model.model_validate(data)
        except ValidationError as exc:
            raise TaskOutputError(f"response does not match schema: {exc}") from exc


__all__ = [
    "StatelessTaskRunner",
    "TaskClient",
    "TaskInput",
    "TaskOutputError",
]
