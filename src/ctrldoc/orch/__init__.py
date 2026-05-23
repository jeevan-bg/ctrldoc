"""Orchestrator (L4) — stateless sub-task primitive.

Every sub-task in a playbook is an independent API call with a fresh
context: the cacheable prefix plus the task-specific evidence and
input. The orchestrator never carries reasoning across sub-tasks.

SPEC-REF: §3.1 (pillar 1), §4.5
"""

from __future__ import annotations

from ctrldoc.orch.task import (
    StatelessTaskRunner,
    TaskClient,
    TaskInput,
    TaskOutputError,
)

__all__ = [
    "StatelessTaskRunner",
    "TaskClient",
    "TaskInput",
    "TaskOutputError",
]
