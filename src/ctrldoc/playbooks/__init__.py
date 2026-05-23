"""L5 playbooks — thin orchestrations on top of L0..L4 substrates."""

from __future__ import annotations

from ctrldoc.playbooks.qa import (
    AnswerReport,
    QAPlaybook,
    QARetriever,
)

__all__ = [
    "AnswerReport",
    "QAPlaybook",
    "QARetriever",
]
