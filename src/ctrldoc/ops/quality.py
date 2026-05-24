"""UC3 — quality audit playbook.

Per §5.3, quality audit is coverage audit plus a generated checklist.
An LLM (or heuristic) enumerates criteria for a `doc_type`, the user
optionally reviews them via the §4.7 HITL checkpoint, then the
playbook delegates to `CoverageAuditPlaybook` to verify each
criterion against the target doc.

The slice ships the `CriteriaGenerator` Protocol, a deterministic
heuristic reference (sufficient for tests and as a fall-back when no
LLM is configured), and the composing `QualityAuditPlaybook`.
A constrained-JSON Anthropic backend can land as a follow-up once
its eval set exists.

SPEC-REF: §5.3
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.ops.audit import (
    ChecklistItem,
    CoverageAuditPlaybook,
    CoverageReport,
)


@runtime_checkable
class CriteriaGenerator(Protocol):
    """Anything that turns a `doc_type` label into a list of criteria."""

    def generate(self, doc_type: str) -> list[ChecklistItem]: ...


class QualityReport(BaseModel):
    """Result of one quality-audit run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_type: str
    criteria: list[ChecklistItem]
    coverage: CoverageReport


class HeuristicCriteriaGenerator:
    """Deterministic reference generator.

    Emits a small list of generic-but-useful criteria the audit can
    actually run on. The output is keyed off `doc_type` so different
    inputs yield structurally distinct checklists (different
    `topic_key`s) — that mirrors how a real LLM-backed generator would
    behave and lets the coverage step exercise its multi-cluster path.

    Not a stand-in for a real generator in production: the actual
    `AnthropicCriteriaGenerator` will use a constrained-JSON Opus
    call as documented in §5.3.
    """

    _BASE_CRITERIA: tuple[tuple[str, str, str], ...] = (
        ("clarity", "The document explains its scope and audience.", "scope"),
        ("completeness", "Every component documented has an interface contract.", "interfaces"),
        ("safety", "Failure modes and recovery paths are addressed.", "failure"),
        ("examples", "Concrete examples illustrate non-trivial behaviour.", "examples"),
    )

    def generate(self, doc_type: str) -> list[ChecklistItem]:
        if not doc_type.strip():
            raise ValueError("doc_type must not be blank")
        slug = _slugify(doc_type)
        items: list[ChecklistItem] = []
        for short_id, text, axis in self._BASE_CRITERIA:
            items.append(
                ChecklistItem(
                    id=f"{slug}/{short_id}",
                    text=text,
                    topic_key=f"{slug}/{axis}",
                )
            )
        return items


class QualityAuditPlaybook:
    """Generate criteria, then delegate to the coverage audit."""

    def __init__(
        self,
        *,
        criteria_generator: CriteriaGenerator,
        coverage_audit: CoverageAuditPlaybook,
    ) -> None:
        self._generator = criteria_generator
        self._coverage = coverage_audit

    def run(self, doc_type: str) -> QualityReport:
        if not doc_type.strip():
            raise ValueError("doc_type must not be blank")
        criteria = self._generator.generate(doc_type)
        coverage = self._coverage.run(criteria)
        return QualityReport(doc_type=doc_type, criteria=criteria, coverage=coverage)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


__all__ = [
    "CriteriaGenerator",
    "HeuristicCriteriaGenerator",
    "QualityAuditPlaybook",
    "QualityReport",
]
