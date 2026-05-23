"""L5 playbooks — thin orchestrations on top of L0..L4 substrates."""

from __future__ import annotations

from ctrldoc.playbooks.coverage import (
    ChecklistItem,
    CoverageAuditPlaybook,
    CoverageReport,
    CoverageRetriever,
)
from ctrldoc.playbooks.qa import (
    AnswerReport,
    QAPlaybook,
    QARetriever,
)
from ctrldoc.playbooks.quality import (
    CriteriaGenerator,
    HeuristicCriteriaGenerator,
    QualityAuditPlaybook,
    QualityReport,
)
from ctrldoc.playbooks.review import (
    AnalyticalReviewPlaybook,
    HeuristicLensGenerator,
    Lens,
    LensGenerator,
    LensSweeper,
    ReviewNarrative,
    ReviewReport,
)

__all__ = [
    "AnalyticalReviewPlaybook",
    "AnswerReport",
    "ChecklistItem",
    "CoverageAuditPlaybook",
    "CoverageReport",
    "CoverageRetriever",
    "CriteriaGenerator",
    "HeuristicCriteriaGenerator",
    "HeuristicLensGenerator",
    "Lens",
    "LensGenerator",
    "LensSweeper",
    "QAPlaybook",
    "QARetriever",
    "QualityAuditPlaybook",
    "QualityReport",
    "ReviewNarrative",
    "ReviewReport",
]
