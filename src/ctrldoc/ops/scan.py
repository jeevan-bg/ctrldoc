"""UC5 — anomaly scan playbook.

A `Detector` is a small, deterministic-or-LLM pass over the index
that emits `Finding` records pointing at suspicious chunks or
sections. The playbook runs every configured detector and returns
an `AnomalyQueue` ranked by severity (`critical` > `warn` > `info`)
so a human can triage from the top.

Two deterministic detectors ship here as references:

  - `HedgeWordDetector` — regex over chunk text, flags hedging
    language (`usually`, `may`, `should`, …).
  - `EmptySummaryDetector` — flags sections whose summary is empty
    or whitespace-only.

The remaining four §5.5 detectors (asymmetry, justification gap,
undefined terms, boundary silence, embedding outlier) plug into the
same Protocol; each can land as a follow-up once its eval set
exists (§8.1).

SPEC-REF: §5.5 (UC5 anomaly_scan)
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.models import Finding, SeverityLiteral, Span
from ctrldoc.store import Store


@runtime_checkable
class Detector(Protocol):
    """One pass over the index that emits suspicious-chunk findings."""

    name: str

    def detect(self, *, store: Store) -> list[Finding]: ...


class AnomalyQueue(BaseModel):
    """Ranked queue of anomaly findings (highest severity first)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: list[Finding]


_HEDGE_WORDS = (
    "usually",
    "typically",
    "should",
    "may",
    "might",
    "possibly",
    "sometimes",
)
_HEDGE_RE = re.compile(r"\b(" + "|".join(_HEDGE_WORDS) + r")\b", flags=re.IGNORECASE)


class HedgeWordDetector:
    """Regex over chunk text; one `Finding` per matched hedge word."""

    name = "hedge_word"

    def detect(self, *, store: Store) -> list[Finding]:
        findings: list[Finding] = []
        for chunk in store.iter_chunks():
            for match in _HEDGE_RE.finditer(chunk.text):
                findings.append(
                    Finding(
                        ctrldoc=self.name,
                        location=Span(
                            chunk_id=chunk.id,
                            char_start=match.start(),
                            char_end=match.end(),
                            text=match.group(0),
                        ),
                        claim=f"hedge word {match.group(0)!r} weakens commitment",
                        severity="warn",
                    )
                )
        return findings


class EmptySummaryDetector:
    """Flag sections whose summary field is blank.

    A section without a summary won't contribute to the document
    skeleton (S-025), so it's effectively invisible to the planner
    and retrieval layers. Flagging at `info` severity surfaces the
    gap without crowding higher-severity findings.
    """

    name = "empty_summary"

    def detect(self, *, store: Store) -> list[Finding]:
        findings: list[Finding] = []
        for section in store.iter_sections():
            if section.summary.strip():
                continue
            findings.append(
                Finding(
                    ctrldoc=self.name,
                    location=Span(
                        chunk_id=section.id,
                        char_start=0,
                        char_end=len(section.title),
                        text=section.title,
                    ),
                    claim=f"section {section.id!r} has no summary",
                    severity="info",
                )
            )
        return findings


_SEVERITY_RANK: dict[SeverityLiteral, int] = {"critical": 0, "warn": 1, "info": 2}


class AnomalyScanPlaybook:
    """Run every detector, aggregate findings, rank by severity."""

    def __init__(self, *, detectors: list[Detector]) -> None:
        self._detectors = list(detectors)

    def run(self, *, store: Store) -> AnomalyQueue:
        all_findings: list[Finding] = []
        for detector in self._detectors:
            all_findings.extend(detector.detect(store=store))
        ranked = sorted(all_findings, key=lambda f: _SEVERITY_RANK[f.severity])
        return AnomalyQueue(findings=ranked)


__all__ = [
    "AnomalyQueue",
    "AnomalyScanPlaybook",
    "Detector",
    "EmptySummaryDetector",
    "HedgeWordDetector",
]
