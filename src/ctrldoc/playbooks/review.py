"""UC4 — analytical review playbook.

A `Lens` is one analytical perspective (assumptions, boundary cases,
consistency, ambiguity, scope gaps, …). The playbook enumerates
lenses for the `doc_type`, fans out one sweep per lens to collect
`Finding` records, then synthesises a `ReviewNarrative` over the
aggregated findings via the S-067 reduce primitive — never feeding
the raw document into the synthesis call (§3.1 pillar 1).

The lens enumeration and per-lens sweep are dependency-injected so
a playbook author can swap heuristic stubs for LLM-backed
implementations without changing the composition logic.

SPEC-REF: §5.4
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import Finding
from ctrldoc.orch.synthesis import SynthesisInput, SynthesisRunner


class Lens(BaseModel):
    """One analytical perspective applied to the doc."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    description: str


@runtime_checkable
class LensGenerator(Protocol):
    """Enumerates lenses appropriate for a `doc_type`."""

    def generate(self, doc_type: str) -> list[Lens]: ...


@runtime_checkable
class LensSweeper(Protocol):
    """Sweeps the document through one lens, returning findings."""

    def sweep(self, lens: Lens) -> list[Finding]: ...


class ReviewNarrative(BaseModel):
    """Synthesised narrative emitted by the reduce step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    headline: str
    sections: list[str]
    summary: str


class ReviewReport(BaseModel):
    """Final analytical-review output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_type: str
    findings: list[Finding]
    narrative: ReviewNarrative


_SYNTHESIS_INSTRUCTION = (
    "Write a structured analytical review. Group related findings into "
    "named sections and write a one-paragraph summary covering the most "
    "important issues. Cite findings by their list position; do not "
    "invent new findings."
)


class HeuristicLensGenerator:
    """Returns the canonical §5.4 lens set, deterministically.

    The five baseline lenses (`assumptions`, `boundary_cases`,
    `consistency`, `ambiguity`, `scope_gaps`) match the spec's
    starting list. A production-grade `AnthropicLensGenerator` will
    adapt the lens set to the doc_type via a constrained-JSON Opus
    call; this reference is sufficient for tests and as a fall-back
    when no LLM is configured.
    """

    _NAMES: tuple[tuple[str, str], ...] = (
        ("assumptions", "Identify load-bearing assumptions and check they are stated."),
        ("boundary_cases", "Examine edge conditions, limits, and failure boundaries."),
        ("consistency", "Surface contradictions across sections and definitions."),
        ("ambiguity", "Flag undefined terms and statements with multiple readings."),
        ("scope_gaps", "Note topics the doc claims to cover but doesn't actually address."),
    )

    def generate(self, doc_type: str) -> list[Lens]:
        if not doc_type.strip():
            raise ValueError("doc_type must not be blank")
        return [
            Lens(id=f"lens/{name}", name=name, description=description)
            for name, description in self._NAMES
        ]


class AnalyticalReviewPlaybook:
    """Compose lens enumeration, per-lens sweep, and synthesis."""

    def __init__(
        self,
        *,
        prefix: CacheablePrefix,
        lens_generator: LensGenerator,
        sweeper: LensSweeper,
        synthesis_runner: SynthesisRunner,
    ) -> None:
        self._prefix = prefix
        self._lens_generator = lens_generator
        self._sweeper = sweeper
        self._synthesis_runner = synthesis_runner

    def run(self, doc_type: str) -> ReviewReport:
        if not doc_type.strip():
            raise ValueError("doc_type must not be blank")

        lenses = self._lens_generator.generate(doc_type)
        if not lenses:
            return ReviewReport(
                doc_type=doc_type,
                findings=[],
                narrative=ReviewNarrative(headline="", sections=[], summary=""),
            )

        all_findings: list[Finding] = []
        for lens in lenses:
            all_findings.extend(self._sweeper.sweep(lens))

        synth_input = SynthesisInput(
            prefix=self._prefix,
            findings=[finding.model_dump() for finding in all_findings],
            instruction=_SYNTHESIS_INSTRUCTION,
        )
        narrative = self._synthesis_runner.run(synth_input, output_model=ReviewNarrative)
        return ReviewReport(doc_type=doc_type, findings=all_findings, narrative=narrative)


__all__ = [
    "AnalyticalReviewPlaybook",
    "HeuristicLensGenerator",
    "Lens",
    "LensGenerator",
    "LensSweeper",
    "ReviewNarrative",
    "ReviewReport",
]
