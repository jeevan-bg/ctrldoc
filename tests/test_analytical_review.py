"""UC4 `analytical_review` playbook — lens fan-out + synthesis.

Per §5.4 the playbook:
  1. enumerates analytical lenses for a `doc_type` (assumptions,
     boundary cases, consistency, ambiguity, scope gaps, …);
  2. sweeps the doc through each lens (fan-out), collecting
     `Finding` records;
  3. synthesises a structured `ReviewNarrative` from the aggregated
     findings via the S-067 reduce primitive — the synthesis call
     never sees the raw doc, only the structured findings JSON
     (§3.1 pillar 1).

SPEC-REF: §5.4 (UC4 analytical_review)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import Finding, Span
from ctrldoc.ops.review import (
    AnalyticalReviewPlaybook,
    HeuristicLensGenerator,
    Lens,
    LensGenerator,
    LensSweeper,
    ReviewNarrative,
    ReviewReport,
)
from ctrldoc.orch.synthesis import SynthesisRunner

# --- fixtures ---


def _prefix() -> CacheablePrefix:
    return CacheablePrefix(
        system_prompt="You are an analytical reviewer.",
        doc_skeleton="# §1\n\nbody",
        entity_glossary="- **e/1** [concept]",
    )


def _lens(name: str) -> Lens:
    return Lens(
        id=f"lens-{name}", name=name, description=f"Examine the doc through the {name} lens."
    )


def _finding(lens_name: str, claim: str, chunk_id: str = "c1") -> Finding:
    return Finding(
        ctrldoc=lens_name,
        location=Span(chunk_id=chunk_id, char_start=0, char_end=len(claim), text=claim),
        claim=claim,
        severity="warn",
    )


# --- stubs ---


@dataclass
class _StubLensGenerator:
    lenses: list[Lens]
    calls: list[str] = field(default_factory=list)

    def generate(self, doc_type: str) -> list[Lens]:
        self.calls.append(doc_type)
        return list(self.lenses)


@dataclass
class _StubSweeper:
    """Maps lens id → findings."""

    findings_by_lens: dict[str, list[Finding]]
    calls: list[str] = field(default_factory=list)

    def sweep(self, lens: Lens) -> list[Finding]:
        self.calls.append(lens.id)
        return list(self.findings_by_lens.get(lens.id, []))


@dataclass
class _StubTaskClient:
    response: str
    calls: list[tuple[str, str]] = field(default_factory=list)

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def _narrative_response(*, headline: str, sections: list[str], summary: str) -> str:
    return json.dumps({"headline": headline, "sections": sections, "summary": summary})


def _playbook(
    *,
    lenses: list[Lens],
    findings_by_lens: dict[str, list[Finding]],
    narrative: str,
) -> tuple[AnalyticalReviewPlaybook, _StubLensGenerator, _StubSweeper, _StubTaskClient]:
    generator = _StubLensGenerator(lenses=lenses)
    sweeper = _StubSweeper(findings_by_lens=findings_by_lens)
    task_client = _StubTaskClient(response=narrative)
    playbook = AnalyticalReviewPlaybook(
        prefix=_prefix(),
        lens_generator=generator,
        sweeper=sweeper,
        synthesis_runner=SynthesisRunner(client=task_client),
    )
    return playbook, generator, sweeper, task_client


# --- happy path ---


def test_review_runs_fan_out_then_synthesises() -> None:
    lens_a = _lens("assumptions")
    lens_b = _lens("boundary_cases")
    findings = {
        lens_a.id: [_finding("assumptions", "Assumes single-node deployment.")],
        lens_b.id: [
            _finding("boundary_cases", "What happens on partition?"),
            _finding("boundary_cases", "Reset behaviour under quorum loss is undefined."),
        ],
    }
    playbook, generator, sweeper, task_client = _playbook(
        lenses=[lens_a, lens_b],
        findings_by_lens=findings,
        narrative=_narrative_response(
            headline="Two lenses, three findings",
            sections=["Assumptions", "Boundary cases"],
            summary="Synthesised summary.",
        ),
    )

    report = playbook.run("L0 kernel spec")

    assert isinstance(report, ReviewReport)
    assert report.doc_type == "L0 kernel spec"
    assert len(report.findings) == 3
    assert [f.ctrldoc for f in report.findings] == [
        "assumptions",
        "boundary_cases",
        "boundary_cases",
    ]
    assert report.narrative.headline == "Two lenses, three findings"
    assert report.narrative.sections == ["Assumptions", "Boundary cases"]
    assert report.narrative.summary == "Synthesised summary."

    # Composition:
    assert generator.calls == ["L0 kernel spec"]
    assert sweeper.calls == [lens_a.id, lens_b.id]
    assert len(task_client.calls) == 1


# --- fan-out order ---


def test_findings_preserve_lens_iteration_order() -> None:
    lens_a = _lens("a")
    lens_b = _lens("b")
    lens_c = _lens("c")
    playbook, _, _, _ = _playbook(
        lenses=[lens_a, lens_b, lens_c],
        findings_by_lens={
            lens_a.id: [_finding("a", "from-a-1"), _finding("a", "from-a-2")],
            lens_b.id: [_finding("b", "from-b")],
            lens_c.id: [_finding("c", "from-c-1"), _finding("c", "from-c-2")],
        },
        narrative=_narrative_response(headline="x", sections=[], summary=""),
    )
    report = playbook.run("any")
    claims = [f.claim for f in report.findings]
    assert claims == ["from-a-1", "from-a-2", "from-b", "from-c-1", "from-c-2"]


# --- synthesis prompt layout ---


def test_synthesis_call_sees_findings_json_not_raw_doc() -> None:
    """The synthesis user message carries the findings JSON; nothing
    in the prompt should resemble the raw document body."""
    lens = _lens("assumptions")
    playbook, _, _, task_client = _playbook(
        lenses=[lens],
        findings_by_lens={
            lens.id: [_finding("assumptions", "RAW-CLAIM-MARKER")],
        },
        narrative=_narrative_response(headline="x", sections=[], summary=""),
    )
    playbook.run("any")
    _, user = task_client.calls[0]
    assert "RAW-CLAIM-MARKER" in user
    # The doc skeleton sits inside the system message (cacheable prefix),
    # so the synthesis tail should not also repeat it.
    assert "doc_skeleton" not in user
    # No retrieval plan leakage either.
    assert "search(" not in user


def test_synthesis_system_message_is_the_rendered_prefix() -> None:
    lens = _lens("assumptions")
    prefix = _prefix()
    playbook, _, _, task_client = _playbook(
        lenses=[lens],
        findings_by_lens={lens.id: []},
        narrative=_narrative_response(headline="x", sections=[], summary=""),
    )
    playbook.run("any")
    system, _ = task_client.calls[0]
    assert system == prefix.render()


# --- empty cases ---


def test_no_lenses_short_circuits_without_sweeping_or_synthesising() -> None:
    """If the generator returns no lenses, there's nothing to review —
    no sweep calls, no synthesis call, an empty report."""
    playbook, _, sweeper, task_client = _playbook(
        lenses=[],
        findings_by_lens={},
        narrative="",
    )
    report = playbook.run("any")
    assert report.findings == []
    assert sweeper.calls == []
    assert task_client.calls == []


def test_lenses_but_no_findings_still_synthesises_over_empty_list() -> None:
    """A lens producing zero findings is a legitimate review signal —
    the synthesis call still fires so the model can comment on coverage."""
    lens = _lens("assumptions")
    playbook, _, sweeper, task_client = _playbook(
        lenses=[lens],
        findings_by_lens={lens.id: []},
        narrative=_narrative_response(headline="no findings", sections=[], summary=""),
    )
    report = playbook.run("any")
    assert report.findings == []
    assert sweeper.calls == [lens.id]
    assert len(task_client.calls) == 1
    assert report.narrative.headline == "no findings"


# --- input validation ---


def test_blank_doc_type_rejected() -> None:
    playbook, _, _, _ = _playbook(lenses=[], findings_by_lens={}, narrative="")
    with pytest.raises(ValueError, match="doc_type"):
        playbook.run("   ")


def test_review_report_is_frozen() -> None:
    narrative = ReviewNarrative(headline="x", sections=[], summary="")
    report = ReviewReport(doc_type="x", findings=[], narrative=narrative)
    with pytest.raises(ValidationError):
        report.doc_type = "y"  # type: ignore[misc]


def test_lens_is_frozen() -> None:
    lens = _lens("a")
    with pytest.raises(ValidationError):
        lens.name = "b"  # type: ignore[misc]


# --- protocols ---


def test_lens_generator_protocol_isinstance() -> None:
    stub = _StubLensGenerator(lenses=[])
    assert isinstance(stub, LensGenerator)


def test_lens_sweeper_protocol_isinstance() -> None:
    stub = _StubSweeper(findings_by_lens={})
    assert isinstance(stub, LensSweeper)


# --- heuristic lens generator ---


def test_heuristic_lens_generator_returns_canonical_lens_set() -> None:
    """The §5.4 baseline lens set lands as a deterministic list."""
    gen = HeuristicLensGenerator()
    lenses = gen.generate("L0 kernel spec")
    names = [lens.name for lens in lenses]
    # All canonical names from §5.4 are present.
    assert "assumptions" in names
    assert "boundary_cases" in names
    assert "consistency" in names
    assert "ambiguity" in names
    assert "scope_gaps" in names
    # Ids are unique within one generation.
    ids = [lens.id for lens in lenses]
    assert len(ids) == len(set(ids))


def test_heuristic_lens_generator_is_deterministic() -> None:
    gen = HeuristicLensGenerator()
    a = gen.generate("L0 kernel spec")
    b = gen.generate("L0 kernel spec")
    assert a == b


def test_heuristic_lens_generator_rejects_blank_doc_type() -> None:
    with pytest.raises(ValueError, match="doc_type"):
        HeuristicLensGenerator().generate("")
