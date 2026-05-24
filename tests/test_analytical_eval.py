"""analytical_eval — runner + 3-case starter set.

The runner drives `AnalyticalReviewPlaybook` per case and scores
recall on `SeededWeakness` entries. A weakness is matched when the
playbook emits at least one `Finding` with the same lens name and
a case-insensitive substring containing the seeded `claim_pattern`.

SPEC-REF: §8.1 (analytical_eval), §8.2 (analytical_review metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.eval.analytical import (
    WEAKNESS_RECALL_THRESHOLD,
    AnalyticalEvalCase,
    AnalyticalEvalRunner,
    SeededWeakness,
    matches_seeded,
    weakness_recall,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.models import Finding, Span
from ctrldoc.ops.review import (
    AnalyticalReviewPlaybook,
    Lens,
)
from ctrldoc.orch.synthesis import SynthesisRunner

ANALYTICAL_EVAL_PATH = Path(__file__).parent / "eval" / "analytical_eval.jsonl"


def _cases() -> list[AnalyticalEvalCase]:
    return load_jsonl_cases(ANALYTICAL_EVAL_PATH, case_model=AnalyticalEvalCase)


# --- helpers ---


def _span() -> Span:
    return Span(chunk_id="c1", char_start=0, char_end=4, text="text")


def _finding(lens: str, claim: str) -> Finding:
    return Finding(
        ctrldoc=lens,
        location=_span(),
        claim=claim,
        severity="warn",
    )


def _seed(seed_id: str, lens: str, pattern: str) -> SeededWeakness:
    return SeededWeakness(id=seed_id, lens=lens, claim_pattern=pattern)


# --- matches_seeded ---


def test_matches_seeded_same_lens_and_substring_returns_true() -> None:
    finding = _finding("assumptions", "The doc assumes single-node deployment throughout.")
    seed = _seed("w", "assumptions", "single-node deployment")
    assert matches_seeded(finding, seed) is True


def test_matches_seeded_is_case_insensitive() -> None:
    finding = _finding("ambiguity", "QUORUM terminology is undefined.")
    seed = _seed("w", "ambiguity", "quorum")
    assert matches_seeded(finding, seed) is True


def test_matches_seeded_wrong_lens_returns_false() -> None:
    finding = _finding("ambiguity", "single-node deployment is assumed")
    seed = _seed("w", "assumptions", "single-node deployment")
    assert matches_seeded(finding, seed) is False


def test_matches_seeded_pattern_absent_returns_false() -> None:
    finding = _finding("assumptions", "The doc covers many scenarios.")
    seed = _seed("w", "assumptions", "specific-missing-token")
    assert matches_seeded(finding, seed) is False


# --- weakness_recall ---


def test_recall_all_seeded_matched_returns_one() -> None:
    findings = [
        _finding("assumptions", "single-node deployment assumption"),
        _finding("boundary_cases", "partition behaviour"),
    ]
    seeded = [
        _seed("w-1", "assumptions", "single-node deployment"),
        _seed("w-2", "boundary_cases", "partition"),
    ]
    assert weakness_recall(findings, seeded) == pytest.approx(1.0)


def test_recall_partial_match() -> None:
    findings = [_finding("assumptions", "single-node deployment assumption")]
    seeded = [
        _seed("w-1", "assumptions", "single-node deployment"),
        _seed("w-2", "boundary_cases", "partition"),
    ]
    assert weakness_recall(findings, seeded) == pytest.approx(0.5)


def test_recall_no_seeded_returns_zero() -> None:
    assert weakness_recall([], []) == pytest.approx(0.0)


def test_recall_one_finding_can_match_multiple_seeded_entries() -> None:
    """If one finding covers two seeded patterns (different lenses
    on the same content), each still counts as matched independently."""
    findings = [
        _finding("scope_gaps", "rollback strategy is not addressed"),
        _finding("ambiguity", "rollback semantics under partition"),
    ]
    seeded = [
        _seed("w-1", "scope_gaps", "rollback"),
        _seed("w-2", "ambiguity", "rollback"),
    ]
    assert weakness_recall(findings, seeded) == pytest.approx(1.0)


# --- dataset invariants ---


def test_analytical_eval_set_has_three_cases() -> None:
    cases = _cases()
    assert len(cases) == 3


def test_every_case_has_at_least_one_seeded_weakness() -> None:
    for case in _cases():
        assert case.seeded_weaknesses, f"case {case.id!r} has no seeded weaknesses"


def test_case_ids_unique() -> None:
    ids = [case.id for case in _cases()]
    assert len(set(ids)) == len(ids)


def test_seeded_weakness_ids_unique_within_case() -> None:
    for case in _cases():
        ids = [seed.id for seed in case.seeded_weaknesses]
        assert len(set(ids)) == len(ids), f"case {case.id!r} has duplicate seeded ids"


def test_case_schema_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        AnalyticalEvalCase(
            id="x",
            doc_type="d",
            seeded_weaknesses=[],
            extra_field="bad",  # type: ignore[call-arg]
        )


# --- runner: oracle sweeper hits every seeded weakness ---


def _build_playbook(
    *,
    lenses: list[Lens],
    findings_per_lens: dict[str, list[Finding]],
    narrative_response: str,
) -> AnalyticalReviewPlaybook:
    @dataclass
    class _StubGenerator:
        lenses_: list[Lens]

        def generate(self, doc_type: str) -> list[Lens]:
            return list(self.lenses_)

    @dataclass
    class _StubSweeper:
        by_lens: dict[str, list[Finding]]

        def sweep(self, lens: Lens) -> list[Finding]:
            return list(self.by_lens.get(lens.name, []))

    @dataclass
    class _StubClient:
        response: str

        def call(self, *, system: str, user: str) -> str:
            return self.response

    return AnalyticalReviewPlaybook(
        prefix=CacheablePrefix(
            system_prompt="analyst",
            doc_skeleton="# d",
            entity_glossary="- **e** [concept]",
        ),
        lens_generator=_StubGenerator(lenses_=lenses),
        sweeper=_StubSweeper(by_lens=findings_per_lens),
        synthesis_runner=SynthesisRunner(client=_StubClient(response=narrative_response)),
    )


def test_runner_with_oracle_findings_clears_threshold() -> None:
    """Wire a playbook whose stub sweeper emits exactly the seeded findings."""
    case = AnalyticalEvalCase(
        id="r-1",
        doc_type="d",
        seeded_weaknesses=[
            _seed("w-1", "assumptions", "single-node deployment"),
            _seed("w-2", "boundary_cases", "partition"),
        ],
    )
    lenses = [
        Lens(id="lens/assumptions", name="assumptions", description="."),
        Lens(id="lens/boundary_cases", name="boundary_cases", description="."),
    ]
    findings = {
        "assumptions": [_finding("assumptions", "Assumes single-node deployment globally.")],
        "boundary_cases": [_finding("boundary_cases", "Behaviour under partition is undefined.")],
    }
    playbook = _build_playbook(
        lenses=lenses,
        findings_per_lens=findings,
        narrative_response='{"headline": "x", "sections": [], "summary": ""}',
    )
    runner = AnalyticalEvalRunner(playbook=playbook)
    result = runner.run_case(case)
    assert result.metrics["weakness_recall"] == pytest.approx(1.0)
    assert result.passed is True


def test_runner_with_empty_findings_fails_threshold() -> None:
    """No findings ⇒ zero recall ⇒ case fails."""
    case = AnalyticalEvalCase(
        id="r-empty",
        doc_type="d",
        seeded_weaknesses=[_seed("w-1", "assumptions", "x")],
    )
    playbook = _build_playbook(
        lenses=[Lens(id="lens/x", name="assumptions", description=".")],
        findings_per_lens={"assumptions": []},
        narrative_response='{"headline": "", "sections": [], "summary": ""}',
    )
    runner = AnalyticalEvalRunner(playbook=playbook)
    result = runner.run_case(case)
    assert result.metrics["weakness_recall"] == pytest.approx(0.0)
    assert result.passed is False


def test_runner_partial_recall_below_threshold_fails() -> None:
    """4 seeded, 2 matched ⇒ recall = 0.5 < 0.80 threshold."""
    case = AnalyticalEvalCase(
        id="r-partial",
        doc_type="d",
        seeded_weaknesses=[
            _seed("w-1", "assumptions", "alpha"),
            _seed("w-2", "assumptions", "bravo"),
            _seed("w-3", "ambiguity", "charlie"),
            _seed("w-4", "ambiguity", "delta"),
        ],
    )
    playbook = _build_playbook(
        lenses=[
            Lens(id="lens/a", name="assumptions", description="."),
            Lens(id="lens/b", name="ambiguity", description="."),
        ],
        findings_per_lens={
            "assumptions": [_finding("assumptions", "alpha was assumed")],
            "ambiguity": [_finding("ambiguity", "charlie is undefined")],
        },
        narrative_response='{"headline": "", "sections": [], "summary": ""}',
    )
    runner = AnalyticalEvalRunner(playbook=playbook)
    result = runner.run_case(case)
    assert result.metrics["weakness_recall"] == pytest.approx(0.5)
    assert result.passed is False


# --- end-to-end via harness ---


def test_starter_set_passes_with_an_oracle_playbook() -> None:
    """Build a one-finding-per-seeded-weakness oracle playbook from each
    case's data and confirm the harness aggregate matches §8.2."""

    @dataclass
    class _OracleRunner:
        calls: list[str] = field(default_factory=list)

        def run_case(self, case: AnalyticalEvalCase):  # type: ignore[no-untyped-def]
            self.calls.append(case.id)
            lens_names = sorted({seed.lens for seed in case.seeded_weaknesses})
            lenses = [Lens(id=f"lens/{name}", name=name, description=".") for name in lens_names]
            findings_by_lens: dict[str, list[Finding]] = {}
            for seed in case.seeded_weaknesses:
                findings_by_lens.setdefault(seed.lens, []).append(
                    _finding(seed.lens, f"Issue: {seed.claim_pattern} surfaces here.")
                )
            playbook = _build_playbook(
                lenses=lenses,
                findings_per_lens=findings_by_lens,
                narrative_response='{"headline": "", "sections": [], "summary": ""}',
            )
            return AnalyticalEvalRunner(playbook=playbook).run_case(case)

    report = run_eval(
        set_name="analytical_eval",
        cases=_cases(),
        runner=_OracleRunner(),
        thresholds={"weakness_recall": WEAKNESS_RECALL_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["weakness_recall"] == pytest.approx(1.0)
