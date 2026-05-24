"""quality_eval — runner + 3-doc starter set against expert checklists.

Each case feeds a `doc_type` to a `CriteriaGenerator` and grades the
generated criteria against the expert's gold list. Matching uses
Jaccard token overlap with a configurable threshold so the eval is
deterministic and substrate-free; an LLM-judge variant can replace
it later (S-089) without changing the case schema.

SPEC-REF: §8.1 (quality_eval), §8.2 (quality_audit metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.eval.quality import (
    CRITERIA_COVERAGE_THRESHOLD,
    QualityEvalCase,
    QualityEvalRunner,
    criteria_coverage,
)
from ctrldoc.ops.audit import ChecklistItem
from ctrldoc.ops.quality import (
    CriteriaGenerator,
    HeuristicCriteriaGenerator,
)

QUALITY_EVAL_PATH = Path(__file__).parent / "eval" / "quality_eval.jsonl"


def _cases() -> list[QualityEvalCase]:
    return load_jsonl_cases(QUALITY_EVAL_PATH, case_model=QualityEvalCase)


# --- criteria_coverage helper ---


def test_criteria_coverage_perfect_match_returns_one() -> None:
    gold = ["alpha bravo charlie", "delta echo foxtrot"]
    gen = ["alpha bravo charlie", "delta echo foxtrot"]
    assert criteria_coverage(gold_texts=gold, generated_texts=gen) == pytest.approx(1.0)


def test_criteria_coverage_no_overlap_returns_zero() -> None:
    gold = ["alpha bravo"]
    gen = ["xray yankee zulu"]
    assert criteria_coverage(gold_texts=gold, generated_texts=gen) == pytest.approx(0.0)


def test_criteria_coverage_partial_overlap_counts_each_gold_independently() -> None:
    gold = [
        "shared tokens with first criterion",
        "totally unrelated phrasing here",
    ]
    gen = ["this shares many shared tokens with the first criterion description"]
    # The first gold criterion matches; the second does not.
    coverage = criteria_coverage(gold_texts=gold, generated_texts=gen)
    assert coverage == pytest.approx(0.5)


def test_criteria_coverage_empty_gold_returns_zero() -> None:
    assert criteria_coverage(gold_texts=[], generated_texts=["anything"]) == pytest.approx(0.0)


def test_criteria_coverage_match_threshold_controls_strictness() -> None:
    gold = ["alpha bravo charlie"]
    gen = ["alpha bravo something different"]
    # Jaccard = |{alpha, bravo}| / |{alpha, bravo, charlie, something, different}| = 2/5 = 0.4
    high = criteria_coverage(gold_texts=gold, generated_texts=gen, match_threshold=0.5)
    low = criteria_coverage(gold_texts=gold, generated_texts=gen, match_threshold=0.3)
    assert high == pytest.approx(0.0)
    assert low == pytest.approx(1.0)


# --- dataset invariants ---


def test_quality_eval_set_has_three_cases() -> None:
    """Per §8.1: 3 docs with expert-rated quality reports."""
    assert len(_cases()) == 3


def test_every_case_has_non_empty_gold_criteria() -> None:
    for case in _cases():
        assert case.gold_criteria_texts, f"case {case.id!r} has empty gold criteria"


def test_quality_eval_case_rejects_empty_gold_at_construction() -> None:
    with pytest.raises(ValidationError):
        QualityEvalCase(id="bad", doc_type="x", gold_criteria_texts=[])


def test_every_case_has_a_doc_type() -> None:
    for case in _cases():
        assert case.doc_type.strip(), f"case {case.id!r} has blank doc_type"


def test_case_ids_unique() -> None:
    ids = [case.id for case in _cases()]
    assert len(set(ids)) == len(ids)


# --- runner: end-to-end ---


def test_runner_with_heuristic_generator_matches_starter_gold() -> None:
    """The starter gold criteria are drawn from HeuristicCriteriaGenerator's
    own output (S-072), so the heuristic should clear the §8.2 threshold."""
    runner = QualityEvalRunner(generator=HeuristicCriteriaGenerator())
    report = run_eval(
        set_name="quality_eval",
        cases=_cases(),
        runner=runner,
        thresholds={"criteria_coverage": CRITERIA_COVERAGE_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["criteria_coverage"] == pytest.approx(1.0)


def test_runner_with_empty_generator_fails_threshold() -> None:
    """A generator that returns no criteria emits 0.0 coverage."""

    @dataclass
    class _EmptyGenerator:
        calls: list[str] = field(default_factory=list)

        def generate(self, doc_type: str) -> list[ChecklistItem]:
            self.calls.append(doc_type)
            return []

    runner = QualityEvalRunner(generator=_EmptyGenerator())
    report = run_eval(
        set_name="quality_eval",
        cases=_cases(),
        runner=runner,
        thresholds={"criteria_coverage": CRITERIA_COVERAGE_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["criteria_coverage"] == pytest.approx(0.0)


def test_runner_threshold_falls_between_perfect_and_empty() -> None:
    """A generator that matches only half the gold criteria should fall
    below the 0.85 threshold."""

    @dataclass
    class _HalfMatchGenerator:
        def generate(self, doc_type: str) -> list[ChecklistItem]:
            # Match the first gold criterion text; omit the rest.
            case_by_doc = {case.doc_type: case for case in _cases()}
            case = case_by_doc[doc_type]
            return [
                ChecklistItem(
                    id="g/0",
                    text=case.gold_criteria_texts[0],
                    topic_key="t",
                )
            ]

    runner = QualityEvalRunner(generator=_HalfMatchGenerator())
    report = run_eval(
        set_name="quality_eval",
        cases=_cases(),
        runner=runner,
        thresholds={"criteria_coverage": CRITERIA_COVERAGE_THRESHOLD},
    )
    # Each case has 4 gold criteria; one matches → coverage = 0.25 per case.
    assert report.aggregate["criteria_coverage"] == pytest.approx(0.25)
    assert report.passed is False


# --- protocol conformance ---


def test_heuristic_generator_satisfies_criteria_generator_protocol() -> None:
    assert isinstance(HeuristicCriteriaGenerator(), CriteriaGenerator)
