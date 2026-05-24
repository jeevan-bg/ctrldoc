"""anomaly_eval — runner + 3-case starter set scoring triage precision.

Each case carries inline chunks and sections; the runner builds an
`InMemoryStore` from them, runs the configured `AnomalyScanPlaybook`,
and reports `triage_precision` over the resulting queue. Per §8.2
the threshold is `≥0.60`.

SPEC-REF: §8.1 (anomaly_eval), §8.2 (anomaly_scan metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.anomaly import (
    TRIAGE_PRECISION_THRESHOLD,
    AnomalyEvalCase,
    AnomalyEvalRunner,
    ChunkSeed,
    SeededAnomaly,
    is_true_positive,
    triage_precision,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.models import Finding, Span
from ctrldoc.ops.scan import (
    EmptySummaryDetector,
    HedgeWordDetector,
)

ANOMALY_EVAL_PATH = Path(__file__).parent / "eval" / "anomaly_eval.jsonl"


def _cases() -> list[AnomalyEvalCase]:
    return load_jsonl_cases(ANOMALY_EVAL_PATH, case_model=AnomalyEvalCase)


# --- helpers ---


def _span() -> Span:
    return Span(chunk_id="c1", char_start=0, char_end=4, text="text")


def _finding(detector: str, claim: str) -> Finding:
    return Finding(ctrldoc=detector, location=_span(), claim=claim, severity="warn")


def _seed(seed_id: str, detector: str, pattern: str) -> SeededAnomaly:
    return SeededAnomaly(id=seed_id, detector=detector, claim_pattern=pattern)


# --- is_true_positive ---


def test_true_positive_requires_matching_detector_and_substring() -> None:
    finding = _finding("hedge_word", "should retry on transient failure")
    assert is_true_positive(finding, [_seed("a", "hedge_word", "should")]) is True


def test_true_positive_wrong_detector_returns_false() -> None:
    finding = _finding("hedge_word", "should retry")
    assert is_true_positive(finding, [_seed("a", "empty_summary", "should")]) is False


def test_true_positive_case_insensitive_pattern() -> None:
    finding = _finding("hedge_word", "SHOULD always retry")
    assert is_true_positive(finding, [_seed("a", "hedge_word", "should")]) is True


def test_true_positive_no_seeded_returns_false() -> None:
    assert is_true_positive(_finding("hedge_word", "x"), []) is False


# --- triage_precision ---


def test_precision_all_findings_are_true_positives_returns_one() -> None:
    findings = [
        _finding("hedge_word", "should retry"),
        _finding("hedge_word", "may diverge"),
    ]
    seeded = [
        _seed("a-1", "hedge_word", "should"),
        _seed("a-2", "hedge_word", "may"),
    ]
    assert triage_precision(findings, seeded) == pytest.approx(1.0)


def test_precision_mixed_returns_fraction() -> None:
    findings = [
        _finding("hedge_word", "should retry"),  # TP
        _finding("hedge_word", "unrelated"),  # FP
        _finding("empty_summary", "ghost"),  # FP
    ]
    seeded = [_seed("a-1", "hedge_word", "should")]
    assert triage_precision(findings, seeded) == pytest.approx(1 / 3)


def test_precision_empty_queue_returns_zero() -> None:
    """An empty queue is treated as 0.0 — the playbook produced no
    triage work and the spec's precision metric is undefined for empty.
    Zero is the right "didn't produce a useful queue" signal."""
    assert triage_precision([], [_seed("a", "x", "y")]) == pytest.approx(0.0)


# --- dataset invariants ---


def test_anomaly_eval_set_has_three_cases() -> None:
    assert len(_cases()) == 3


def test_every_case_carries_at_least_one_seeded_anomaly() -> None:
    for case in _cases():
        assert case.seeded_anomalies, f"case {case.id!r} has no seeded anomalies"


def test_case_ids_unique() -> None:
    ids = [case.id for case in _cases()]
    assert len(set(ids)) == len(ids)


def test_seeded_anomaly_ids_unique_within_case() -> None:
    for case in _cases():
        ids = [seed.id for seed in case.seeded_anomalies]
        assert len(set(ids)) == len(ids), f"case {case.id!r} has duplicate seeded ids"


def test_case_schema_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        AnomalyEvalCase(
            id="x",
            seeded_anomalies=[],
            extra_field="bad",  # type: ignore[call-arg]
        )


def test_chunk_seed_validates_required_fields() -> None:
    with pytest.raises(ValidationError):
        ChunkSeed(chunk_id="c1", section_id="s")  # type: ignore[call-arg]


# --- runner: hedge-word detector ---


def test_runner_hedge_word_findings_match_seeded_anomalies() -> None:
    case = AnomalyEvalCase(
        id="r-1",
        chunks=[
            ChunkSeed(chunk_id="c1", section_id="s/1", text="Operations should retry."),
            ChunkSeed(chunk_id="c2", section_id="s/1", text="Replicas may diverge."),
        ],
        sections=[],
        seeded_anomalies=[
            _seed("a-1", "hedge_word", "should"),
            _seed("a-2", "hedge_word", "may"),
        ],
    )
    runner = AnomalyEvalRunner(detectors=[HedgeWordDetector()])
    result = runner.run_case(case)
    assert result.metrics["triage_precision"] == pytest.approx(1.0)
    assert result.passed is True


def test_runner_drops_precision_when_real_anomalies_dilute_queue() -> None:
    """All findings are real hedge words, but only one is seeded.
    Precision is 1.0 only if every queue item maps to a seed — others
    count as false positives in this metric."""
    case = AnomalyEvalCase(
        id="r-dilute",
        chunks=[
            ChunkSeed(
                chunk_id="c1", section_id="s/1", text="May fail. Should retry. Typically OK."
            ),
        ],
        sections=[],
        seeded_anomalies=[_seed("a-1", "hedge_word", "should")],
    )
    runner = AnomalyEvalRunner(detectors=[HedgeWordDetector()])
    result = runner.run_case(case)
    # Three hedge words → three findings → only 'should' matches → 1/3.
    assert result.metrics["triage_precision"] == pytest.approx(1 / 3)
    assert result.passed is False


# --- runner: empty-summary detector ---


def test_runner_empty_summary_findings_against_seeded() -> None:
    from ctrldoc.eval.anomaly import SectionSeed

    case = AnomalyEvalCase(
        id="r-summary",
        chunks=[],
        sections=[
            SectionSeed(section_id="sec/full", title="Full", summary="non-empty"),
            SectionSeed(section_id="sec/blank", title="Blank", summary=""),
        ],
        seeded_anomalies=[_seed("a-1", "empty_summary", "sec/blank")],
    )
    runner = AnomalyEvalRunner(detectors=[EmptySummaryDetector()])
    result = runner.run_case(case)
    assert result.metrics["triage_precision"] == pytest.approx(1.0)


# --- end-to-end via harness ---


def test_starter_set_meets_threshold_with_both_detectors() -> None:
    """The starter cases are calibrated so the two reference detectors
    surface only seeded anomalies — precision should clear 0.60."""
    runner = AnomalyEvalRunner(detectors=[HedgeWordDetector(), EmptySummaryDetector()])
    report = run_eval(
        set_name="anomaly_eval",
        cases=_cases(),
        runner=runner,
        thresholds={"triage_precision": TRIAGE_PRECISION_THRESHOLD},
    )
    assert report.passed is True
    assert report.aggregate["triage_precision"] >= TRIAGE_PRECISION_THRESHOLD


def test_starter_set_fails_with_no_detectors() -> None:
    """Empty detector list ⇒ empty queue ⇒ 0.0 precision ⇒ fail."""

    @dataclass
    class _NoOpRunner:
        calls: list[str] = field(default_factory=list)

        def run_case(self, case: AnomalyEvalCase):  # type: ignore[no-untyped-def]
            self.calls.append(case.id)
            return AnomalyEvalRunner(detectors=[]).run_case(case)

    report = run_eval(
        set_name="anomaly_eval",
        cases=_cases(),
        runner=_NoOpRunner(),
        thresholds={"triage_precision": TRIAGE_PRECISION_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["triage_precision"] == pytest.approx(0.0)
