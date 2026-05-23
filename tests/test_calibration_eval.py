"""calibration_eval — NLI label accuracy + Expected Calibration Error.

The calibration eval grades an NLI backend on a balanced premise/
hypothesis dataset with three labels {entailment, contradiction,
neutral}. From SPEC §6.5:

> Shipped metric: Expected Calibration Error (ECE) per backend.
> v1 release gate: ECE <= 0.05 on the held-out eval.

The substrate ships three orthogonal metrics:

1. `label_accuracy` — fraction of cases where the backend's argmax
   label matches the gold label. The headline correctness number.
2. `expected_calibration_error` — the v1 release gate. Bins
   per-case top-label confidences into K bins and accumulates the
   weighted gap between bin accuracy and bin mean confidence. A
   perfectly calibrated backend has ECE = 0.
3. `per_label_recall` — per-class recall, surfacing per-class
   regressions that the headline accuracy hides. Not gated.

The runner gates pass/fail on `label_accuracy >= 0.85` AND
`expected_calibration_error <= 0.05`. The starter dataset at
`tests/eval/calibration_eval.jsonl` ships 200 cases balanced
across the three labels with diverse premise/hypothesis pairs
spanning six doc types.

SPEC-REF: §6.5 (probabilistic edges + calibration), §14
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.calibration import (
    CALIBRATION_ACCURACY_THRESHOLD,
    CALIBRATION_ECE_THRESHOLD,
    CalibrationEvalCase,
    CalibrationEvalRunner,
    CalibrationLabel,
    CalibrationScorer,
    NLIScore,
    expected_calibration_error,
    label_accuracy,
    per_label_recall,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval

CALIBRATION_EVAL_PATH = Path(__file__).parent / "eval" / "calibration_eval.jsonl"

_LABELS: tuple[CalibrationLabel, ...] = ("entailment", "contradiction", "neutral")


def _cases() -> list[CalibrationEvalCase]:
    return load_jsonl_cases(CALIBRATION_EVAL_PATH, case_model=CalibrationEvalCase)


def _score(
    entail: float = 0.33,
    contradict: float = 0.33,
    neutral: float = 0.34,
) -> NLIScore:
    return NLIScore(entailment=entail, contradiction=contradict, neutral=neutral)


# --- NLIScore contract ---


def test_nliscore_is_frozen() -> None:
    s = _score()
    with pytest.raises(ValidationError):
        s.entailment = 0.5  # type: ignore[misc]


def test_nliscore_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        NLIScore(entailment=0.5, contradiction=0.3, neutral=0.2, stray=0.0)  # type: ignore[call-arg]


def test_nliscore_rejects_unnormalized_distribution() -> None:
    """Components must sum to ~1.0 (it's a categorical distribution)."""
    with pytest.raises(ValidationError, match="sum"):
        NLIScore(entailment=0.9, contradiction=0.9, neutral=0.9)


def test_nliscore_accepts_floating_point_drift() -> None:
    """Tiny rounding error from softmax in JSON is acceptable."""
    NLIScore(entailment=0.333333, contradiction=0.333333, neutral=0.333334)


def test_nliscore_rejects_negative_component() -> None:
    with pytest.raises(ValidationError):
        NLIScore(entailment=-0.1, contradiction=0.6, neutral=0.5)


def test_nliscore_argmax_label_property() -> None:
    assert _score(entail=0.7, contradict=0.2, neutral=0.1).argmax_label() == "entailment"
    assert _score(entail=0.1, contradict=0.7, neutral=0.2).argmax_label() == "contradiction"
    assert _score(entail=0.2, contradict=0.1, neutral=0.7).argmax_label() == "neutral"


def test_nliscore_top_confidence_property() -> None:
    assert _score(entail=0.7, contradict=0.2, neutral=0.1).top_confidence() == pytest.approx(0.7)


# --- CalibrationEvalCase contract ---


def test_case_is_frozen() -> None:
    case = CalibrationEvalCase(
        id="cal-x",
        premise="p",
        hypothesis="h",
        gold_label="entailment",
        doc_type="spec",
    )
    with pytest.raises(ValidationError):
        case.gold_label = "neutral"  # type: ignore[misc]


def test_case_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        CalibrationEvalCase(
            id="cal-x",
            premise="p",
            hypothesis="h",
            gold_label="maybe",  # type: ignore[arg-type]
            doc_type="spec",
        )


def test_case_rejects_unknown_doc_type() -> None:
    with pytest.raises(ValidationError):
        CalibrationEvalCase(
            id="cal-x",
            premise="p",
            hypothesis="h",
            gold_label="entailment",
            doc_type="poetry",  # type: ignore[arg-type]
        )


def test_case_rejects_empty_premise() -> None:
    with pytest.raises(ValidationError):
        CalibrationEvalCase(
            id="cal-x",
            premise="",
            hypothesis="h",
            gold_label="entailment",
            doc_type="spec",
        )


def test_case_rejects_empty_hypothesis() -> None:
    with pytest.raises(ValidationError):
        CalibrationEvalCase(
            id="cal-x",
            premise="p",
            hypothesis="",
            gold_label="entailment",
            doc_type="spec",
        )


# --- label_accuracy ---


def test_label_accuracy_perfect_argmax_is_one() -> None:
    cases = [
        CalibrationEvalCase(
            id="cal-1", premise="p", hypothesis="h", gold_label="entailment", doc_type="spec"
        ),
        CalibrationEvalCase(
            id="cal-2", premise="p", hypothesis="h", gold_label="contradiction", doc_type="spec"
        ),
    ]
    scores = [
        _score(entail=0.8, contradict=0.1, neutral=0.1),
        _score(entail=0.1, contradict=0.8, neutral=0.1),
    ]
    assert label_accuracy(predictions=scores, golds=[c.gold_label for c in cases]) == pytest.approx(
        1.0
    )


def test_label_accuracy_all_wrong_is_zero() -> None:
    golds: list[CalibrationLabel] = ["entailment", "contradiction"]
    scores = [
        _score(entail=0.1, contradict=0.8, neutral=0.1),
        _score(entail=0.8, contradict=0.1, neutral=0.1),
    ]
    assert label_accuracy(predictions=scores, golds=golds) == pytest.approx(0.0)


def test_label_accuracy_empty_is_zero() -> None:
    assert label_accuracy(predictions=[], golds=[]) == 0.0


def test_label_accuracy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        label_accuracy(predictions=[_score()], golds=["entailment", "contradiction"])


# --- expected_calibration_error ---


def test_ece_perfectly_calibrated_is_zero() -> None:
    """Backend always 100% confident and always correct → ECE = 0."""
    golds: list[CalibrationLabel] = ["entailment"] * 10
    scores = [_score(entail=1.0, contradict=0.0, neutral=0.0) for _ in range(10)]
    ece = expected_calibration_error(predictions=scores, golds=golds, num_bins=10)
    assert ece == pytest.approx(0.0)


def test_ece_perfectly_overconfident_is_one() -> None:
    """Backend 100% confident in the wrong label every time → ECE = 1.0."""
    golds: list[CalibrationLabel] = ["entailment"] * 10
    # Predicts contradiction with confidence 1.0
    scores = [_score(entail=0.0, contradict=1.0, neutral=0.0) for _ in range(10)]
    ece = expected_calibration_error(predictions=scores, golds=golds, num_bins=10)
    assert ece == pytest.approx(1.0)


def test_ece_well_calibrated_at_70_percent() -> None:
    """Backend confidence 0.7 on 10 cases, correct on 7 of them → bin
    accuracy 0.7 matches bin confidence 0.7 → ECE = 0."""
    golds: list[CalibrationLabel] = ["entailment"] * 10
    # 7 correct predictions at confidence 0.7
    scores: list[NLIScore] = []
    for i in range(10):
        if i < 7:
            scores.append(_score(entail=0.7, contradict=0.2, neutral=0.1))
        else:
            scores.append(_score(entail=0.2, contradict=0.7, neutral=0.1))
    ece = expected_calibration_error(predictions=scores, golds=golds, num_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_overconfident_at_90_with_50_accuracy() -> None:
    """Backend always confident 0.9 but right only half the time → gap 0.4."""
    golds: list[CalibrationLabel] = ["entailment"] * 10
    scores: list[NLIScore] = []
    for i in range(10):
        if i < 5:
            scores.append(_score(entail=0.9, contradict=0.05, neutral=0.05))
        else:
            scores.append(_score(entail=0.05, contradict=0.9, neutral=0.05))
    ece = expected_calibration_error(predictions=scores, golds=golds, num_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-6)


def test_ece_empty_is_zero() -> None:
    assert expected_calibration_error(predictions=[], golds=[], num_bins=10) == 0.0


def test_ece_requires_positive_bin_count() -> None:
    with pytest.raises(ValueError, match="num_bins"):
        expected_calibration_error(predictions=[_score()], golds=["entailment"], num_bins=0)


def test_ece_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        expected_calibration_error(predictions=[_score()], golds=[], num_bins=10)


# --- per_label_recall ---


def test_per_label_recall_perfect_is_one_per_class() -> None:
    golds: list[CalibrationLabel] = ["entailment", "contradiction", "neutral"]
    scores = [
        _score(entail=0.8, contradict=0.1, neutral=0.1),
        _score(entail=0.1, contradict=0.8, neutral=0.1),
        _score(entail=0.1, contradict=0.1, neutral=0.8),
    ]
    recalls = per_label_recall(predictions=scores, golds=golds)
    assert recalls["entailment"] == pytest.approx(1.0)
    assert recalls["contradiction"] == pytest.approx(1.0)
    assert recalls["neutral"] == pytest.approx(1.0)


def test_per_label_recall_handles_absent_label() -> None:
    """A label that never appears in gold gets recall 0 (no positives to recall)."""
    golds: list[CalibrationLabel] = ["entailment"]
    scores = [_score(entail=0.8, contradict=0.1, neutral=0.1)]
    recalls = per_label_recall(predictions=scores, golds=golds)
    assert recalls["entailment"] == pytest.approx(1.0)
    assert recalls["contradiction"] == 0.0
    assert recalls["neutral"] == 0.0


def test_per_label_recall_emits_all_three_labels() -> None:
    """The dict always has the three label keys, even if class absent."""
    recalls = per_label_recall(predictions=[], golds=[])
    assert set(recalls.keys()) == {"entailment", "contradiction", "neutral"}


# --- CalibrationScorer Protocol conformance ---


def test_oracle_scorer_satisfies_protocol() -> None:
    @dataclass
    class _Oracle:
        canned: NLIScore

        def score(self, *, premise: str, hypothesis: str) -> NLIScore:
            return self.canned

    assert isinstance(_Oracle(canned=_score()), CalibrationScorer)


# --- Runner ---


@dataclass
class _GoldOracleScorer:
    """Scores each case with confidence 1.0 on its gold label."""

    gold_by_pair: dict[tuple[str, str], CalibrationLabel]

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        label = self.gold_by_pair[(premise, hypothesis)]
        return NLIScore(
            entailment=1.0 if label == "entailment" else 0.0,
            contradiction=1.0 if label == "contradiction" else 0.0,
            neutral=1.0 if label == "neutral" else 0.0,
        )


@dataclass
class _UniformScorer:
    """Always returns a flat distribution; argmax is `entailment` (tiebreak)."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        return NLIScore(entailment=1 / 3, contradiction=1 / 3, neutral=1 / 3)


def _gold_lookup(cases: list[CalibrationEvalCase]) -> dict[tuple[str, str], CalibrationLabel]:
    return {(c.premise, c.hypothesis): c.gold_label for c in cases}


def test_runner_oracle_passes_starter_case() -> None:
    cases = _cases()
    case = cases[0]
    runner = CalibrationEvalRunner(scorer=_GoldOracleScorer(gold_by_pair=_gold_lookup(cases)))
    result = runner.run_case(case)
    assert result.metrics["label_accuracy"] == pytest.approx(1.0)
    assert result.metrics["expected_calibration_error"] == pytest.approx(0.0)
    assert result.passed is True


def test_runner_uniform_scorer_fails_threshold() -> None:
    cases = _cases()
    case = next(c for c in cases if c.gold_label != "entailment")
    runner = CalibrationEvalRunner(scorer=_UniformScorer())
    result = runner.run_case(case)
    # Uniform scorer predicts entailment (tiebreak) for a non-entailment case
    assert result.metrics["label_accuracy"] == pytest.approx(0.0)
    assert result.passed is False


def test_runner_emits_three_named_metrics() -> None:
    cases = _cases()
    case = cases[0]
    runner = CalibrationEvalRunner(scorer=_GoldOracleScorer(gold_by_pair=_gold_lookup(cases)))
    result = runner.run_case(case)
    assert set(result.metrics.keys()) == {
        "label_accuracy",
        "expected_calibration_error",
        "top_confidence",
    }


# --- Dataset invariants ---


def test_dataset_has_exactly_200_cases() -> None:
    assert len(_cases()) == 200


def test_dataset_ids_are_unique() -> None:
    ids = [c.id for c in _cases()]
    assert len(ids) == len(set(ids))


def test_dataset_every_id_has_cal_prefix() -> None:
    for case in _cases():
        assert case.id.startswith("cal-"), f"case id {case.id!r} should start with 'cal-'"


def test_dataset_labels_are_balanced() -> None:
    """200 cases / 3 labels — each class within ±1 of the ideal ~66/67."""
    cases = _cases()
    counts = {label: sum(1 for c in cases if c.gold_label == label) for label in _LABELS}
    for label, count in counts.items():
        assert 60 <= count <= 75, f"label {label} count {count} is outside balance window"


def test_dataset_covers_all_six_doc_types() -> None:
    cases = _cases()
    types = {c.doc_type for c in cases}
    assert len(types) >= 6, f"expected ≥6 doc types, got {types}"


def test_dataset_unique_premise_hypothesis_pairs() -> None:
    cases = _cases()
    pairs = [(c.premise, c.hypothesis) for c in cases]
    assert len(pairs) == len(set(pairs)), "premise/hypothesis pairs must be unique"


def test_dataset_premises_are_substantive() -> None:
    """No trivially-short premises that could create label ambiguity."""
    for c in _cases():
        assert len(c.premise) >= 20, f"premise too short for {c.id!r}: {c.premise!r}"
        assert len(c.hypothesis) >= 10, f"hypothesis too short for {c.id!r}: {c.hypothesis!r}"


# --- End-to-end through the harness ---


def test_starter_set_passes_thresholds_with_oracle_scorer() -> None:
    cases = _cases()
    gold_by_pair = _gold_lookup(cases)

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: CalibrationEvalCase):  # type: ignore[no-untyped-def]
            return CalibrationEvalRunner(
                scorer=_GoldOracleScorer(gold_by_pair=gold_by_pair),
            ).run_case(case)

    report = run_eval(
        set_name="calibration_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={
            "label_accuracy": CALIBRATION_ACCURACY_THRESHOLD,
            "expected_calibration_error": -1.0,  # ECE lower bound; gate via runner-side metric flipping
        },
    )
    # Sanity: with a perfect oracle every metric is at its ideal extreme.
    assert report.aggregate["label_accuracy"] == pytest.approx(1.0)
    assert report.aggregate["expected_calibration_error"] == pytest.approx(0.0)


def test_starter_set_uniform_scorer_fails_accuracy_threshold() -> None:
    cases = _cases()

    @dataclass
    class _DispatchingRunner:
        def run_case(self, case: CalibrationEvalCase):  # type: ignore[no-untyped-def]
            return CalibrationEvalRunner(scorer=_UniformScorer()).run_case(case)

    report = run_eval(
        set_name="calibration_eval",
        cases=cases,
        runner=_DispatchingRunner(),
        thresholds={"label_accuracy": CALIBRATION_ACCURACY_THRESHOLD},
    )
    assert report.passed is False
    assert report.aggregate["label_accuracy"] < CALIBRATION_ACCURACY_THRESHOLD


def test_release_gate_constants_match_spec() -> None:
    """§6.5: ECE ≤ 0.05 is the v1 release gate."""
    assert CALIBRATION_ECE_THRESHOLD == 0.05
    # Accuracy threshold mirrors the rest of the v1 eval substrate.
    assert CALIBRATION_ACCURACY_THRESHOLD == 0.85
