"""calibration_eval — NLI label accuracy + Expected Calibration Error.

Per SPEC §6.5, every probabilistic edge in the claim graph carries a
calibrated confidence and the v1 release gate is shipped Expected
Calibration Error (ECE) per backend:

> Shipped metric: Expected Calibration Error (ECE) per backend.
> v1 release gate: ECE <= 0.05 on the held-out eval.

This module is the substrate that grades any 3-way NLI scorer on a
labelled premise/hypothesis dataset. The substrate ships three
orthogonal metrics:

1. `label_accuracy` — fraction of cases where the scorer's argmax
   label matches the gold label. The headline correctness number.
2. `expected_calibration_error` — the v1 release gate. Bins
   per-case top-label confidences into K equal-width bins on
   [0, 1] and accumulates the weighted gap between bin accuracy
   and bin mean confidence. A perfectly calibrated scorer has
   ECE = 0; a scorer always 100% confident in a wrong answer has
   ECE = 1.
3. `per_label_recall` — per-class recall on {entailment,
   contradiction, neutral}, surfacing per-class regressions that
   headline accuracy hides.

The runner gates pass/fail on `label_accuracy >= 0.85` AND
`expected_calibration_error <= 0.05` per case. Per-case ECE is
degenerate (one sample = one bin), so the gate is really meaningful
at the aggregate level via `run_eval`'s averaging — but the per-case
metric flows through so the harness can aggregate over the set.

SPEC-REF: §6.5 (probabilistic edges + calibration), §14
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ctrldoc.eval.claim_extraction import DocTypeLiteral
from ctrldoc.eval.harness import EvalResult

CalibrationLabel: TypeAlias = Literal["entailment", "contradiction", "neutral"]

CALIBRATION_LABELS: tuple[CalibrationLabel, ...] = ("entailment", "contradiction", "neutral")

CALIBRATION_ACCURACY_THRESHOLD = 0.85
CALIBRATION_ECE_THRESHOLD = 0.05
DEFAULT_ECE_BINS = 10

# Pydantic v2 frozen models validate component sums with a small
# tolerance to absorb JSON round-tripping of softmax outputs.
_DISTRIBUTION_SUM_TOLERANCE = 1e-3


class NLIScore(BaseModel):
    """3-way softmax distribution from an NLI backend.

    Components must be non-negative and sum to ~1.0 (tolerance
    1e-3, which absorbs JSON serialization drift from a softmax).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entailment: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _components_sum_to_one(self) -> NLIScore:
        total = self.entailment + self.contradiction + self.neutral
        if abs(total - 1.0) > _DISTRIBUTION_SUM_TOLERANCE:
            raise ValueError(
                f"NLIScore components must sum to 1.0 (got {total:.6f}); "
                "ensure the backend emits a normalized softmax distribution"
            )
        return self

    def argmax_label(self) -> CalibrationLabel:
        """Label with maximum probability mass; ties → entailment.

        The tiebreak is deterministic and documented — a uniform
        distribution always argmaxes to entailment so callers can
        reason about behavior on degenerate scorers.
        """
        best_label: CalibrationLabel = "entailment"
        best_value = self.entailment
        if self.contradiction > best_value:
            best_label = "contradiction"
            best_value = self.contradiction
        if self.neutral > best_value:
            best_label = "neutral"
            best_value = self.neutral
        return best_label

    def top_confidence(self) -> float:
        """Probability mass on the argmax label."""
        return max(self.entailment, self.contradiction, self.neutral)


class CalibrationEvalCase(BaseModel):
    """One NLI case: premise + hypothesis + gold label + doc type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    premise: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    gold_label: CalibrationLabel
    doc_type: DocTypeLiteral


def label_accuracy(
    *,
    predictions: list[NLIScore],
    golds: list[CalibrationLabel],
) -> float:
    """Fraction of cases where the argmax label matches gold.

    Empty input returns 0.0 — the harness uses this on per-case
    runs (length 1) so an empty list is a degenerate case the
    caller can interpret directly.
    """
    if len(predictions) != len(golds):
        raise ValueError(f"length mismatch: {len(predictions)} predictions vs {len(golds)} golds")
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, golds, strict=True) if p.argmax_label() == g)
    return correct / len(predictions)


def expected_calibration_error(
    *,
    predictions: list[NLIScore],
    golds: list[CalibrationLabel],
    num_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """ECE on top-label confidence: weighted gap between bin
    accuracy and bin mean confidence, normalised by sample count.

    Defined per Guo et al. 2017:

        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where bins partition [0, 1] into `num_bins` equal-width
    intervals on the top-label confidence. A perfectly calibrated
    classifier has ECE = 0; a classifier always 100% confident in
    a wrong answer has ECE = 1.

    Empty input returns 0.0. Per-case calls (length 1) put the
    sole prediction into a single bin and return either 0 (correct)
    or `top_confidence` (incorrect) — the harness's aggregation
    averages these to recover the population ECE.
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive (got {num_bins})")
    if len(predictions) != len(golds):
        raise ValueError(f"length mismatch: {len(predictions)} predictions vs {len(golds)} golds")
    if not predictions:
        return 0.0

    # Equal-width bins on [0, 1]; bin index = floor(conf * num_bins),
    # capped at num_bins - 1 so confidence == 1.0 falls into the top bin.
    bin_totals = [0] * num_bins
    bin_correct = [0] * num_bins
    bin_confidence_sum = [0.0] * num_bins

    for pred, gold in zip(predictions, golds, strict=True):
        conf = pred.top_confidence()
        idx = min(int(conf * num_bins), num_bins - 1)
        bin_totals[idx] += 1
        bin_confidence_sum[idx] += conf
        if pred.argmax_label() == gold:
            bin_correct[idx] += 1

    n = len(predictions)
    ece = 0.0
    for b in range(num_bins):
        if bin_totals[b] == 0:
            continue
        bin_acc = bin_correct[b] / bin_totals[b]
        bin_conf = bin_confidence_sum[b] / bin_totals[b]
        ece += (bin_totals[b] / n) * abs(bin_acc - bin_conf)
    return ece


def per_label_recall(
    *,
    predictions: list[NLIScore],
    golds: list[CalibrationLabel],
) -> dict[CalibrationLabel, float]:
    """Per-class recall on {entailment, contradiction, neutral}.

    A class absent from `golds` gets recall 0.0 (no positives to
    recall). The return dict always contains all three keys so
    aggregation does not silently drop a class.
    """
    if len(predictions) != len(golds):
        raise ValueError(f"length mismatch: {len(predictions)} predictions vs {len(golds)} golds")

    recalls: dict[CalibrationLabel, float] = dict.fromkeys(CALIBRATION_LABELS, 0.0)
    for label in CALIBRATION_LABELS:
        positives = sum(1 for g in golds if g == label)
        if positives == 0:
            continue
        true_positives = sum(
            1
            for p, g in zip(predictions, golds, strict=True)
            if g == label and p.argmax_label() == label
        )
        recalls[label] = true_positives / positives
    return recalls


@runtime_checkable
class CalibrationScorer(Protocol):
    """3-way NLI scorer under evaluation."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


class CalibrationEvalRunner:
    """Adapt a `CalibrationScorer` into the harness `CaseRunner` shape.

    The runner gates pass/fail on `label_accuracy >= 0.85` AND
    `expected_calibration_error <= 0.05`. Per-case ECE is degenerate
    on one sample (one bin), but the metric flows through so the
    harness's aggregator can recover the population ECE.
    """

    def __init__(
        self,
        *,
        scorer: CalibrationScorer,
        accuracy_threshold: float = CALIBRATION_ACCURACY_THRESHOLD,
        ece_threshold: float = CALIBRATION_ECE_THRESHOLD,
        num_bins: int = DEFAULT_ECE_BINS,
    ) -> None:
        self._scorer = scorer
        self._accuracy_threshold = accuracy_threshold
        self._ece_threshold = ece_threshold
        self._num_bins = num_bins

    def run_case(self, case: CalibrationEvalCase) -> EvalResult:
        score = self._scorer.score(premise=case.premise, hypothesis=case.hypothesis)
        predictions = [score]
        golds: list[CalibrationLabel] = [case.gold_label]

        acc = label_accuracy(predictions=predictions, golds=golds)
        ece = expected_calibration_error(
            predictions=predictions, golds=golds, num_bins=self._num_bins
        )
        top_conf = score.top_confidence()

        passed = acc >= self._accuracy_threshold and ece <= self._ece_threshold
        return EvalResult(
            case_id=case.id,
            passed=passed,
            score=acc,
            metrics={
                "label_accuracy": acc,
                "expected_calibration_error": ece,
                "top_confidence": top_conf,
            },
            notes=(
                f"gold={case.gold_label}, pred={score.argmax_label()}, "
                f"top_conf={top_conf:.3f}, ece={ece:.3f}"
            ),
        )


__all__ = [
    "CALIBRATION_ACCURACY_THRESHOLD",
    "CALIBRATION_ECE_THRESHOLD",
    "CALIBRATION_LABELS",
    "DEFAULT_ECE_BINS",
    "CalibrationEvalCase",
    "CalibrationEvalRunner",
    "CalibrationLabel",
    "CalibrationScorer",
    "NLIScore",
    "expected_calibration_error",
    "label_accuracy",
    "per_label_recall",
]
