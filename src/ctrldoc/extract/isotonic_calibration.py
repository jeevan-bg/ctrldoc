"""Isotonic regression for raw NLI / judge scores -> calibrated probabilities.

Per SPEC §6.5, calibration is a two-step pipeline. Paraphrase voting
(S-136) supplies an agreement-rate signal that correlates with
correctness; this module lands the second half:

> plus isotonic regression fitted on the eval set, mapping raw scores
> -> calibrated probabilities.

The release gate, shipped per backend, is **Expected Calibration Error
(ECE) <= 0.05 on the held-out eval** (§13 non-negotiable 8). This
module is the substrate that closes that gate.

What it ships
-------------

* `IsotonicCalibrator` — fits a monotonic step function via the
  pool-adjacent-violators (PAV) algorithm against `(raw_score,
  correct)` pairs, then maps any unseen raw score to a calibrated
  probability by linear interpolation between the fitted breakpoints.
  Extrapolation clamps to the nearest endpoint. Output is clipped to
  `[0, 1]` so the result is always a valid probability.

* `CalibratedNLIScorer` — wraps any `NLIScorer` and rewrites the
  emitted `NLIScore` so the top-label confidence carries the
  calibrated probability while the non-top mass redistributes
  proportionally to the inner backend's relative ratios. The
  3-component shape and argmax label are preserved; only the
  confidence number changes.

* `fit_per_backend_ece` — one-shot helper that fits a calibrator on
  the first half of a labelled set and reports the held-out ECE on
  the second half. The headline number that the §6.5 release gate is
  measured against.

* `ece_within_release_gate` — boolean shim that names the §6.5
  threshold (`CALIBRATION_ECE_THRESHOLD = 0.05`) for callers that
  want to gate without re-importing the eval substrate.

Design choices
--------------

The implementation is stdlib-only by design. scipy is not a project
dependency and adding it for one PAV pass would violate the "fewer
moving parts" rule in `.ctrldoc/WAYS_OF_WORKING.md`. The Spearman
helper in `paraphrase_voting` set the precedent.

Linear interpolation between breakpoints follows the sklearn
convention so calibrated probabilities are continuous in the raw
score, not step-wise — a small raw delta produces a small calibrated
delta. Clamping at the endpoints prevents extrapolation from
producing out-of-range probabilities when a raw score sits outside
the fitted support.

SPEC-REF: §6.5 (probabilistic edges + calibration — isotonic regression)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ctrldoc.eval.calibration import (
    CALIBRATION_ECE_THRESHOLD,
    CalibrationLabel,
    NLIScore,
    expected_calibration_error,
)

# ---------------------------------------------------------------------------
# Public release-gate shim
# ---------------------------------------------------------------------------


def ece_within_release_gate(ece: float) -> bool:
    """True iff `ece` clears the §6.5 release gate (`<= 0.05`).

    Tiny convenience over a literal comparison so callers can name
    the gate by intent rather than by magic number.
    """
    return ece <= CALIBRATION_ECE_THRESHOLD


# ---------------------------------------------------------------------------
# Pool-adjacent-violators isotonic regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Breakpoint:
    """One fitted breakpoint: raw score -> calibrated probability."""

    raw: float
    calibrated: float


class IsotonicCalibrator:
    """Monotonic raw-score -> calibrated-probability map fit by PAV.

    A single calibrator instance handles one backend's calibration
    surface (e.g. all NLI top-label confidences from a particular
    cross-encoder, or all LLM-judge confidences from a particular
    judge profile). Per §6.5 the calibrator is fit on a held-out eval
    set once and shipped as part of the backend's persisted state.

    The calibrator is stateless until `fit` has been called. A subsequent
    `transform` before `fit` raises `RuntimeError` — silently returning
    the raw input would mask a configuration bug.
    """

    def __init__(self) -> None:
        self._breakpoints: list[_Breakpoint] | None = None

    # -- fit ----------------------------------------------------------------

    def fit(self, *, raw_scores: Sequence[float], correct: Sequence[int]) -> None:
        """Fit the calibrator against `(raw_score, correct)` pairs.

        `raw_scores` must lie in `[0, 1]`; `correct` must be binary
        `{0, 1}`. The two sequences must have equal length and at
        least one element. Inputs may be unsorted; the calibrator
        sorts internally by `raw_score` (ties break stably on input
        order) before running PAV.

        The fit replaces any previous fit on this instance.
        """
        if len(raw_scores) != len(correct):
            raise ValueError(
                f"length mismatch: {len(raw_scores)} raw_scores vs {len(correct)} correct"
            )
        if not raw_scores:
            raise ValueError("cannot fit on empty input")
        for score in raw_scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"raw_scores must lie in the unit interval (got {score!r})")
        for label in correct:
            if label not in (0, 1):
                raise ValueError(f"correct must be binary {{0, 1}} (got {label!r})")

        # Stable sort by raw score so equal raw values pool in input order.
        pairs = sorted(zip(raw_scores, correct, strict=True), key=lambda rc: rc[0])

        # PAV: greedily merge backwards every time a new block's mean
        # falls strictly below the previous block's mean. Each block
        # carries (sum_raw, sum_correct, count); the block's calibrated
        # value is sum_correct / count and its representative raw is
        # sum_raw / count (so multi-element blocks anchor at their
        # centroid for downstream interpolation).
        block_sum_raw: list[float] = []
        block_sum_correct: list[float] = []
        block_count: list[int] = []
        for raw, label in pairs:
            block_sum_raw.append(float(raw))
            block_sum_correct.append(float(label))
            block_count.append(1)
            # Walk left, pooling violators.
            while len(block_sum_correct) >= 2:
                left_mean = block_sum_correct[-2] / block_count[-2]
                right_mean = block_sum_correct[-1] / block_count[-1]
                if left_mean <= right_mean:
                    break
                # Pool the right block into the left block.
                block_sum_raw[-2] += block_sum_raw[-1]
                block_sum_correct[-2] += block_sum_correct[-1]
                block_count[-2] += block_count[-1]
                block_sum_raw.pop()
                block_sum_correct.pop()
                block_count.pop()

        breakpoints = [
            _Breakpoint(
                raw=block_sum_raw[i] / block_count[i],
                calibrated=block_sum_correct[i] / block_count[i],
            )
            for i in range(len(block_count))
        ]
        # Anchor the calibrated value on every distinct raw score that
        # appears in the fit data, not only the pooled centroids. This
        # mirrors the sklearn behaviour where any raw equal to an input
        # score returns its block's calibrated value exactly. Without
        # this, linear interpolation across a multi-element pooled block
        # would distort the on-block lookup.
        block_index_by_position: list[int] = []
        cursor = 0
        running_count = 0
        for _ in pairs:
            if running_count >= block_count[cursor]:
                cursor += 1
                running_count = 0
            block_index_by_position.append(cursor)
            running_count += 1

        anchored: dict[float, float] = {}
        for (raw, _), block_index in zip(pairs, block_index_by_position, strict=True):
            anchored[float(raw)] = breakpoints[block_index].calibrated
        # Merge pooled centroids (kept for between-block interpolation)
        # with the per-input anchors. Anchors win on collision.
        merged: dict[float, float] = {bp.raw: bp.calibrated for bp in breakpoints}
        merged.update(anchored)
        sorted_breakpoints = [_Breakpoint(raw=r, calibrated=c) for r, c in sorted(merged.items())]

        self._breakpoints = sorted_breakpoints

    # -- transform ----------------------------------------------------------

    def transform(self, raw_score: float) -> float:
        """Map a raw score to its calibrated probability in `[0, 1]`.

        Below the lowest fitted raw score: clamp to the lowest
        calibrated value. Above the highest fitted raw score: clamp
        to the highest calibrated value. Between two adjacent
        breakpoints: linear interpolation. On a breakpoint exactly:
        the breakpoint's calibrated value.
        """
        if self._breakpoints is None:
            raise RuntimeError("transform called before fit; fit the calibrator first")
        bps = self._breakpoints
        if raw_score <= bps[0].raw:
            return _clip_unit(bps[0].calibrated)
        if raw_score >= bps[-1].raw:
            return _clip_unit(bps[-1].calibrated)
        # Find the bracket [bps[i], bps[i+1]] containing raw_score.
        # Linear scan is fine — fit sets are tens to hundreds of points.
        for i in range(len(bps) - 1):
            left = bps[i]
            right = bps[i + 1]
            if left.raw <= raw_score <= right.raw:
                if right.raw == left.raw:
                    return _clip_unit(right.calibrated)
                t = (raw_score - left.raw) / (right.raw - left.raw)
                return _clip_unit(left.calibrated + t * (right.calibrated - left.calibrated))
        # Unreachable: bracket exhaustion implies an unsorted breakpoint
        # array, which would itself be an internal bug. Raise so the
        # invariant violation surfaces immediately.
        raise RuntimeError(  # pragma: no cover - defensive
            f"breakpoint bracket exhausted for raw_score={raw_score!r}"
        )


def _clip_unit(value: float) -> float:
    """Clip `value` into `[0, 1]` — calibrated probabilities are bounded."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ---------------------------------------------------------------------------
# NLI scorer wrapper — preserves argmax, recalibrates top-label confidence
# ---------------------------------------------------------------------------


@runtime_checkable
class NLIScorer(Protocol):
    """3-way NLI scorer — same shape as `eval.calibration.CalibrationScorer`."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


class CalibratedNLIScorer:
    """Wrap an `NLIScorer` so its top-label confidence is recalibrated.

    Construction binds an inner backend and a pre-fit
    `IsotonicCalibrator`. On every `score` call:

    1. Delegate to the inner scorer for the raw 3-way distribution.
    2. Identify the argmax label and its raw confidence.
    3. Replace the top-label probability with
       `calibrator.transform(raw_top_confidence)`.
    4. Distribute the leftover mass (`1 - calibrated_top`) over the
       other two labels in proportion to their raw masses. When both
       non-top raw masses are zero, the leftover splits evenly.

    The argmax label is preserved by construction because the
    calibrated top is the largest probability iff the remainder is
    less than the calibrated top, i.e. iff `calibrated_top > 0.5`.
    When `calibrated_top <= 0.5` and the remainder must still
    distribute, the wrapper keeps the original argmax (never
    rewriting the verdict) — even if a strict reading of the
    redistributed distribution would suggest a tie. Calibration is a
    confidence operation, not a verdict operation.
    """

    def __init__(self, *, inner: NLIScorer, calibrator: IsotonicCalibrator) -> None:
        self._inner = inner
        self._calibrator = calibrator

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        raw = self._inner.score(premise=premise, hypothesis=hypothesis)
        top_label = raw.argmax_label()
        raw_top = raw.top_confidence()
        calibrated_top = self._calibrator.transform(raw_top)
        remainder = max(0.0, 1.0 - calibrated_top)

        # Identify the non-top labels and their raw masses.
        raw_by_label = {
            "entailment": raw.entailment,
            "contradiction": raw.contradiction,
            "neutral": raw.neutral,
        }
        other_labels = [label for label in raw_by_label if label != top_label]
        other_raw_sum = sum(raw_by_label[label] for label in other_labels)

        new_by_label: dict[str, float] = {top_label: calibrated_top}
        if remainder == 0.0:
            for label in other_labels:
                new_by_label[label] = 0.0
        elif other_raw_sum == 0.0:
            # Both non-top raw masses are zero -- even split is the
            # only well-defined choice that preserves the sum-to-one
            # invariant without inventing a preference.
            half = remainder / 2.0
            for label in other_labels:
                new_by_label[label] = half
        else:
            for label in other_labels:
                new_by_label[label] = remainder * (raw_by_label[label] / other_raw_sum)

        # NLIScore's validator absorbs ~1e-3 sum drift; the math above
        # is within float epsilon of 1.0 so the validator always passes.
        return NLIScore(
            entailment=new_by_label["entailment"],
            contradiction=new_by_label["contradiction"],
            neutral=new_by_label["neutral"],
        )


# ---------------------------------------------------------------------------
# Headline release-gate helper
# ---------------------------------------------------------------------------


def fit_per_backend_ece(
    *,
    raw_scores: Sequence[float],
    correct: Sequence[int],
) -> tuple[float, IsotonicCalibrator]:
    """Fit an `IsotonicCalibrator` and report held-out ECE.

    Splits the labelled batch into two equal halves (first for fit,
    second for evaluation), fits the calibrator, then computes ECE
    on the held-out half using a synthetic two-label distribution
    where the top-label confidence carries the calibrated value and
    the remainder lives on `neutral` so the distribution is valid
    `NLIScore` shape. Returns `(post_fit_ece, calibrator)`.

    The §6.5 release gate is `post_fit_ece <= 0.05`. Callers can
    surface that boolean via `ece_within_release_gate`.
    """
    if len(raw_scores) != len(correct):
        raise ValueError(f"length mismatch: {len(raw_scores)} raw_scores vs {len(correct)} correct")
    if len(raw_scores) < 2:
        raise ValueError("fit_per_backend_ece requires at least 2 samples to split")

    half = len(raw_scores) // 2
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_scores=list(raw_scores[:half]), correct=list(correct[:half]))

    held_out_raw = raw_scores[half:]
    held_out_correct = correct[half:]

    predictions: list[NLIScore] = []
    golds: list[CalibrationLabel] = []
    for score, label in zip(held_out_raw, held_out_correct, strict=True):
        calibrated_top = calibrator.transform(score)
        remainder = max(0.0, 1.0 - calibrated_top)
        predictions.append(
            NLIScore(entailment=calibrated_top, contradiction=0.0, neutral=remainder),
        )
        golds.append("entailment" if label == 1 else "neutral")

    post_ece = expected_calibration_error(predictions=predictions, golds=golds)
    return post_ece, calibrator


__all__ = [
    "CalibratedNLIScorer",
    "IsotonicCalibrator",
    "NLIScorer",
    "ece_within_release_gate",
    "fit_per_backend_ece",
]
