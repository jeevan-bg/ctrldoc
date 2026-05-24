"""Isotonic calibration — raw scores -> calibrated probabilities; ECE <= 0.05.

Per SPEC §6.5, the calibration pipeline is two-step: paraphrase voting
(landed in S-136) followed by **isotonic regression fitted on the eval
set, mapping raw scores -> calibrated probabilities**. The release gate
is shipped Expected Calibration Error (ECE) per backend:

> Shipped metric: Expected Calibration Error (ECE) per backend.
> v1 release gate: ECE <= 0.05 on the held-out eval.

`IsotonicCalibrator` fits a monotonic step function via the
pool-adjacent-violators (PAV) algorithm against `(raw_score,
correct_label)` pairs, then maps any new raw score to a calibrated
probability by linear interpolation between the fitted breakpoints.
The wrapper `CalibratedNLIScorer` adapts an `NLIScorer` so its
emitted `NLIScore` carries the calibrated top-label probability with
the remaining mass distributed proportionally over the other two
labels — the 3-class shape is preserved, the top-label argmax is
preserved, only its confidence is recalibrated.

The contract is empirical: starting from a deliberately
miscalibrated synthetic backend (raw top-confidences inflated by a
fixed offset so the headline ECE clears 0.05), fitting an
`IsotonicCalibrator` on a held-out batch must compress the ECE on a
disjoint test batch to <= 0.05. That is the §6.5 release gate.

SPEC-REF: §6.5 (probabilistic edges + calibration — isotonic regression)
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.calibration import (
    CALIBRATION_ECE_THRESHOLD,
    CalibrationLabel,
    NLIScore,
    expected_calibration_error,
)
from ctrldoc.extract.isotonic_calibration import (
    CalibratedNLIScorer,
    IsotonicCalibrator,
    ece_within_release_gate,
    fit_per_backend_ece,
)

# ---------------------------------------------------------------------------
# Public surface — release gate constant matches §6.5
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_release_gate_constant_matches_spec() -> None:
    """§6.5 release gate is ECE <= 0.05; the helper threshold must agree."""
    assert ece_within_release_gate(0.04) is True
    assert ece_within_release_gate(CALIBRATION_ECE_THRESHOLD) is True
    assert ece_within_release_gate(0.0500001) is False


# ---------------------------------------------------------------------------
# PAV algorithm correctness on a small worked example
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_pav_pools_a_known_violator_block() -> None:
    """PAV on [(0.1, 0), (0.4, 1), (0.5, 0), (0.9, 1)] pools the middle pair.

    Sorted-by-raw correctness sequence is [0, 1, 0, 1]. Index 1 has
    avg 1.0 and index 2 has avg 0.0 — a violator. PAV pools indices
    1 and 2 into a block of mean 0.5. Final calibrated breakpoints:

        raw 0.1 -> 0.0
        raw 0.4 -> 0.5  (pooled with 0.5)
        raw 0.5 -> 0.5
        raw 0.9 -> 1.0
    """
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.1, 0.4, 0.5, 0.9], correct=[0, 1, 0, 1])

    assert cal.transform(0.1) == pytest.approx(0.0)
    assert cal.transform(0.4) == pytest.approx(0.5)
    assert cal.transform(0.5) == pytest.approx(0.5)
    assert cal.transform(0.9) == pytest.approx(1.0)


@pytest.mark.family_determinism
def test_already_monotonic_input_is_preserved() -> None:
    """If correctness is already non-decreasing in raw, no pooling occurs."""
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.1, 0.3, 0.6, 0.9], correct=[0, 0, 1, 1])

    assert cal.transform(0.1) == pytest.approx(0.0)
    assert cal.transform(0.3) == pytest.approx(0.0)
    assert cal.transform(0.6) == pytest.approx(1.0)
    assert cal.transform(0.9) == pytest.approx(1.0)


@pytest.mark.family_determinism
def test_unsorted_input_is_sorted_before_pav() -> None:
    """Caller may pass raw_scores in any order; fit sorts internally."""
    cal_sorted = IsotonicCalibrator()
    cal_sorted.fit(raw_scores=[0.1, 0.4, 0.5, 0.9], correct=[0, 1, 0, 1])

    cal_unsorted = IsotonicCalibrator()
    cal_unsorted.fit(raw_scores=[0.5, 0.9, 0.1, 0.4], correct=[0, 1, 0, 1])

    for probe in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert cal_sorted.transform(probe) == pytest.approx(cal_unsorted.transform(probe))


# ---------------------------------------------------------------------------
# Output invariants — monotonicity, clipping, interpolation
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_transform_is_monotonic_non_decreasing() -> None:
    """For any two raw inputs a <= b, transform(a) <= transform(b)."""
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.1, 0.4, 0.5, 0.9], correct=[0, 1, 0, 1])

    probes = [i / 20.0 for i in range(21)]
    calibrated = [cal.transform(p) for p in probes]
    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i + 1] + 1e-9


@pytest.mark.family_determinism
def test_extrapolation_clamps_to_endpoint_values() -> None:
    """Probes below min(raw) clamp to min calibrated; above max(raw) to max."""
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.2, 0.5, 0.8], correct=[0, 1, 1])

    # min calibrated = 0.0 (sole 0 in the block before pooling), max = 1.0
    assert cal.transform(0.0) == pytest.approx(cal.transform(0.2))
    assert cal.transform(1.0) == pytest.approx(cal.transform(0.8))


@pytest.mark.family_determinism
def test_interpolation_between_breakpoints_is_linear() -> None:
    """Between two breakpoints, transform interpolates linearly."""
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.0, 1.0], correct=[0, 1])

    # Two breakpoints: (0.0, 0.0) and (1.0, 1.0). Midpoint -> 0.5.
    assert cal.transform(0.5) == pytest.approx(0.5)
    assert cal.transform(0.25) == pytest.approx(0.25)
    assert cal.transform(0.75) == pytest.approx(0.75)


@pytest.mark.family_determinism
def test_transform_output_is_clipped_to_unit_interval() -> None:
    """Calibrated probabilities live in [0, 1] under all inputs."""
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.1, 0.4, 0.5, 0.9], correct=[0, 1, 0, 1])

    for probe in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        out = cal.transform(probe)
        assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# Determinism — same input bytes -> same calibrator
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_repeat_fit_is_byte_stable() -> None:
    """Two calibrators fit on the same data emit identical transforms."""
    data_raw = [0.1, 0.2, 0.35, 0.4, 0.6, 0.7, 0.85, 0.95]
    data_correct = [0, 0, 1, 0, 1, 1, 0, 1]

    cal_a = IsotonicCalibrator()
    cal_a.fit(raw_scores=data_raw, correct=data_correct)

    cal_b = IsotonicCalibrator()
    cal_b.fit(raw_scores=list(data_raw), correct=list(data_correct))

    for probe in [i / 50.0 for i in range(51)]:
        assert cal_a.transform(probe) == cal_b.transform(probe)


# ---------------------------------------------------------------------------
# Input validation — fail fast on caller errors
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_fit_rejects_length_mismatch() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(ValueError, match="length mismatch"):
        cal.fit(raw_scores=[0.1, 0.2], correct=[1])


@pytest.mark.family_determinism
def test_fit_rejects_empty_input() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(ValueError, match="empty"):
        cal.fit(raw_scores=[], correct=[])


@pytest.mark.family_determinism
def test_fit_rejects_non_binary_correctness() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(ValueError, match="binary"):
        cal.fit(raw_scores=[0.1, 0.5], correct=[0, 2])


@pytest.mark.family_determinism
def test_fit_rejects_out_of_range_raw_scores() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(ValueError, match="unit interval"):
        cal.fit(raw_scores=[0.1, 1.5], correct=[0, 1])


@pytest.mark.family_determinism
def test_transform_before_fit_raises() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="fit"):
        cal.transform(0.5)


# ---------------------------------------------------------------------------
# Release gate — ECE compresses below 0.05 after fitting on a held-out batch
# ---------------------------------------------------------------------------


def _miscalibrated_top_confidence(true_p: float, inflation: float = 0.20) -> float:
    """Synthetic backend: top confidence over-reports by a fixed offset.

    Empirical correctness probability is `true_p`; emitted top
    confidence is `min(true_p + inflation, 1.0)`. This is the classic
    over-confident pattern that isotonic regression compresses.
    """
    return min(true_p + inflation, 1.0)


@pytest.mark.family_verifier_calibration
def test_isotonic_calibration_brings_ece_under_release_gate() -> None:
    """Miscalibrated backend ECE > 0.05; after isotonic fit ECE <= 0.05.

    Build a controlled, miscalibrated batch where the true success
    probability per case is `p` and the backend over-reports
    confidence by 0.20. Pre-fit aggregate ECE clears the §6.5 release
    gate. Post-fit, transformed top confidences should bring ECE
    back under the gate. Fit and held-out halves are interleaved so
    both cover the full raw-score range and the calibrator does not
    have to extrapolate.
    """
    # Generate 200 cases evenly across 5 true-probability bands.
    # For each band, the backend emits an inflated top confidence; the
    # gold correctness is sampled in a deterministic pattern so the
    # empirical correct-rate matches the band.
    fit_raw: list[float] = []
    fit_correct: list[int] = []
    test_raw: list[float] = []
    test_correct: list[int] = []
    bands = [0.10, 0.30, 0.50, 0.70, 0.90]
    cases_per_band = 40
    for true_p in bands:
        n_correct = round(true_p * cases_per_band)
        for j in range(cases_per_band):
            raw = _miscalibrated_top_confidence(true_p)
            ok = 1 if j < n_correct else 0
            # Even-indexed cases fit the calibrator; odd-indexed cases
            # are the held-out evaluation. Both halves see every band.
            if j % 2 == 0:
                fit_raw.append(raw)
                fit_correct.append(ok)
            else:
                test_raw.append(raw)
                test_correct.append(ok)

    # Pre-fit ECE on the held-out half: build NLIScore objects where
    # the top label is entailment with the inflated raw confidence.
    # Gold is entailment when correct=1 else neutral, so per-case
    # argmax matches gold iff correct=1 — the calibration target is
    # the top-label confidence vs. correctness.
    predictions_pre: list[NLIScore] = []
    golds: list[CalibrationLabel] = []
    for score, ok in zip(test_raw, test_correct, strict=True):
        predictions_pre.append(
            NLIScore(entailment=score, contradiction=0.0, neutral=1.0 - score),
        )
        golds.append("entailment" if ok == 1 else "neutral")
    pre_ece = expected_calibration_error(predictions=predictions_pre, golds=golds)
    assert (
        pre_ece > CALIBRATION_ECE_THRESHOLD
    ), f"synthetic backend must be miscalibrated above the release gate; got pre_ece={pre_ece:.4f}"

    # Fit on the training half.
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=fit_raw, correct=fit_correct)

    # Post-fit: rebuild predictions on the held-out half using the
    # calibrated top-label confidence. Remainder splits onto neutral
    # so the softmax sums to 1 and the argmax stays on entailment.
    predictions_post: list[NLIScore] = []
    for score in test_raw:
        calibrated_top = cal.transform(score)
        remainder = max(0.0, 1.0 - calibrated_top)
        predictions_post.append(
            NLIScore(entailment=calibrated_top, contradiction=0.0, neutral=remainder),
        )

    post_ece = expected_calibration_error(predictions=predictions_post, golds=golds)
    assert post_ece <= CALIBRATION_ECE_THRESHOLD, (
        f"isotonic calibration must bring ECE under the §6.5 release gate "
        f"(0.05); pre={pre_ece:.4f}, post={post_ece:.4f}"
    )


# ---------------------------------------------------------------------------
# CalibratedNLIScorer wrapper — preserves argmax, recalibrates confidence
# ---------------------------------------------------------------------------


class _FixedScorer:
    """An `NLIScorer` that returns a pre-built score per `(premise, hypothesis)`."""

    def __init__(self, table: dict[tuple[str, str], NLIScore]) -> None:
        self._table = table

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        return self._table[(premise, hypothesis)]


@pytest.mark.family_verifier_calibration
def test_calibrated_wrapper_preserves_argmax_label() -> None:
    """Wrapping does not change which label has the highest mass."""
    inner = _FixedScorer({("p", "h"): NLIScore(entailment=0.80, contradiction=0.05, neutral=0.15)})
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.20, 0.50, 0.80], correct=[0, 0, 1])
    wrapped = CalibratedNLIScorer(inner=inner, calibrator=cal)

    out = wrapped.score(premise="p", hypothesis="h")
    assert out.argmax_label() == "entailment"


@pytest.mark.family_verifier_calibration
def test_calibrated_wrapper_replaces_top_confidence_with_calibrated_value() -> None:
    """The wrapper's top confidence is the calibrator output, not the raw."""
    inner = _FixedScorer({("p", "h"): NLIScore(entailment=0.80, contradiction=0.05, neutral=0.15)})
    # Identity-ish calibrator that maps 0.80 -> 0.40 (severe deflation).
    cal = IsotonicCalibrator()
    cal.fit(
        raw_scores=[0.20, 0.50, 0.80],
        correct=[0, 0, 1],
    )
    # PAV on [0, 0, 1] is already monotone, so transform(0.80) = 1.0.
    # Override with an explicit fit that yields the 0.40 mapping at 0.80.
    cal2 = IsotonicCalibrator()
    cal2.fit(
        raw_scores=[0.20, 0.50, 0.80, 0.90],
        correct=[0, 1, 0, 1],
    )
    # PAV pools (0.50 -> 1, 0.80 -> 0) into mean 0.5; check the contract.
    assert cal2.transform(0.80) == pytest.approx(0.5)
    wrapped = CalibratedNLIScorer(inner=inner, calibrator=cal2)

    out = wrapped.score(premise="p", hypothesis="h")
    assert out.top_confidence() == pytest.approx(0.5)


@pytest.mark.family_verifier_calibration
def test_calibrated_wrapper_emits_normalized_distribution() -> None:
    """Output NLIScore components still sum to 1.0 within tolerance."""
    inner = _FixedScorer({("p", "h"): NLIScore(entailment=0.80, contradiction=0.05, neutral=0.15)})
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.20, 0.50, 0.80, 0.90], correct=[0, 1, 0, 1])
    wrapped = CalibratedNLIScorer(inner=inner, calibrator=cal)

    out = wrapped.score(premise="p", hypothesis="h")
    total = out.entailment + out.contradiction + out.neutral
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.family_verifier_calibration
def test_calibrated_wrapper_distributes_remainder_proportionally() -> None:
    """Non-top labels keep their relative ratio after re-normalisation."""
    inner = _FixedScorer({("p", "h"): NLIScore(entailment=0.80, contradiction=0.05, neutral=0.15)})
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.20, 0.50, 0.80, 0.90], correct=[0, 1, 0, 1])
    # transform(0.80) == 0.5 -> remainder = 0.5 for the other two labels.
    wrapped = CalibratedNLIScorer(inner=inner, calibrator=cal)
    out = wrapped.score(premise="p", hypothesis="h")

    # Raw ratio of contradiction:neutral was 0.05 : 0.15 = 1 : 3.
    # After re-normalising to total 0.5: 0.125 : 0.375.
    assert out.contradiction == pytest.approx(0.125)
    assert out.neutral == pytest.approx(0.375)


@pytest.mark.family_verifier_calibration
def test_calibrated_wrapper_splits_remainder_evenly_when_raw_others_are_zero() -> None:
    """When the non-top labels both carry zero raw mass and calibration leaves
    a positive remainder, the remainder splits evenly across the other labels.

    This is the only path that risks a div-by-zero in a proportional
    re-normalisation, so the wrapper must take the even-split branch
    deterministically.
    """
    # Raw distribution: entailment hogs all the mass (0.80), the other
    # two carry zero. Calibrator deflates the top to 0.50 -- remainder
    # is 0.50 with zero raw mass to redistribute against. Even split
    # is the only well-defined choice.
    inner = _FixedScorer({("p", "h"): NLIScore(entailment=0.80, contradiction=0.0, neutral=0.20)})
    cal = IsotonicCalibrator()
    cal.fit(raw_scores=[0.20, 0.50, 0.80, 0.90], correct=[0, 1, 0, 1])
    # transform(0.80) -> 0.5 (verified in another test).
    wrapped = CalibratedNLIScorer(inner=inner, calibrator=cal)
    out = wrapped.score(premise="p", hypothesis="h")
    assert out.entailment == pytest.approx(0.5)
    # Raw split contradiction:neutral = 0.0 : 0.20 = 0 : 1, so proportional
    # re-normalisation puts the entire remainder onto neutral.
    assert out.contradiction == pytest.approx(0.0)
    assert out.neutral == pytest.approx(0.5)

    # Now exercise the genuinely degenerate path: both non-top raw masses
    # are zero. Even split divides the remainder equally between the
    # other two labels.
    inner_all_top = _FixedScorer(
        {("p", "h"): NLIScore(entailment=1.0, contradiction=0.0, neutral=0.0)},
    )
    # Use a calibrator whose highest breakpoint sits at 0.5 so the
    # transform deflates (otherwise transform(1.0) -> 1.0 and the
    # remainder is zero, masking the even-split path).
    cal_low = IsotonicCalibrator()
    cal_low.fit(raw_scores=[0.20, 0.50, 0.80], correct=[0, 1, 0])
    # PAV on [0,1,0] -> pool indices 1,2 to mean 0.5; transform(1.0)=0.5.
    assert cal_low.transform(1.0) == pytest.approx(0.5)
    wrapped_low = CalibratedNLIScorer(inner=inner_all_top, calibrator=cal_low)
    out_low = wrapped_low.score(premise="p", hypothesis="h")
    assert out_low.entailment == pytest.approx(0.5)
    assert out_low.contradiction == pytest.approx(0.25)
    assert out_low.neutral == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Helper — fit_per_backend_ece reports headline ECE the release gate uses
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_fit_per_backend_ece_returns_post_fit_ece() -> None:
    """`fit_per_backend_ece` fits a calibrator and reports the held-out ECE.

    Inputs are interleaved across true-probability bands AND across
    correctness within each band so the helper's contiguous split-
    half sees the same empirical correctness rate per raw value on
    both sides. Otherwise PAV fits one regime and the held-out
    evaluation lives in a different one — the calibrator would still
    work in production but the unit-level assertion needs same-
    distribution halves.
    """
    bands = [0.10, 0.30, 0.50, 0.70, 0.90]
    cases_per_band = 40
    # Build per-band correctness sequences, alternating 1s and 0s in
    # the same proportion so any contiguous window of size 2 holds
    # the band's average correct rate on average.
    per_band_pairs: list[list[tuple[float, int]]] = []
    for true_p in bands:
        n_correct = round(true_p * cases_per_band)
        n_total = cases_per_band
        # Stride-place the correct labels uniformly across the band.
        labels = [0] * n_total
        if n_correct > 0:
            stride = n_total / n_correct
            placed = {int(i * stride) for i in range(n_correct)}
            for idx in placed:
                labels[min(idx, n_total - 1)] = 1
        raw = min(true_p + 0.20, 1.0)
        per_band_pairs.append([(raw, label) for label in labels])

    raw_scores: list[float] = []
    correct: list[int] = []
    for j in range(cases_per_band):
        for band_pairs in per_band_pairs:
            raw, label = band_pairs[j]
            raw_scores.append(raw)
            correct.append(label)

    post_ece, calibrator = fit_per_backend_ece(raw_scores=raw_scores, correct=correct)
    assert isinstance(calibrator, IsotonicCalibrator)
    assert post_ece <= CALIBRATION_ECE_THRESHOLD
