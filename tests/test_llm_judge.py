"""LLM-as-judge bias-control invariants for UC4 / UC5 evals.

Per §8.7 the substrate ships deterministic helpers around an
LLM-judge so UC4 / UC5 subjective evals stay honest:

  - Rubric: 3-5 dimensions, each on a 1-5 scale, with exemplars.
  - Bias controls: blind-shuffle, A/B swap, multi-seed averaging.
  - Inter-rater check: Cohen's κ ≥ 0.7 on the 10% human overlap.
  - Stability check: per-dimension variance < 0.5.
  - Drift tracking: anchor outputs scored on every commit; >0.5
    mean shift on any dimension is an alert.

The tests are hermetic — `_DeterministicJudge` is a stub that
returns scripted scores per (seed, output_text). Real LLM-backed
judges plug into the same `LLMJudge` Protocol.

SPEC-REF: §8.7
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import pytest
from pydantic import ValidationError

from ctrldoc.eval.llm_judge import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_KAPPA_FLOOR,
    DEFAULT_N_SEEDS,
    DEFAULT_STABILITY_THRESHOLD,
    AggregatedScore,
    JudgeScore,
    LLMJudge,
    PairwiseLLMJudge,
    Rubric,
    RubricDimension,
    bias_controlled_pairwise,
    bias_controlled_score,
    blind_shuffle,
    cohens_kappa,
    flag_judge_drift,
    flag_unstable,
    inter_rater_check,
    validate_score_against_rubric,
)


def _rubric(name: str = "review") -> Rubric:
    return Rubric(
        name=name,
        dimensions=[
            RubricDimension(name="clarity", description="Is the output clear?"),
            RubricDimension(name="evidence", description="Are claims well-supported?"),
            RubricDimension(name="depth", description="Does it cover non-trivial cases?"),
        ],
    )


# --- Rubric validation ---


def test_rubric_accepts_three_dimensions() -> None:
    r = _rubric()
    assert r.dimension_names() == ["clarity", "evidence", "depth"]


def test_rubric_accepts_five_dimensions() -> None:
    dims = [RubricDimension(name=f"d-{i}", description=f"desc {i}") for i in range(5)]
    r = Rubric(name="x", dimensions=dims)
    assert len(r.dimensions) == 5


def test_rubric_rejects_two_dimensions() -> None:
    with pytest.raises(ValidationError, match="3-5 dimensions"):
        Rubric(
            name="x",
            dimensions=[
                RubricDimension(name="a", description="x"),
                RubricDimension(name="b", description="y"),
            ],
        )


def test_rubric_rejects_six_dimensions() -> None:
    with pytest.raises(ValidationError, match="3-5 dimensions"):
        Rubric(
            name="x",
            dimensions=[RubricDimension(name=f"d-{i}", description="d") for i in range(6)],
        )


def test_rubric_rejects_duplicate_dimension_names() -> None:
    with pytest.raises(ValidationError, match="duplicate dimension names"):
        Rubric(
            name="x",
            dimensions=[
                RubricDimension(name="a", description="x"),
                RubricDimension(name="a", description="y"),
                RubricDimension(name="b", description="z"),
            ],
        )


def test_rubric_dimension_scale_must_be_well_ordered() -> None:
    with pytest.raises(ValidationError, match="scale_max"):
        RubricDimension(name="a", description="x", scale_min=5, scale_max=3)


# --- JudgeScore validation against rubric ---


def test_validate_score_round_trip() -> None:
    rubric = _rubric()
    score = JudgeScore(
        rubric_name="review",
        seed=0,
        per_dimension={"clarity": 4, "evidence": 3, "depth": 5},
    )
    validate_score_against_rubric(score, rubric)


def test_validate_score_rejects_missing_dimension() -> None:
    rubric = _rubric()
    score = JudgeScore(
        rubric_name="review",
        seed=0,
        per_dimension={"clarity": 4, "evidence": 3},
    )
    with pytest.raises(ValueError, match="missing=\\['depth'\\]"):
        validate_score_against_rubric(score, rubric)


def test_validate_score_rejects_extra_dimension() -> None:
    rubric = _rubric()
    score = JudgeScore(
        rubric_name="review",
        seed=0,
        per_dimension={"clarity": 4, "evidence": 3, "depth": 5, "ghost": 1},
    )
    with pytest.raises(ValueError, match="extra=\\['ghost'\\]"):
        validate_score_against_rubric(score, rubric)


def test_validate_score_rejects_out_of_range() -> None:
    rubric = _rubric()
    score = JudgeScore(
        rubric_name="review",
        seed=0,
        per_dimension={"clarity": 6, "evidence": 3, "depth": 5},
    )
    with pytest.raises(ValueError, match="out of range"):
        validate_score_against_rubric(score, rubric)


def test_validate_score_rejects_rubric_name_mismatch() -> None:
    rubric = _rubric()
    score = JudgeScore(
        rubric_name="other",
        seed=0,
        per_dimension={"clarity": 4, "evidence": 3, "depth": 5},
    )
    with pytest.raises(ValueError, match="rubric mismatch"):
        validate_score_against_rubric(score, rubric)


# --- stub judges ---


@dataclass
class _DeterministicJudge:
    """Returns scripted scores keyed by (seed, output_text) → per_dim dict.

    Falls back to a uniform score when no entry matches.
    """

    rubric_name: str
    table: dict[tuple[int, str], dict[str, int]] = field(default_factory=dict)
    default: dict[str, int] = field(
        default_factory=lambda: {"clarity": 3, "evidence": 3, "depth": 3}
    )

    def score(self, output_text: str, *, rubric: Rubric, seed: int) -> JudgeScore:
        per_dim = self.table.get((seed, output_text), self.default)
        return JudgeScore(
            rubric_name=self.rubric_name,
            seed=seed,
            per_dimension=dict(per_dim),
        )


@dataclass
class _PairwiseStubJudge:
    """Scripted pairwise judge: returns the chosen outcome per `(a, b, seed)`."""

    table: dict[tuple[str, str, int], Literal["a", "b", "tie"]] = field(default_factory=dict)
    default: Literal["a", "b", "tie"] = "tie"

    def compare(
        self,
        a_text: str,
        b_text: str,
        *,
        rubric: Rubric,
        seed: int,
    ) -> Literal["a", "b", "tie"]:
        return self.table.get((a_text, b_text, seed), self.default)


# --- protocol conformance ---


def test_deterministic_judge_satisfies_llm_judge_protocol() -> None:
    judge = _DeterministicJudge(rubric_name="review")
    assert isinstance(judge, LLMJudge)


def test_pairwise_stub_satisfies_pairwise_protocol() -> None:
    judge = _PairwiseStubJudge()
    assert isinstance(judge, PairwiseLLMJudge)


# --- bias_controlled_score ---


def test_bias_controlled_score_averages_across_seeds() -> None:
    rubric = _rubric()
    judge = _DeterministicJudge(
        rubric_name="review",
        table={
            (0, "out"): {"clarity": 5, "evidence": 4, "depth": 4},
            (1, "out"): {"clarity": 3, "evidence": 4, "depth": 4},
            (2, "out"): {"clarity": 4, "evidence": 4, "depth": 4},
        },
    )
    aggregated = bias_controlled_score(
        "out",
        judge=judge,
        rubric=rubric,
        n_seeds=3,
    )
    assert aggregated.per_dimension_mean["clarity"] == pytest.approx(4.0)
    assert aggregated.per_dimension_mean["evidence"] == pytest.approx(4.0)
    assert aggregated.per_dimension_mean["depth"] == pytest.approx(4.0)


def test_bias_controlled_score_records_per_dimension_variance() -> None:
    """When the three seeds emit 5/3/4 the dimension's population
    variance is ((5-4)^2 + (3-4)^2 + (4-4)^2)/3 = 2/3 ≈ 0.667."""
    rubric = _rubric()
    judge = _DeterministicJudge(
        rubric_name="review",
        table={
            (0, "out"): {"clarity": 5, "evidence": 3, "depth": 3},
            (1, "out"): {"clarity": 3, "evidence": 3, "depth": 3},
            (2, "out"): {"clarity": 4, "evidence": 3, "depth": 3},
        },
    )
    aggregated = bias_controlled_score("out", judge=judge, rubric=rubric, n_seeds=3)
    assert aggregated.per_dimension_variance["clarity"] == pytest.approx(2 / 3)
    assert aggregated.per_dimension_variance["evidence"] == pytest.approx(0.0)


def test_bias_controlled_score_default_seeds_are_range_n() -> None:
    """Two callers using n_seeds=3 must hit seeds {0, 1, 2}, so their
    runs are directly comparable."""
    rubric = _rubric()

    @dataclass
    class _SeedRecordingJudge:
        rubric_name: str
        seen_seeds: list[int] = field(default_factory=list)

        def score(self, output_text: str, *, rubric: Rubric, seed: int) -> JudgeScore:
            self.seen_seeds.append(seed)
            return JudgeScore(
                rubric_name=self.rubric_name,
                seed=seed,
                per_dimension={"clarity": 3, "evidence": 3, "depth": 3},
            )

    judge = _SeedRecordingJudge(rubric_name="review")
    bias_controlled_score("out", judge=judge, rubric=rubric, n_seeds=3)
    assert judge.seen_seeds == [0, 1, 2]


def test_bias_controlled_score_zero_seeds_rejected() -> None:
    judge = _DeterministicJudge(rubric_name="review")
    with pytest.raises(ValueError, match="at least one seed"):
        bias_controlled_score("out", judge=judge, rubric=_rubric(), seeds=[])


def test_overall_mean_and_max_variance() -> None:
    """Aggregated convenience properties roll up across dimensions."""
    rubric = _rubric()
    judge = _DeterministicJudge(
        rubric_name="review",
        table={
            (0, "out"): {"clarity": 5, "evidence": 3, "depth": 3},
            (1, "out"): {"clarity": 1, "evidence": 3, "depth": 3},
        },
    )
    aggregated = bias_controlled_score("out", judge=judge, rubric=rubric, n_seeds=2)
    assert aggregated.overall_mean == pytest.approx((3.0 + 3.0 + 3.0) / 3)
    assert aggregated.max_dimension_variance == pytest.approx(
        4.0
    )  # clarity = (5-3)^2 + (1-3)^2 / 2


# --- stability flag ---


def test_flag_unstable_returns_empty_on_low_variance() -> None:
    """Default threshold is 0.5; variance below that is healthy."""
    agg = AggregatedScore(
        rubric_name="review",
        per_dimension_mean={"clarity": 4.0},
        per_dimension_variance={"clarity": 0.25},
        raw_scores=(),
    )
    assert flag_unstable(agg) == []


def test_flag_unstable_returns_dimension_above_threshold() -> None:
    agg = AggregatedScore(
        rubric_name="review",
        per_dimension_mean={"clarity": 4.0, "evidence": 4.0},
        per_dimension_variance={"clarity": 0.667, "evidence": 0.1},
        raw_scores=(),
    )
    assert flag_unstable(agg) == ["clarity"]


def test_flag_unstable_strict_inequality_at_threshold() -> None:
    """A dimension exactly at the threshold is not flagged — the
    threshold is the strict-greater boundary."""
    agg = AggregatedScore(
        rubric_name="review",
        per_dimension_mean={"clarity": 4.0},
        per_dimension_variance={"clarity": DEFAULT_STABILITY_THRESHOLD},
        raw_scores=(),
    )
    assert flag_unstable(agg) == []


# --- pairwise A/B swap ---


def test_pairwise_unbiased_consistent_winner() -> None:
    """If the judge always picks the same content regardless of A/B
    position, the seed is unbiased and the winner is the content."""
    rubric = _rubric()
    judge = _PairwiseStubJudge(
        table={
            ("alpha", "beta", 0): "a",
            ("beta", "alpha", 0): "b",  # reversed frame ⇒ same content wins
            ("alpha", "beta", 1): "a",
            ("beta", "alpha", 1): "b",
            ("alpha", "beta", 2): "a",
            ("beta", "alpha", 2): "b",
        },
    )
    result = bias_controlled_pairwise(
        "alpha",
        "beta",
        judge=judge,
        rubric=rubric,
        n_seeds=3,
    )
    assert result.winner == "a"
    assert result.position_bias_detected is False


def test_pairwise_majority_seed_bias_returns_biased() -> None:
    """When most seeds show position bias (forward and reverse pick
    different content), the aggregate is `biased`."""
    rubric = _rubric()
    # Judge picks position-1 every time → forward says "a" and
    # reversed also says "a" (which un-swaps to "b"), so every seed
    # is biased.
    judge = _PairwiseStubJudge(default="a")
    result = bias_controlled_pairwise(
        "alpha",
        "beta",
        judge=judge,
        rubric=rubric,
        n_seeds=3,
    )
    assert result.winner == "biased"
    assert result.position_bias_detected is True


def test_pairwise_tie_when_no_majority() -> None:
    """Three unbiased seeds split 1-1-1 across {a, b, tie} → tie."""
    rubric = _rubric()
    judge = _PairwiseStubJudge(
        table={
            # Seed 0: a wins both → forward "a", reverse "b"
            ("alpha", "beta", 0): "a",
            ("beta", "alpha", 0): "b",
            # Seed 1: b wins both → forward "b", reverse "a"
            ("alpha", "beta", 1): "b",
            ("beta", "alpha", 1): "a",
            # Seed 2: tie both ways
            ("alpha", "beta", 2): "tie",
            ("beta", "alpha", 2): "tie",
        },
    )
    result = bias_controlled_pairwise(
        "alpha",
        "beta",
        judge=judge,
        rubric=rubric,
        n_seeds=3,
    )
    assert result.winner == "tie"
    assert result.position_bias_detected is False


# --- Cohen's kappa ---


def test_kappa_perfect_agreement_returns_one() -> None:
    ratings = [1, 3, 5, 2, 4, 3, 1, 5]
    assert cohens_kappa(ratings, ratings) == pytest.approx(1.0)


def test_kappa_no_agreement_returns_negative() -> None:
    """Completely disagreeing raters → κ ≤ 0 (the chance-correction
    can push it below zero when observed < expected)."""
    a = [1, 2, 3, 4, 5]
    b = [5, 4, 3, 2, 1]
    k = cohens_kappa(a, b)
    assert k < 0.5
    assert k > -1.0


def test_kappa_empty_returns_zero() -> None:
    assert cohens_kappa([], []) == 0.0


def test_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mismatched lengths"):
        cohens_kappa([1, 2, 3], [1, 2])


def test_kappa_random_uncorrelated_near_zero() -> None:
    """Two raters with identical marginal distributions but no real
    correlation should land near κ=0."""
    # Permuted pair: same multiset of scores but shifted by one.
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    b = [2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
    k = cohens_kappa(a, b)
    assert k == pytest.approx(0.0, abs=0.30)


# --- inter-rater check ---


def test_inter_rater_check_passes_at_perfect_agreement() -> None:
    judge = [3, 4, 5, 2, 1, 3, 4, 5]
    human = list(judge)
    report = inter_rater_check(judge, human)
    assert report.kappa == pytest.approx(1.0)
    assert report.passed is True
    assert report.threshold == pytest.approx(DEFAULT_KAPPA_FLOOR)


def test_inter_rater_check_fails_below_floor() -> None:
    judge = [1, 2, 3, 4, 5]
    human = [5, 4, 3, 2, 1]
    report = inter_rater_check(judge, human)
    assert report.passed is False


def test_inter_rater_check_threshold_can_be_overridden() -> None:
    """A stricter project policy can demand higher κ."""
    judge = [1, 2, 3, 4, 5]
    human = [1, 2, 3, 4, 4]
    relaxed = inter_rater_check(judge, human, min_kappa=0.5)
    strict = inter_rater_check(judge, human, min_kappa=0.99)
    assert relaxed.passed is True
    assert strict.passed is False


def test_inter_rater_check_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        inter_rater_check([1, 2, 3], [1, 2])


# --- judge drift ---


def _agg(rubric: Rubric, means: dict[str, float]) -> AggregatedScore:
    return AggregatedScore(
        rubric_name=rubric.name,
        per_dimension_mean=dict(means),
        per_dimension_variance=dict.fromkeys(means, 0.0),
        raw_scores=(),
    )


def test_flag_judge_drift_within_threshold_clean() -> None:
    rubric = _rubric()
    base = _agg(rubric, {"clarity": 4.0, "evidence": 3.5, "depth": 3.0})
    cur = _agg(rubric, {"clarity": 4.2, "evidence": 3.5, "depth": 3.3})
    report = flag_judge_drift(base, cur)
    assert report.flagged is False
    assert report.max_shift == pytest.approx(0.3, abs=1e-9)


def test_flag_judge_drift_above_threshold_flags() -> None:
    rubric = _rubric()
    base = _agg(rubric, {"clarity": 4.0, "evidence": 3.5, "depth": 3.0})
    cur = _agg(rubric, {"clarity": 4.6, "evidence": 3.5, "depth": 3.0})
    report = flag_judge_drift(base, cur)
    assert report.flagged is True
    assert "clarity" in report.flagged_dimensions
    assert "evidence" not in report.flagged_dimensions


def test_flag_judge_drift_negative_shift_also_flagged() -> None:
    rubric = _rubric()
    base = _agg(rubric, {"clarity": 4.5, "evidence": 3.5, "depth": 3.0})
    cur = _agg(rubric, {"clarity": 3.5, "evidence": 3.5, "depth": 3.0})  # -1.0
    report = flag_judge_drift(base, cur)
    assert report.flagged is True
    assert "clarity" in report.flagged_dimensions


def test_flag_judge_drift_threshold_default_matches_spec() -> None:
    assert pytest.approx(0.5) == DEFAULT_DRIFT_THRESHOLD


def test_flag_judge_drift_rubric_mismatch_raises() -> None:
    base = _agg(_rubric("a"), {"clarity": 4.0, "evidence": 3.0, "depth": 3.0})
    cur = _agg(_rubric("b"), {"clarity": 4.0, "evidence": 3.0, "depth": 3.0})
    with pytest.raises(ValueError, match="rubric mismatch"):
        flag_judge_drift(base, cur)


def test_flag_judge_drift_missing_dimension_raises() -> None:
    base = _agg(_rubric(), {"clarity": 4.0, "evidence": 3.0, "depth": 3.0})
    cur = AggregatedScore(
        rubric_name="review",
        per_dimension_mean={"clarity": 4.0, "evidence": 3.0},  # depth missing
        per_dimension_variance={"clarity": 0.0, "evidence": 0.0},
        raw_scores=(),
    )
    with pytest.raises(ValueError, match="missing dimension"):
        flag_judge_drift(base, cur)


# --- blind shuffle ---


def test_blind_shuffle_returns_every_input_exactly_once() -> None:
    outs = ["alpha", "beta", "gamma", "delta", "epsilon"]
    shuffled = blind_shuffle(outs, seed=42)
    assert len(shuffled) == len(outs)
    assert sorted(idx for idx, _ in shuffled) == list(range(len(outs)))
    assert {text for _, text in shuffled} == set(outs)


def test_blind_shuffle_is_deterministic_for_same_seed() -> None:
    outs = ["alpha", "beta", "gamma", "delta"]
    a = blind_shuffle(outs, seed=7)
    b = blind_shuffle(outs, seed=7)
    assert a == b


def test_blind_shuffle_differs_across_seeds() -> None:
    """Two non-trivial seeds should produce different orders for a
    multi-element input."""
    outs = [f"out-{i}" for i in range(10)]
    a = blind_shuffle(outs, seed=1)
    b = blind_shuffle(outs, seed=999)
    # Order must differ (vanishingly unlikely to match by chance).
    assert a != b


# --- defaults ---


def test_default_constants_match_section_87() -> None:
    assert DEFAULT_N_SEEDS == 3
    assert pytest.approx(0.5) == DEFAULT_STABILITY_THRESHOLD
    assert pytest.approx(0.70) == DEFAULT_KAPPA_FLOOR
    assert pytest.approx(0.5) == DEFAULT_DRIFT_THRESHOLD


# --- end-to-end composition ---


def test_full_pipeline_passes_for_clean_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score an output across 3 seeds, confirm stability + drift flags are clean
    and the κ overlap against a perfectly-agreeing human passes."""
    rubric = _rubric()
    judge = _DeterministicJudge(
        rubric_name="review",
        table={
            (0, "out"): {"clarity": 4, "evidence": 4, "depth": 4},
            (1, "out"): {"clarity": 4, "evidence": 4, "depth": 4},
            (2, "out"): {"clarity": 4, "evidence": 4, "depth": 4},
        },
    )
    aggregated = bias_controlled_score("out", judge=judge, rubric=rubric, n_seeds=3)

    # Stability flag: variance is zero on every dimension.
    assert flag_unstable(aggregated) == []

    # Inter-rater: trivial human agreement on the three raw scores.
    judge_dim = [score.per_dimension["clarity"] for score in aggregated.raw_scores]
    human_dim: Sequence[int] = list(judge_dim)
    report = inter_rater_check(judge_dim, human_dim)
    assert report.passed is True

    # Drift: same aggregated values vs themselves → no flag.
    drift = flag_judge_drift(aggregated, aggregated)
    assert drift.flagged is False
