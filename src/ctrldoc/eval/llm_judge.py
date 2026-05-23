"""LLM-as-judge eval helpers with §8.7 bias controls.

UC4 (`analytical_review`) and UC5 (`anomaly_scan`) produce partly
subjective outputs that can't be fully scored against a gold answer.
§8.7 specifies the discipline:

  - The judge model must differ from the generator (no self-grading);
    callers wire this at the orchestrator layer.
  - Scoring uses a structured rubric: 3-5 dimensions, each on a 1-5
    scale, with exemplars.
  - Bias controls: blind-shuffle output order, A/B swap on pairwise
    comparisons, average over 3 seeds.
  - Inter-rater check: 10% of items also human-scored; require
    Cohen's κ ≥ 0.7.
  - Stability check: same output scored twice → variance < 0.5.
  - Drift tracking: anchor outputs scored on every commit; >0.5
    shift between runs is a regression.

This module ships the deterministic helpers — `Rubric`, `JudgeScore`,
`bias_controlled_score`, `bias_controlled_pairwise`, `cohens_kappa`,
`flag_judge_drift` — and the `LLMJudge` Protocol that production
LLM-backed implementations plug into. The helpers are pure-Python,
hermetic, and exercised in `tests/test_llm_judge.py`.

SPEC-REF: §8.7
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_N_SEEDS = 3
DEFAULT_STABILITY_THRESHOLD = 0.5
DEFAULT_KAPPA_FLOOR = 0.70
DEFAULT_DRIFT_THRESHOLD = 0.5
DEFAULT_HUMAN_SAMPLE_PCT = 0.10
SCALE_MIN = 1
SCALE_MAX = 5


# --- rubric ---


class RubricDimension(BaseModel):
    """One scoring axis of a rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    exemplars: list[str] = Field(default_factory=list)
    scale_min: int = SCALE_MIN
    scale_max: int = SCALE_MAX

    @model_validator(mode="after")
    def _scale_valid(self) -> RubricDimension:
        if self.scale_max <= self.scale_min:
            raise ValueError(
                f"scale_max ({self.scale_max}) must be greater than scale_min ({self.scale_min})"
            )
        return self


class Rubric(BaseModel):
    """Structured §8.7 rubric — 3-5 dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dimensions: list[RubricDimension]

    @model_validator(mode="after")
    def _dimension_count(self) -> Rubric:
        n = len(self.dimensions)
        if not 3 <= n <= 5:
            raise ValueError(f"rubric must have 3-5 dimensions, got {n}")
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dimension names: {names}")
        return self

    def dimension_names(self) -> list[str]:
        return [d.name for d in self.dimensions]


# --- judge output ---


class JudgeScore(BaseModel):
    """One scoring of an output across every rubric dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rubric_name: str
    seed: int
    per_dimension: dict[str, int]


def validate_score_against_rubric(score: JudgeScore, rubric: Rubric) -> None:
    """Raise `ValueError` if `score` doesn't satisfy `rubric`'s schema."""
    if score.rubric_name != rubric.name:
        raise ValueError(
            f"rubric mismatch: score.rubric_name={score.rubric_name!r}, rubric.name={rubric.name!r}"
        )
    expected = set(rubric.dimension_names())
    actual = set(score.per_dimension)
    if expected != actual:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(f"dimension mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for dim in rubric.dimensions:
        value = score.per_dimension[dim.name]
        if not (dim.scale_min <= value <= dim.scale_max):
            raise ValueError(
                f"dimension {dim.name!r}={value} out of range [{dim.scale_min}, {dim.scale_max}]"
            )


# --- judge protocols ---


@runtime_checkable
class LLMJudge(Protocol):
    """Single-output judge — score a generation against a rubric."""

    def score(self, output_text: str, *, rubric: Rubric, seed: int) -> JudgeScore: ...


@runtime_checkable
class PairwiseLLMJudge(Protocol):
    """Pairwise judge — pick which of two outputs is better."""

    def compare(
        self,
        a_text: str,
        b_text: str,
        *,
        rubric: Rubric,
        seed: int,
    ) -> Literal["a", "b", "tie"]: ...


# --- aggregation ---


@dataclass(frozen=True)
class AggregatedScore:
    """Mean + variance across multiple seeds, per rubric dimension."""

    rubric_name: str
    per_dimension_mean: dict[str, float]
    per_dimension_variance: dict[str, float]
    raw_scores: tuple[JudgeScore, ...]

    @property
    def overall_mean(self) -> float:
        if not self.per_dimension_mean:
            return 0.0
        return sum(self.per_dimension_mean.values()) / len(self.per_dimension_mean)

    @property
    def max_dimension_variance(self) -> float:
        if not self.per_dimension_variance:
            return 0.0
        return max(self.per_dimension_variance.values())


def bias_controlled_score(
    output_text: str,
    *,
    judge: LLMJudge,
    rubric: Rubric,
    seeds: Sequence[int] | None = None,
    n_seeds: int = DEFAULT_N_SEEDS,
) -> AggregatedScore:
    """Score one output across N seeds, returning per-dimension mean + variance.

    Default seeds are `range(n_seeds)` so two callers using the same
    `n_seeds` produce comparable runs. Pass an explicit `seeds`
    iterable to anchor against a fixed seed set.
    """
    chosen_seeds = list(seeds) if seeds is not None else list(range(n_seeds))
    if not chosen_seeds:
        raise ValueError("need at least one seed")
    scores: list[JudgeScore] = []
    for seed in chosen_seeds:
        score = judge.score(output_text, rubric=rubric, seed=seed)
        validate_score_against_rubric(score, rubric)
        scores.append(score)
    return _aggregate(scores, rubric)


def _aggregate(scores: Sequence[JudgeScore], rubric: Rubric) -> AggregatedScore:
    means: dict[str, float] = {}
    variances: dict[str, float] = {}
    for dim in rubric.dimensions:
        values = [score.per_dimension[dim.name] for score in scores]
        means[dim.name] = statistics.fmean(values)
        # `pvariance` gives the population variance — what §8.7's
        # variance threshold describes ("same output scored twice").
        variances[dim.name] = statistics.pvariance(values) if len(values) > 1 else 0.0
    return AggregatedScore(
        rubric_name=rubric.name,
        per_dimension_mean=means,
        per_dimension_variance=variances,
        raw_scores=tuple(scores),
    )


# --- pairwise + A/B swap ---


@dataclass(frozen=True)
class PairwiseResult:
    """Outcome of a bias-controlled A/B comparison."""

    winner: Literal["a", "b", "tie", "biased"]
    raw_outcomes: tuple[Literal["a", "b", "tie"], ...]
    position_bias_detected: bool


def bias_controlled_pairwise(
    a_text: str,
    b_text: str,
    *,
    judge: PairwiseLLMJudge,
    rubric: Rubric,
    seeds: Sequence[int] | None = None,
    n_seeds: int = DEFAULT_N_SEEDS,
) -> PairwiseResult:
    """A/B compare two outputs with position swap + multi-seed.

    For each seed the judge sees both orderings:
      - forward:  compare(a, b)
      - reversed: compare(b, a)  — the winner returned is in the
        reversed frame and is mapped back to {a, b}.

    A consistent winner across both orderings on a given seed means
    that seed was unbiased. A flip means position bias for that seed.
    The aggregate `winner` is the majority of unbiased outcomes; if
    more than half the seeds show position bias the result is
    `"biased"` and the caller must escalate.
    """
    chosen_seeds = list(seeds) if seeds is not None else list(range(n_seeds))
    if not chosen_seeds:
        raise ValueError("need at least one seed")
    per_seed: list[Literal["a", "b", "tie"]] = []
    biased_seeds = 0
    for seed in chosen_seeds:
        forward = judge.compare(a_text, b_text, rubric=rubric, seed=seed)
        reverse = judge.compare(b_text, a_text, rubric=rubric, seed=seed)
        unswapped = _unswap(reverse)
        if forward == unswapped:
            per_seed.append(forward)
        else:
            biased_seeds += 1
            per_seed.append("tie")
    bias_majority = biased_seeds > len(chosen_seeds) / 2
    if bias_majority:
        winner: Literal["a", "b", "tie", "biased"] = "biased"
    else:
        counts = Counter(per_seed)
        top, top_count = counts.most_common(1)[0]
        winner = top if top_count > len(chosen_seeds) / 2 else "tie"
    return PairwiseResult(
        winner=winner,
        raw_outcomes=tuple(per_seed),
        position_bias_detected=biased_seeds > 0,
    )


def _unswap(outcome: Literal["a", "b", "tie"]) -> Literal["a", "b", "tie"]:
    """Map a winner labelled in the reversed frame back to forward labels."""
    if outcome == "tie":
        return "tie"
    return "b" if outcome == "a" else "a"


# --- stability + inter-rater + drift ---


def flag_unstable(
    aggregated: AggregatedScore,
    *,
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
) -> list[str]:
    """Names of dimensions whose variance exceeds the stability threshold."""
    return [name for name, var in aggregated.per_dimension_variance.items() if var > threshold]


def cohens_kappa(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Cohen's κ between two raters' categorical scores.

    Returns 0.0 for empty input. When chance agreement is already
    perfect (a degenerate rater that always picks the same label and
    both raters agree), returns 1.0 so the test of "perfect raters"
    doesn't get penalised by the chance-correction division.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"mismatched lengths: {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class InterRaterReport:
    """Result of comparing judge scores against the human-scored overlap."""

    sample_size: int
    kappa: float
    passed: bool
    threshold: float


def inter_rater_check(
    judge_scores: Sequence[int],
    human_scores: Sequence[int],
    *,
    min_kappa: float = DEFAULT_KAPPA_FLOOR,
) -> InterRaterReport:
    """Run a Cohen's κ check on the judge vs human overlap.

    The caller is responsible for selecting the 10% sample (the
    `judge_scores` / `human_scores` lists are expected to already be
    the matched overlap). The function returns the κ value and a
    pass/fail against `min_kappa`.
    """
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge and human score lists must have equal length")
    kappa = cohens_kappa(judge_scores, human_scores)
    return InterRaterReport(
        sample_size=len(judge_scores),
        kappa=kappa,
        passed=kappa >= min_kappa,
        threshold=min_kappa,
    )


@dataclass(frozen=True)
class DriftReport:
    """Difference between a baseline and a current run of anchor outputs."""

    per_dimension_shift: dict[str, float]
    max_shift: float
    flagged_dimensions: list[str]
    threshold: float

    @property
    def flagged(self) -> bool:
        return bool(self.flagged_dimensions)


def flag_judge_drift(
    baseline: AggregatedScore,
    current: AggregatedScore,
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> DriftReport:
    """Compare per-dimension means against a stored anchor.

    Dimensions whose absolute mean shift exceeds `threshold` are
    flagged. Used on §8.7 anchor outputs scored on every commit:
    a sustained >0.5 shift on any dimension is an alert.
    """
    if baseline.rubric_name != current.rubric_name:
        raise ValueError(
            f"rubric mismatch: baseline={baseline.rubric_name!r}, current={current.rubric_name!r}"
        )
    shifts: dict[str, float] = {}
    flagged: list[str] = []
    for name, baseline_mean in baseline.per_dimension_mean.items():
        if name not in current.per_dimension_mean:
            raise ValueError(f"current run missing dimension {name!r}")
        shift = current.per_dimension_mean[name] - baseline_mean
        shifts[name] = shift
        if abs(shift) > threshold:
            flagged.append(name)
    max_shift = max((abs(s) for s in shifts.values()), default=0.0)
    return DriftReport(
        per_dimension_shift=shifts,
        max_shift=max_shift,
        flagged_dimensions=flagged,
        threshold=threshold,
    )


# --- blind-shuffle helper ---


def blind_shuffle(outputs: Sequence[str], *, seed: int) -> list[tuple[int, str]]:
    """Return `outputs` in a deterministic shuffled order tagged with
    their original index.

    Per §8.7 the judge should see outputs in a blinded order so it
    can't bias toward "the first one" or "the third one." The returned
    `(original_index, text)` pairs let the caller un-blind after
    scoring without leaking position to the judge.
    """
    rng = random.Random(seed)
    indexed = list(enumerate(outputs))
    rng.shuffle(indexed)
    return indexed


__all__ = [
    "DEFAULT_DRIFT_THRESHOLD",
    "DEFAULT_HUMAN_SAMPLE_PCT",
    "DEFAULT_KAPPA_FLOOR",
    "DEFAULT_N_SEEDS",
    "DEFAULT_STABILITY_THRESHOLD",
    "SCALE_MAX",
    "SCALE_MIN",
    "AggregatedScore",
    "DriftReport",
    "InterRaterReport",
    "JudgeScore",
    "LLMJudge",
    "PairwiseLLMJudge",
    "PairwiseResult",
    "Rubric",
    "RubricDimension",
    "bias_controlled_pairwise",
    "bias_controlled_score",
    "blind_shuffle",
    "cohens_kappa",
    "flag_judge_drift",
    "flag_unstable",
    "inter_rater_check",
    "validate_score_against_rubric",
]
