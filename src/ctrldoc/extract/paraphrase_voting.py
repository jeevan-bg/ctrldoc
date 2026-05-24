"""Paraphrase voting — agreement-rate as a confidence proxy for §6.5.

Per SPEC §6.5, the calibration pipeline is two-step:

> Calibration: paraphrase voting (run NLI / judge on 3-5 paraphrases of
> the claim; agreement → cheap high-confidence; disagreement → escalate)
> plus isotonic regression fitted on the eval set, mapping raw scores
> → calibrated probabilities.

This module ships the first half. A `ParaphraseVoter` consumes an
`NLIScorer` (the same 3-way protocol used by the Tier-2 NLI edge
inferer and the calibration eval substrate) plus a `Paraphraser`
backend, and on every `vote(premise, hypothesis)` call:

1. Asks the paraphraser for `num_paraphrases` re-wordings of the
   hypothesis. Per §6.5 the band is `[3, 5]`; the voter enforces the
   bound at construction.
2. Scores each `(premise, paraphrase)` pair under the underlying NLI
   scorer exactly once — cost is `O(num_paraphrases)` per anchor.
3. Aggregates the per-paraphrase argmax labels into a `ParaphraseVote`
   carrying the majority label, per-label vote counts, the
   agreement rate (fraction of paraphrases that voted for the
   majority label), and the mean top-label confidence across only the
   majority-voting paraphrases.

The agreement rate is the confidence proxy the upstream isotonic
regression layer (S-137) consumes — paraphrase-voting alone does not
ship a calibrated probability, only a signal that empirically
correlates with correctness. The §6.5 acceptance contract for this
slice is the **correlation gate**: across a labelled batch, Spearman
rank correlation between agreement rate and binary correctness must
clear `PARAPHRASE_CORRELATION_THRESHOLD = 0.5`. The Spearman helper
lives in this module so callers (release-gate eval scripts in
particular) do not have to reach for scipy.

The voter holds no per-call state — it is safe to share across
threads as long as the underlying scorer and paraphraser are.

SPEC-REF: §6.5 (probabilistic edges + calibration — paraphrase voting)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.eval.calibration import CALIBRATION_LABELS, CalibrationLabel, NLIScore

# ---------------------------------------------------------------------------
# Public constants — bind to the §6.5 spec band and acceptance gate
# ---------------------------------------------------------------------------


MIN_NUM_PARAPHRASES: int = 3
"""§6.5 floor on the paraphrase count."""

MAX_NUM_PARAPHRASES: int = 5
"""§6.5 ceiling on the paraphrase count."""

DEFAULT_NUM_PARAPHRASES: int = 3
"""Cheap default inside the §6.5 [3, 5] band."""

PARAPHRASE_CORRELATION_THRESHOLD: float = 0.5
"""§6.5 acceptance gate on Spearman rank correlation between agreement and correctness."""


# ---------------------------------------------------------------------------
# Backend protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class NLIScorer(Protocol):
    """3-way NLI scorer — same shape as `eval.calibration.CalibrationScorer`."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


@runtime_checkable
class Paraphraser(Protocol):
    """Given a hypothesis and a count `k`, return `k` paraphrases.

    Implementations are expected to be deterministic across repeat
    calls with the same input — the calibration pipeline is replayable
    per §13 non-negotiable 4 and a stochastic paraphraser would break
    that. Production backends pin the seed; tests use a fixed table.
    """

    def paraphrase(self, text: str, *, k: int) -> list[str]: ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class ParaphraseVote(BaseModel):
    """Aggregate output of one `ParaphraseVoter.vote` call.

    `majority_label` is the label that received the plurality of votes
    across the paraphrases. Ties on plurality fall through the standard
    `Counter.most_common` ordering — production callers should treat
    the agreement rate as the signal, not the label, when ties occur.

    `agreement_rate` is the fraction of paraphrases that voted for the
    majority label, in `[1 / num_paraphrases, 1.0]`. A unanimous vote
    yields `1.0`; a three-way split with `num_paraphrases = 3` yields
    `1 / 3`.

    `mean_top_confidence` is the average top-label probability across
    only the majority-voting paraphrases — dissenting paraphrases do
    not pull the confidence down toward the wrong label. This is the
    raw score the §6.5 isotonic regression layer consumes alongside
    the agreement rate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    majority_label: CalibrationLabel
    agreement_rate: float = Field(ge=0.0, le=1.0)
    mean_top_confidence: float = Field(ge=0.0, le=1.0)
    num_paraphrases: int = Field(ge=MIN_NUM_PARAPHRASES, le=MAX_NUM_PARAPHRASES)
    label_votes: dict[CalibrationLabel, int]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class ParaphraseVoter:
    """Aggregate per-paraphrase NLI verdicts into a single voted result.

    Construction validates the `num_paraphrases` band so the voter
    cannot escape the §6.5 [3, 5] envelope at runtime. The scorer and
    paraphraser are stored by reference and never mutated.
    """

    def __init__(
        self,
        *,
        scorer: NLIScorer,
        paraphraser: Paraphraser,
        num_paraphrases: int = DEFAULT_NUM_PARAPHRASES,
    ) -> None:
        if num_paraphrases < MIN_NUM_PARAPHRASES or num_paraphrases > MAX_NUM_PARAPHRASES:
            raise ValueError(
                f"num_paraphrases must lie in [{MIN_NUM_PARAPHRASES}, "
                f"{MAX_NUM_PARAPHRASES}] per SPEC §6.5 (got {num_paraphrases})"
            )
        self._scorer = scorer
        self._paraphraser = paraphraser
        self._num_paraphrases = num_paraphrases

    def vote(self, *, premise: str, hypothesis: str) -> ParaphraseVote:
        """Score the premise against `num_paraphrases` paraphrases of the hypothesis.

        Returns a `ParaphraseVote` aggregating the per-paraphrase
        argmax labels, agreement rate, and mean top-label confidence
        over the majority-voting paraphrases.
        """
        paraphrases = self._paraphraser.paraphrase(hypothesis, k=self._num_paraphrases)
        if len(paraphrases) < self._num_paraphrases:
            raise ValueError(
                f"paraphraser returned {len(paraphrases)} paraphrases; "
                f"voter requires {self._num_paraphrases}"
            )

        scores = [
            self._scorer.score(premise=premise, hypothesis=ph)
            for ph in paraphrases[: self._num_paraphrases]
        ]
        labels: list[CalibrationLabel] = [s.argmax_label() for s in scores]

        # Build the vote tally. Pre-populate all three labels so the
        # surfaced dict has a stable, complete shape regardless of which
        # labels were actually voted.
        label_votes: dict[CalibrationLabel, int] = dict.fromkeys(CALIBRATION_LABELS, 0)
        for label in labels:
            label_votes[label] += 1

        # Counter.most_common breaks ties by insertion order, which is
        # itself stable per CPython's dict ordering — so the majority
        # is deterministic when callers pin the paraphraser seed.
        majority_label = Counter(labels).most_common(1)[0][0]

        majority_count = label_votes[majority_label]
        agreement_rate_value = majority_count / self._num_paraphrases

        majority_top_confidences = [
            score.top_confidence()
            for score, label in zip(scores, labels, strict=True)
            if label == majority_label
        ]
        mean_top_confidence = sum(majority_top_confidences) / len(majority_top_confidences)

        return ParaphraseVote(
            majority_label=majority_label,
            agreement_rate=agreement_rate_value,
            mean_top_confidence=mean_top_confidence,
            num_paraphrases=self._num_paraphrases,
            label_votes=label_votes,
        )


# ---------------------------------------------------------------------------
# Helpers — agreement rate + Spearman rank correlation
# ---------------------------------------------------------------------------


def agreement_rate(labels: Sequence[CalibrationLabel]) -> float:
    """Fraction of labels that match the plurality label.

    Empty input is undefined and raises — agreement on zero votes has
    no meaning and the caller is expected to guard before calling.
    """
    if not labels:
        raise ValueError("agreement_rate is undefined on an empty label list")
    counts = Counter(labels)
    majority = counts.most_common(1)[0][1]
    return majority / len(labels)


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation via Pearson on the average-rank vectors.

    Handles ties via the standard average-rank convention (a tied pair
    at positions i, j receives rank `(rank_i + rank_j) / 2`). Returns
    `0.0` when either ranked sequence is constant — a constant
    sequence has zero variance and Pearson correlation is undefined;
    returning zero is the conservative reading ("no monotonic
    relationship").

    The implementation is stdlib-only by design: scipy is not a
    project dependency and adding it for one function would violate
    the "fewer moving parts" rule in WAYS_OF_WORKING.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} xs vs {len(ys)} ys")
    if len(xs) < 2:
        raise ValueError("spearman_rank_correlation requires at least 2 paired observations")

    ranks_x = _average_ranks(xs)
    ranks_y = _average_ranks(ys)

    n = len(xs)
    mean_x = sum(ranks_x) / n
    mean_y = sum(ranks_y) / n

    cov = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(ranks_x, ranks_y, strict=True))
    var_x = sum((rx - mean_x) ** 2 for rx in ranks_x)
    var_y = sum((ry - mean_y) ** 2 for ry in ranks_y)

    denom: float = (var_x * var_y) ** 0.5
    if denom == 0:
        # Constant sequence — Pearson correlation undefined; return 0 conservatively.
        return 0.0
    return float(cov / denom)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return average-rank vector for `values` (1-indexed, ties averaged).

    Example: `[10, 20, 20, 30]` → `[1.0, 2.5, 2.5, 4.0]`.
    """
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks: list[float] = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        # Walk through the tied block sharing the same value.
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # Ranks are 1-indexed; average rank over the tie block is the
        # midpoint of the inclusive (i+1, j+1) integer range.
        avg_rank = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1
    return ranks


__all__ = [
    "DEFAULT_NUM_PARAPHRASES",
    "MAX_NUM_PARAPHRASES",
    "MIN_NUM_PARAPHRASES",
    "PARAPHRASE_CORRELATION_THRESHOLD",
    "NLIScorer",
    "ParaphraseVote",
    "ParaphraseVoter",
    "Paraphraser",
    "agreement_rate",
    "spearman_rank_correlation",
]
