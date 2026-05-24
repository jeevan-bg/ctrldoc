"""Paraphrase voting — agreement-rate correlates with correctness rho >= 0.5.

Per SPEC §6.5, calibration is a two-step pipeline: paraphrase voting
followed by isotonic regression. `ParaphraseVoter` lands the first
half — it runs the underlying `NLIScorer` on 3-5 paraphrases of the
same hypothesis, aggregates the per-paraphrase predictions into a
majority-vote label and an agreement-rate signal, and exposes the
agreement rate as a confidence proxy. The mapping from raw score to
a calibrated probability (isotonic regression) is layered on top by
the §6.5 calibration fitter.

The §6.5 acceptance contract is the **correlation gate**: across a
labelled batch of premise/hypothesis pairs, the agreement rate
produced by paraphrase voting must rank-correlate with binary
correctness at Spearman rho >= 0.5. Intuition: pairs where every
paraphrase agrees should mostly be correct; pairs where paraphrases
disagree should be the hard cases where the scorer is more likely
wrong. If the correlation collapses, the agreement rate is not a
useful confidence proxy and the contract has regressed.

The voter holds no per-call state — it is safe to share across
threads as long as the underlying scorer and paraphraser are.

SPEC-REF: §6.5 (probabilistic edges + calibration — paraphrase voting)
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.extract.paraphrase_voting import (
    DEFAULT_NUM_PARAPHRASES,
    MAX_NUM_PARAPHRASES,
    MIN_NUM_PARAPHRASES,
    PARAPHRASE_CORRELATION_THRESHOLD,
    ParaphraseVote,
    ParaphraseVoter,
    agreement_rate,
    spearman_rank_correlation,
)

# ---------------------------------------------------------------------------
# Test doubles — deterministic scorer + paraphraser
# ---------------------------------------------------------------------------


class _DictScorer:
    """A `NLIScorer` keyed on `(premise, hypothesis)`.

    Missing keys default to a neutral-dominated score so unseen
    paraphrase pairs neither agree nor disagree by accident.
    """

    def __init__(self, table: dict[tuple[str, str], NLIScore]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        if (premise, hypothesis) in self._table:
            return self._table[(premise, hypothesis)]
        return NLIScore(entailment=0.20, contradiction=0.20, neutral=0.60)


class _ListParaphraser:
    """A `Paraphraser` that returns a fixed list of paraphrases per input."""

    def __init__(self, table: dict[str, list[str]]) -> None:
        self._table = table
        self.calls: list[tuple[str, int]] = []

    def paraphrase(self, text: str, *, k: int) -> list[str]:
        self.calls.append((text, k))
        # Default: echo the original `k` times so the voter degenerates
        # gracefully when the table has no entry for `text`.
        items = self._table.get(text, [text] * k)
        return items[:k]


# ---------------------------------------------------------------------------
# Public surface — constants
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_paraphrase_count_defaults_lie_in_spec_band() -> None:
    """§6.5 mandates 3-5 paraphrases; defaults must obey that band."""
    assert MIN_NUM_PARAPHRASES == 3
    assert MAX_NUM_PARAPHRASES == 5
    assert MIN_NUM_PARAPHRASES <= DEFAULT_NUM_PARAPHRASES <= MAX_NUM_PARAPHRASES


@pytest.mark.family_determinism
def test_correlation_threshold_matches_spec_gate() -> None:
    """The §6.5 rho >= 0.5 acceptance gate is the public threshold."""
    assert PARAPHRASE_CORRELATION_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Voter contract — agreement, majority label, scorer-call budget
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_unanimous_agreement_yields_agreement_rate_one() -> None:
    """All paraphrases label the same way → agreement rate = 1.0."""
    premise = "the system uses tls 1.3"
    hypothesis = "the system uses tls"
    paras = ["the system uses tls", "tls is used by the system", "the system runs tls"]
    scorer = _DictScorer(
        {
            (premise, paras[0]): NLIScore(entailment=0.92, contradiction=0.02, neutral=0.06),
            (premise, paras[1]): NLIScore(entailment=0.88, contradiction=0.03, neutral=0.09),
            (premise, paras[2]): NLIScore(entailment=0.90, contradiction=0.02, neutral=0.08),
        }
    )
    paraphraser = _ListParaphraser({hypothesis: paras})
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=3)

    vote = voter.vote(premise=premise, hypothesis=hypothesis)

    assert isinstance(vote, ParaphraseVote)
    assert vote.majority_label == "entailment"
    assert vote.agreement_rate == pytest.approx(1.0)
    assert vote.num_paraphrases == 3
    assert vote.label_votes == {"entailment": 3, "contradiction": 0, "neutral": 0}


@pytest.mark.family_verifier_calibration
def test_split_agreement_yields_fractional_agreement_rate() -> None:
    """2 entail / 1 neutral → majority = entail, agreement = 2/3."""
    premise = "the cache is warm"
    hypothesis = "the cache holds entries"
    paras = ["the cache stores entries", "the cache is populated", "the cache may be empty"]
    scorer = _DictScorer(
        {
            (premise, paras[0]): NLIScore(entailment=0.85, contradiction=0.05, neutral=0.10),
            (premise, paras[1]): NLIScore(entailment=0.80, contradiction=0.05, neutral=0.15),
            (premise, paras[2]): NLIScore(entailment=0.10, contradiction=0.10, neutral=0.80),
        }
    )
    paraphraser = _ListParaphraser({hypothesis: paras})
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=3)

    vote = voter.vote(premise=premise, hypothesis=hypothesis)

    assert vote.majority_label == "entailment"
    assert vote.agreement_rate == pytest.approx(2.0 / 3.0)
    assert vote.label_votes == {"entailment": 2, "contradiction": 0, "neutral": 1}


@pytest.mark.family_verifier_calibration
def test_majority_label_mean_confidence_only_aggregates_majority_paraphrases() -> None:
    """Mean confidence reports the average score on the winning label."""
    premise = "the worker retries on errors"
    hypothesis = "the worker retries"
    paras = ["the worker retries", "the worker retries failed jobs", "the worker recovers"]
    scorer = _DictScorer(
        {
            (premise, paras[0]): NLIScore(entailment=0.90, contradiction=0.05, neutral=0.05),
            (premise, paras[1]): NLIScore(entailment=0.80, contradiction=0.05, neutral=0.15),
            (premise, paras[2]): NLIScore(entailment=0.10, contradiction=0.10, neutral=0.80),
        }
    )
    paraphraser = _ListParaphraser({hypothesis: paras})
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=3)

    vote = voter.vote(premise=premise, hypothesis=hypothesis)

    assert vote.majority_label == "entailment"
    # Mean entailment of the two entail-voting paraphrases: (0.90 + 0.80) / 2
    assert vote.mean_top_confidence == pytest.approx(0.85)


@pytest.mark.family_verifier_calibration
def test_unanimous_disagreement_with_base_returns_dissenting_label() -> None:
    """All paraphrases vote contradiction → majority label = contradiction."""
    premise = "the cache is warm"
    hypothesis = "the cache is cold"
    paras = ["the cache is freezing", "the cache is cold", "the cache contains nothing"]
    scorer = _DictScorer(
        {
            (premise, paras[0]): NLIScore(entailment=0.05, contradiction=0.90, neutral=0.05),
            (premise, paras[1]): NLIScore(entailment=0.04, contradiction=0.92, neutral=0.04),
            (premise, paras[2]): NLIScore(entailment=0.05, contradiction=0.85, neutral=0.10),
        }
    )
    paraphraser = _ListParaphraser({hypothesis: paras})
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=3)

    vote = voter.vote(premise=premise, hypothesis=hypothesis)

    assert vote.majority_label == "contradiction"
    assert vote.agreement_rate == pytest.approx(1.0)


@pytest.mark.family_performance_cost
def test_scorer_called_exactly_once_per_paraphrase() -> None:
    """The voter issues exactly `num_paraphrases` scorer calls per vote."""
    premise = "alpha is fast"
    hypothesis = "alpha is quick"
    paras = ["alpha is quick", "alpha runs quickly", "alpha is rapid", "alpha is speedy"]
    scorer = _DictScorer({})
    paraphraser = _ListParaphraser({hypothesis: paras})
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=4)

    voter.vote(premise=premise, hypothesis=hypothesis)

    assert len(scorer.calls) == 4


@pytest.mark.family_determinism
def test_num_paraphrases_outside_spec_band_rejected() -> None:
    """`num_paraphrases` outside the §6.5 [3, 5] band raises at construction."""
    scorer = _DictScorer({})
    paraphraser = _ListParaphraser({})

    with pytest.raises(ValueError, match="num_paraphrases"):
        ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=2)
    with pytest.raises(ValueError, match="num_paraphrases"):
        ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=6)


@pytest.mark.family_determinism
def test_paraphraser_returning_too_few_paraphrases_raises() -> None:
    """A paraphraser that returns < k paraphrases is a contract violation."""
    premise = "the system uses tls"
    hypothesis = "tls is used"
    # Paraphraser only ships 2 paraphrases but the voter asked for 3.
    paraphraser = _ListParaphraser({hypothesis: ["tls is used", "the system uses tls"]})
    scorer = _DictScorer({})
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=3)

    with pytest.raises(ValueError, match="paraphraser"):
        voter.vote(premise=premise, hypothesis=hypothesis)


# ---------------------------------------------------------------------------
# `agreement_rate` helper
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_agreement_rate_all_same_label() -> None:
    """All same label → 1.0."""
    assert agreement_rate(["entailment", "entailment", "entailment"]) == pytest.approx(1.0)


@pytest.mark.family_determinism
def test_agreement_rate_majority_two_thirds() -> None:
    """2 / 3 share the same label → 2/3."""
    assert agreement_rate(["entailment", "entailment", "neutral"]) == pytest.approx(2.0 / 3.0)


@pytest.mark.family_determinism
def test_agreement_rate_three_way_split() -> None:
    """1/3 each on three different labels → 1/3."""
    assert agreement_rate(["entailment", "contradiction", "neutral"]) == pytest.approx(1.0 / 3.0)


@pytest.mark.family_determinism
def test_agreement_rate_empty_list_raises() -> None:
    """An empty vote list is undefined — caller must guard."""
    with pytest.raises(ValueError, match="empty"):
        agreement_rate([])


# ---------------------------------------------------------------------------
# Spearman rank correlation — helper for the headline gate
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_spearman_perfect_positive_correlation() -> None:
    """Monotonically increasing pairs → rho = +1."""
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert spearman_rank_correlation(xs, ys) == pytest.approx(1.0)


@pytest.mark.family_determinism
def test_spearman_perfect_negative_correlation() -> None:
    """Monotonically opposing pairs → rho = -1."""
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert spearman_rank_correlation(xs, ys) == pytest.approx(-1.0)


@pytest.mark.family_determinism
def test_spearman_handles_ties_via_average_ranks() -> None:
    """Repeated values use average ranks so rho is well-defined on ties."""
    xs = [1.0, 1.0, 2.0, 3.0]
    ys = [1.0, 1.0, 2.0, 3.0]
    # Identical sequences (even with ties) → rho = 1.
    assert spearman_rank_correlation(xs, ys) == pytest.approx(1.0)


@pytest.mark.family_determinism
def test_spearman_length_mismatch_raises() -> None:
    """Length mismatch is always a caller bug."""
    with pytest.raises(ValueError, match="length"):
        spearman_rank_correlation([1.0, 2.0], [1.0])


@pytest.mark.family_determinism
def test_spearman_too_short_input_raises() -> None:
    """Fewer than 2 pairs is undefined."""
    with pytest.raises(ValueError, match="at least"):
        spearman_rank_correlation([1.0], [1.0])


# ---------------------------------------------------------------------------
# Headline contract — agreement rate correlates with correctness rho >= 0.5
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_agreement_rate_correlates_with_correctness() -> None:
    """The §6.5 rho >= 0.5 gate on agreement-rate vs binary correctness.

    Drive 10 labelled pairs through the voter on a deterministic
    paraphraser/scorer where the scorer's correctness on each case is
    well-controlled. Confident-correct cases get unanimous agreement;
    hard wrong cases get split votes; the resulting agreement rates
    must rank-correlate with the binary correctness vector at
    Spearman rho >= 0.5.
    """
    # Each case: (id, premise, hypothesis, paraphrase_scores, gold_label)
    # `paraphrase_scores` is a list of NLIScore — one per paraphrase.
    cases: list[tuple[str, str, str, list[NLIScore], str]] = [
        # Confident, correct entailment — unanimous → high agreement, correct.
        (
            "c1",
            "the system uses tls 1.3",
            "the system uses tls",
            [
                NLIScore(entailment=0.92, contradiction=0.02, neutral=0.06),
                NLIScore(entailment=0.88, contradiction=0.04, neutral=0.08),
                NLIScore(entailment=0.90, contradiction=0.03, neutral=0.07),
            ],
            "entailment",
        ),
        # Confident, correct contradiction — unanimous → high agreement, correct.
        (
            "c2",
            "the cache is warm",
            "the cache is cold",
            [
                NLIScore(entailment=0.04, contradiction=0.90, neutral=0.06),
                NLIScore(entailment=0.03, contradiction=0.92, neutral=0.05),
                NLIScore(entailment=0.05, contradiction=0.88, neutral=0.07),
            ],
            "contradiction",
        ),
        # Confident, correct neutral — unanimous → high agreement, correct.
        (
            "c3",
            "alice wrote the rfc",
            "bob reviewed the rfc",
            [
                NLIScore(entailment=0.08, contradiction=0.06, neutral=0.86),
                NLIScore(entailment=0.10, contradiction=0.05, neutral=0.85),
                NLIScore(entailment=0.06, contradiction=0.06, neutral=0.88),
            ],
            "neutral",
        ),
        # Confident, correct entailment — unanimous → high agreement, correct.
        (
            "c4",
            "the worker retries on errors",
            "the worker retries",
            [
                NLIScore(entailment=0.95, contradiction=0.02, neutral=0.03),
                NLIScore(entailment=0.93, contradiction=0.02, neutral=0.05),
                NLIScore(entailment=0.91, contradiction=0.03, neutral=0.06),
            ],
            "entailment",
        ),
        # Confident, correct entailment — unanimous → high agreement, correct.
        (
            "c5",
            "the proxy forwards every request",
            "the proxy forwards requests",
            [
                NLIScore(entailment=0.89, contradiction=0.03, neutral=0.08),
                NLIScore(entailment=0.86, contradiction=0.04, neutral=0.10),
                NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07),
            ],
            "entailment",
        ),
        # Hard, wrong — paraphrases disagree (1 entail / 1 contra / 1 neutral)
        # and the majority happens to be wrong → low agreement, incorrect.
        (
            "h1",
            "the leader is elected within five seconds",
            "the leader is elected eventually",
            [
                NLIScore(entailment=0.60, contradiction=0.10, neutral=0.30),
                NLIScore(entailment=0.15, contradiction=0.55, neutral=0.30),
                NLIScore(entailment=0.20, contradiction=0.15, neutral=0.65),
            ],
            "neutral",  # gold = neutral; majority by argmax = entailment (wrong)
        ),
        # Hard, wrong — split entail/contra, majority contradiction, wrong.
        (
            "h2",
            "the consensus protocol uses paxos",
            "the consensus protocol uses raft",
            [
                NLIScore(entailment=0.40, contradiction=0.55, neutral=0.05),
                NLIScore(entailment=0.45, contradiction=0.50, neutral=0.05),
                NLIScore(entailment=0.60, contradiction=0.30, neutral=0.10),
            ],
            "entailment",  # gold = entailment; majority by argmax = contradiction
        ),
        # Hard, wrong — three-way split, majority entailment wrong.
        (
            "h3",
            "the worker spawns threads",
            "the worker spawns processes",
            [
                NLIScore(entailment=0.55, contradiction=0.30, neutral=0.15),
                NLIScore(entailment=0.30, contradiction=0.60, neutral=0.10),
                NLIScore(entailment=0.20, contradiction=0.15, neutral=0.65),
            ],
            "contradiction",  # gold = contradiction; majority = entailment
        ),
        # Hard, wrong — split, majority wrong.
        (
            "h4",
            "the store flushes on commit",
            "the store flushes on read",
            [
                NLIScore(entailment=0.50, contradiction=0.45, neutral=0.05),
                NLIScore(entailment=0.55, contradiction=0.40, neutral=0.05),
                NLIScore(entailment=0.25, contradiction=0.30, neutral=0.45),
            ],
            "contradiction",  # gold = contradiction; majority = entailment
        ),
        # Hard, wrong — split, majority wrong.
        (
            "h5",
            "the index supports range queries",
            "the index supports point lookups",
            [
                NLIScore(entailment=0.55, contradiction=0.40, neutral=0.05),
                NLIScore(entailment=0.50, contradiction=0.30, neutral=0.20),
                NLIScore(entailment=0.30, contradiction=0.55, neutral=0.15),
            ],
            "neutral",  # gold = neutral; majority = entailment
        ),
    ]

    scorer_table: dict[tuple[str, str], NLIScore] = {}
    paraphrase_table: dict[str, list[str]] = {}
    for _, premise, hypothesis, paras, _ in cases:
        ph_keys = [f"{hypothesis} :: para{i}" for i in range(len(paras))]
        paraphrase_table[hypothesis] = ph_keys
        for ph_key, ph_score in zip(ph_keys, paras, strict=True):
            scorer_table[(premise, ph_key)] = ph_score

    scorer = _DictScorer(scorer_table)
    paraphraser = _ListParaphraser(paraphrase_table)
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=3)

    agreements: list[float] = []
    correctness: list[float] = []
    for _, premise, hypothesis, _, gold in cases:
        vote = voter.vote(premise=premise, hypothesis=hypothesis)
        agreements.append(vote.agreement_rate)
        correctness.append(1.0 if vote.majority_label == gold else 0.0)

    rho = spearman_rank_correlation(agreements, correctness)

    # Both the helper threshold and the SPEC §6.5 gate.
    assert rho >= PARAPHRASE_CORRELATION_THRESHOLD


# ---------------------------------------------------------------------------
# Determinism — identical inputs produce identical outputs
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_repeated_runs_produce_identical_output() -> None:
    """The voter is deterministic given deterministic dependencies."""
    premise = "the system uses tls 1.3"
    hypothesis = "tls is used"
    paras = ["tls is used", "tls is enabled", "the system has tls"]
    scorer = _DictScorer(
        {
            (premise, paras[0]): NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07),
            (premise, paras[1]): NLIScore(entailment=0.85, contradiction=0.05, neutral=0.10),
            (premise, paras[2]): NLIScore(entailment=0.80, contradiction=0.05, neutral=0.15),
        }
    )
    paraphraser = _ListParaphraser({hypothesis: paras})
    voter = ParaphraseVoter(scorer=scorer, paraphraser=paraphraser, num_paraphrases=3)

    a = voter.vote(premise=premise, hypothesis=hypothesis)
    b = voter.vote(premise=premise, hypothesis=hypothesis)

    assert a.model_dump() == b.model_dump()
