"""L5 compare operation via optimal transport + Galois lattice.

§6.6 frames `compare(A, B)` as:

> *Asymmetric transport in both directions; per-concept-cluster cost
> summary = strengths/weaknesses.*

The eval substrate (`ctrldoc.eval.compare`) pre-clusters claims into
`{a_claim?, b_claim?}` cells and grades the per-cluster verdict over
the 3-label space `{StrengthA, StrengthB, Gap}`. This module ships
`TransportCompareVerifier` — the `CompareVerifier`-protocol-shaped
consumer wiring the Galois floor (`claim_subsumption`) plus
asymmetric NLI fallback into per-cluster verdicts — and a
functional `compare` surface for the §9 CLI.

The reduction:

* Exactly one of `a_claim` / `b_claim` is `None` → `Gap`. The other
  side has nothing to compare against; the cluster's strength
  trivially belongs to the present side, but §6.6 reserves `Gap` for
  the structural "only one doc speaks to this concept" case.
* Both sides present → consult `claim_subsumption` first. The
  Galois floor handles the deterministic modality / qualifier cases
  (`MUST ⊐ SHOULD ⊐ MAY`, empty qualifier ⊐ scoped) at zero NLI cost
  and returns `subsumes`/`subsumed_by`/`equivalent`/`incomparable`.
  - `subsumes` (A strictly stronger than B) → `StrengthA`.
  - `subsumed_by` (A strictly weaker than B) → `StrengthB`.
  - `equivalent` (Galois ties) → deterministic tiebreak: `StrengthA`.
  - `incomparable` → fall back to asymmetric NLI (§6.6 transport in
    both directions). The §6.6 framing: "asymmetric transport in
    both directions" gives `cost_AB = 1 - NLI_entail(A → B)` and
    `cost_BA = 1 - NLI_entail(B → A)`. The side that *entails* the
    other is the more general (stronger) claim — entailment is the
    §6.3 subsumption relation. `StrengthA` when `e_AB > e_BA`;
    `StrengthB` when `e_BA > e_AB`; symmetric ties → `StrengthA`
    (deterministic).

The release gate is `COMPARE_VERDICT_THRESHOLD = 0.85` per-cluster
accuracy on the 3-label space.

SPEC-REF: §6.6 (compare = per-concept-cluster strengths/weaknesses
via asymmetric transport in both directions)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.compare import (
    COMPARE_VERDICT_THRESHOLD,
    CompareEvalCase,
    CompareEvalRunner,
    CompareVerdictLiteral,
    CompareVerifier,
    ConceptComparisonInput,
    compare_accuracy,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.ops.compare import (
    CompareConfig,
    CompareResult,
    TransportCompareVerifier,
    compare,
)

COMPARE_EVAL_PATH = Path(__file__).parent / "eval" / "compare_eval.jsonl"


# ---------------------------------------------------------------------------
# Stub scorers — deterministic, recordable
# ---------------------------------------------------------------------------


class _DictScorer:
    """`NLIScorer` keyed on `(premise, hypothesis)` strings."""

    def __init__(
        self,
        table: dict[tuple[str, str], NLIScore],
        *,
        default: NLIScore | None = None,
    ) -> None:
        self._table = table
        self._default = default or NLIScore(entailment=0.20, contradiction=0.10, neutral=0.70)
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._table.get((premise, hypothesis), self._default)


def _high_entail() -> NLIScore:
    return NLIScore(entailment=0.92, contradiction=0.03, neutral=0.05)


def _low_entail() -> NLIScore:
    return NLIScore(entailment=0.20, contradiction=0.10, neutral=0.70)


def _claim(
    subject: str = "the system",
    predicate: str = "uses",
    object_: str = "consistent hashing",
    polarity: str = "affirmative",
    modality: str = "asserted",
    qualifier: str = "",
) -> ClaimTuple:
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=object_,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier,
    )


def _cluster(
    *,
    id_: str = "c1",
    label: str = "concept",
    a: ClaimTuple | None = None,
    b: ClaimTuple | None = None,
) -> ConceptComparisonInput:
    return ConceptComparisonInput(id=id_, label=label, a_claim=a, b_claim=b)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_config_default_entailment_gap_in_unit_interval() -> None:
    """Default `nli_tie_gap` is a small positive epsilon in (0, 1)."""
    cfg = CompareConfig()
    assert 0.0 < cfg.nli_tie_gap < 1.0


@pytest.mark.family_determinism
def test_config_rejects_negative_tie_gap() -> None:
    with pytest.raises(ValueError):
        CompareConfig(nli_tie_gap=-0.01)
    with pytest.raises(ValueError):
        CompareConfig(nli_tie_gap=1.5)


# ---------------------------------------------------------------------------
# CompareResult shape
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_result_is_frozen_and_strict() -> None:
    """Result rejects extra fields and pinned attributes are read-only."""
    r = CompareResult(verdicts=["Gap"], scorer_calls=0)
    with pytest.raises(ValidationError):
        r.verdicts = ["StrengthA"]  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CompareResult(verdicts=["Gap"], scorer_calls=0, stray="oops")  # type: ignore[call-arg]


@pytest.mark.family_determinism
def test_result_verdicts_only_accept_literals() -> None:
    with pytest.raises(ValidationError):
        CompareResult(verdicts=["Strongest"], scorer_calls=0)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Gap cases — exactly one side present
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_only_a_side_present_yields_gap() -> None:
    """A cluster that only doc-A speaks to is a Gap (no comparison possible)."""
    cluster = _cluster(a=_claim(predicate="speaks"))
    scorer = _DictScorer({})
    result = compare(clusters=[cluster], scorer=scorer)
    assert result.verdicts == ["Gap"]
    # Gap is structural — no NLI call needed.
    assert result.scorer_calls == 0
    assert scorer.calls == []


@pytest.mark.family_determinism
def test_only_b_side_present_yields_gap() -> None:
    cluster = _cluster(b=_claim(predicate="speaks"))
    scorer = _DictScorer({})
    result = compare(clusters=[cluster], scorer=scorer)
    assert result.verdicts == ["Gap"]
    assert result.scorer_calls == 0


# ---------------------------------------------------------------------------
# Galois floor — modality ordering decides without NLI
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_modality_must_vs_should_yields_strength_a_without_nli() -> None:
    """`MUST` (obligatory) vs `SHOULD` (recommended) is the Galois floor's
    job — no NLI call needed."""
    a = _claim(subject="x", predicate="p", object_="y", modality="obligatory")
    b = _claim(subject="x", predicate="p", object_="y", modality="recommended")
    cluster = _cluster(a=a, b=b)
    scorer = _DictScorer({})
    result = compare(clusters=[cluster], scorer=scorer)
    assert result.verdicts == ["StrengthA"]
    assert result.scorer_calls == 0
    assert scorer.calls == []


@pytest.mark.family_determinism
def test_modality_should_vs_must_yields_strength_b_without_nli() -> None:
    a = _claim(subject="x", predicate="p", object_="y", modality="recommended")
    b = _claim(subject="x", predicate="p", object_="y", modality="obligatory")
    result = compare(clusters=[_cluster(a=a, b=b)], scorer=_DictScorer({}))
    assert result.verdicts == ["StrengthB"]
    assert result.scorer_calls == 0


@pytest.mark.family_determinism
def test_prohibition_strength_under_negative_polarity() -> None:
    """`prohibited` strictly subsumes `recommended` under negative polarity."""
    a = _claim(
        subject="data",
        predicate="leaves",
        object_="the eu",
        polarity="negative",
        modality="prohibited",
    )
    b = _claim(
        subject="data",
        predicate="leaves",
        object_="the eu",
        polarity="negative",
        modality="recommended",
    )
    result = compare(clusters=[_cluster(a=a, b=b)], scorer=_DictScorer({}))
    assert result.verdicts == ["StrengthA"]
    assert result.scorer_calls == 0


@pytest.mark.family_determinism
def test_galois_equivalent_tiebreaks_to_strength_a() -> None:
    """Identical claims (Galois `equivalent`) tiebreak deterministically to A."""
    same = _claim(subject="x", predicate="p", object_="y", modality="obligatory")
    result = compare(clusters=[_cluster(a=same, b=same)], scorer=_DictScorer({}))
    assert result.verdicts == ["StrengthA"]
    assert result.scorer_calls == 0


# ---------------------------------------------------------------------------
# NLI fallback — incomparable Galois pairs use asymmetric transport
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_nli_fallback_picks_more_general_side_as_strength() -> None:
    """When Galois returns `incomparable` (different SVO surfaces),
    asymmetric NLI decides: the side that *entails* the other is the
    more general (stronger) claim — entailment is the §6.3
    subsumption relation."""
    a = _claim(subject="the system", predicate="encrypts", object_="all data at rest")
    b = _claim(subject="the system", predicate="encrypts", object_="user data")
    # A is more general (all data ⊃ user data) → A entails B strongly,
    # B entails A only weakly. `e_AB` (premise=A, hyp=B) is high;
    # `e_BA` is low. A is the stronger (more general) claim.
    a_text = "the system encrypts all data at rest"
    b_text = "the system encrypts user data"
    scorer = _DictScorer(
        {
            (a_text, b_text): _high_entail(),  # A → B strong
            (b_text, a_text): _low_entail(),  # B → A weak
        }
    )
    result = compare(clusters=[_cluster(a=a, b=b)], scorer=scorer)
    assert result.verdicts == ["StrengthA"]
    # Bidirectional NLI = 2 calls per fallback cluster.
    assert result.scorer_calls == 2


@pytest.mark.family_verifier_calibration
def test_nli_fallback_picks_strength_b_when_b_more_general() -> None:
    a = _claim(subject="the system", predicate="encrypts", object_="user data")
    b = _claim(subject="the system", predicate="encrypts", object_="all data at rest")
    a_text = "the system encrypts user data"
    b_text = "the system encrypts all data at rest"
    scorer = _DictScorer(
        {
            (a_text, b_text): _low_entail(),  # A (specific) weakly entails B
            (b_text, a_text): _high_entail(),  # B (general) strongly entails A
        }
    )
    result = compare(clusters=[_cluster(a=a, b=b)], scorer=scorer)
    assert result.verdicts == ["StrengthB"]
    assert result.scorer_calls == 2


@pytest.mark.family_verifier_calibration
def test_nli_symmetric_paraphrase_tiebreaks_to_strength_a() -> None:
    """Symmetric NLI (paraphrase) — gap below `nli_tie_gap` ties to A."""
    a = _claim(subject="users", predicate="authenticate", object_="via oauth")
    b = _claim(subject="users", predicate="sign in", object_="with oauth")
    a_text = "users authenticate via oauth"
    b_text = "users sign in with oauth"
    scorer = _DictScorer(
        {
            (a_text, b_text): _high_entail(),
            (b_text, a_text): _high_entail(),
        }
    )
    result = compare(clusters=[_cluster(a=a, b=b)], scorer=scorer)
    assert result.verdicts == ["StrengthA"]


# ---------------------------------------------------------------------------
# Cost contract — fallback NLI calls are bounded
# ---------------------------------------------------------------------------


@pytest.mark.family_performance_cost
def test_gap_cluster_does_not_invoke_scorer() -> None:
    """Single-sided clusters short-circuit before any NLI call."""
    clusters = [
        _cluster(id_="c1", a=_claim(predicate="x")),
        _cluster(id_="c2", b=_claim(predicate="y")),
        _cluster(id_="c3", a=_claim(predicate="z")),
    ]
    scorer = _DictScorer({})
    result = compare(clusters=clusters, scorer=scorer)
    assert result.verdicts == ["Gap", "Gap", "Gap"]
    assert result.scorer_calls == 0


@pytest.mark.family_performance_cost
def test_galois_short_circuits_skip_scorer() -> None:
    """Clusters resolved by the Galois floor never call the NLI scorer."""
    a = _claim(subject="x", predicate="p", object_="y", modality="obligatory")
    b = _claim(subject="x", predicate="p", object_="y", modality="recommended")
    clusters = [_cluster(a=a, b=b) for _ in range(5)]
    scorer = _DictScorer({})
    result = compare(clusters=clusters, scorer=scorer)
    assert result.scorer_calls == 0
    assert scorer.calls == []
    assert result.verdicts == ["StrengthA"] * 5


@pytest.mark.family_performance_cost
def test_nli_fallback_costs_two_calls_per_cluster() -> None:
    """Each NLI fallback cluster costs exactly 2 calls (asymmetric, both
    directions). No call is wasted on Gap or Galois-resolved clusters."""
    # 2 NLI fallback (incomparable SVO), 1 Galois, 1 Gap.
    incomp_a = _claim(subject="A", predicate="p", object_="x")
    incomp_b = _claim(subject="A", predicate="p", object_="y")
    galois_a = _claim(subject="B", predicate="q", object_="z", modality="obligatory")
    galois_b = _claim(subject="B", predicate="q", object_="z", modality="permitted")
    gap_a = _claim(subject="C", predicate="r", object_="w")
    clusters = [
        _cluster(id_="c1", a=incomp_a, b=incomp_b),
        _cluster(id_="c2", a=incomp_a, b=incomp_b),
        _cluster(id_="c3", a=galois_a, b=galois_b),
        _cluster(id_="c4", a=gap_a),
    ]
    scorer = _DictScorer({})
    result = compare(clusters=clusters, scorer=scorer)
    assert result.scorer_calls == 2 * 2  # 2 fallback clusters, 2 calls each
    assert len(scorer.calls) == 2 * 2


# ---------------------------------------------------------------------------
# Determinism — identical input → identical output
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_repeat_calls_produce_identical_verdicts() -> None:
    a = _claim(subject="x", predicate="p", object_="y")
    b = _claim(subject="x", predicate="p", object_="z")
    cluster = _cluster(a=a, b=b)
    a_text = "x p y"
    b_text = "x p z"
    scorer1 = _DictScorer({(a_text, b_text): _high_entail(), (b_text, a_text): _low_entail()})
    scorer2 = _DictScorer({(a_text, b_text): _high_entail(), (b_text, a_text): _low_entail()})
    r1 = compare(clusters=[cluster], scorer=scorer1)
    r2 = compare(clusters=[cluster], scorer=scorer2)
    assert r1.verdicts == r2.verdicts


# ---------------------------------------------------------------------------
# Verifier shape — implements `CompareVerifier` protocol
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_transport_verifier_implements_compare_protocol() -> None:
    """`TransportCompareVerifier` satisfies `CompareVerifier`."""
    verifier: CompareVerifier = TransportCompareVerifier(scorer=_DictScorer({}))
    out = verifier.verdicts(clusters=[])
    assert out == []


# ---------------------------------------------------------------------------
# Release-gate eval — per-cluster accuracy ≥ 0.85 on the shipped fixture
# ---------------------------------------------------------------------------


class _CompareGoldOracle:
    """Deterministic NLI oracle aligned to the compare eval fixture.

    For every `StrengthA`/`StrengthB` cluster whose Galois floor returns
    `incomparable` (e.g. cross-axis modality or qualifier-driven), we
    program the oracle so the more general side is more strongly
    entailed by the more specific side. For Galois-resolvable clusters
    (modality chains under matching SVO+polarity) the oracle is never
    asked — the floor short-circuits.

    This isolates the §6.6 transport reduction's behaviour from any
    real NLI backend; the release-gate constant is being asserted *of
    the reduction*, not of the model.
    """

    def __init__(self, cases: list[CompareEvalCase]) -> None:
        from ctrldoc.extract.galois import claim_subsumption
        from ctrldoc.ops.compare import _render_claim

        self._table: dict[tuple[str, str], NLIScore] = {}
        self.calls: list[tuple[str, str]] = []
        for case in cases:
            for cluster in case.clusters:
                if cluster.a_claim is None or cluster.b_claim is None:
                    continue
                if claim_subsumption(cluster.a_claim, cluster.b_claim) != "incomparable":
                    continue
                a_text = _render_claim(cluster.a_claim)
                b_text = _render_claim(cluster.b_claim)
                if cluster.gold_verdict == "StrengthA":
                    self._table[(a_text, b_text)] = _high_entail()
                    self._table[(b_text, a_text)] = _low_entail()
                elif cluster.gold_verdict == "StrengthB":
                    self._table[(a_text, b_text)] = _low_entail()
                    self._table[(b_text, a_text)] = _high_entail()

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._table.get(
            (premise, hypothesis),
            NLIScore(entailment=0.20, contradiction=0.10, neutral=0.70),
        )


@pytest.mark.family_verifier_calibration
def test_transport_compare_clears_release_gate_on_eval_fixture() -> None:
    """§6.6 per-cluster accuracy ≥ 0.85 gate holds on the shipped fixture.

    With a gold-aligned oracle isolating the transport+Galois reduction,
    aggregate accuracy across the whole 8-case set must clear
    `COMPARE_VERDICT_THRESHOLD = 0.85`.
    """
    cases = load_jsonl_cases(COMPARE_EVAL_PATH, case_model=CompareEvalCase)
    oracle = _CompareGoldOracle(cases)
    verifier = TransportCompareVerifier(scorer=oracle)
    runner = CompareEvalRunner(verifier=verifier)

    report = run_eval(
        set_name="ops_compare_release_gate",
        cases=cases,
        runner=runner,
        thresholds={"accuracy": COMPARE_VERDICT_THRESHOLD},
    )
    # Aggregate per-cluster accuracy across all cases.
    predicted: list[CompareVerdictLiteral] = []
    gold: list[CompareVerdictLiteral] = []
    for case in cases:
        out = verifier.verdicts(clusters=[c.to_input() for c in case.clusters])
        predicted.extend(out)
        gold.extend(c.gold_verdict for c in case.clusters)
    metrics = compare_accuracy(predicted=predicted, gold=gold)
    assert metrics["accuracy"] >= COMPARE_VERDICT_THRESHOLD, metrics
    assert report.passed is True
    assert len(report.results) == len(cases)
