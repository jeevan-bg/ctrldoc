"""L5 coverage + list_check operations via optimal transport.

§6.6 collapses `coverage(A → B)` to "min-cost transport of B's
claim-mass onto A's claim-mass, cost = `1 - NLI_entail(A, B)`. Unmoved
mass = uncovered." `list_check(items, D)` parses the list as a tiny
target doc and answers per-item via the same primitive — exactly the
same algorithm, different input shape.

This module ships:

* `TransportCoverageVerifier` — a `CrossDocCoverageVerifier`-shaped
  consumer that wires an `NLIScorer` into the §6.6 transport reduction
  and emits one `Covered` / `Missing` verdict per target claim.
* `coverage` / `list_check` — thin functional surfaces over the
  verifier, named to match the §9 CLI commands they back.

The release gate is the §6.6 per-claim accuracy ≥ 0.85 contract
(`COVERAGE_VERDICT_ACCURACY_THRESHOLD`). The eval substrate
(`ctrldoc.eval.cross_doc_coverage`) already pins the contract; this
test module runs the verifier with a deterministic gold-aligned NLI
oracle over the shipped 12-case fixture and asserts the aggregate
per-target-claim accuracy clears the bar — i.e. the transport
reduction itself is faithful, independent of any real NLI backend's
quality.

SPEC-REF: §6.6 (optimal-transport core — one algorithm, five queries),
§14 (eval substrate)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.cross_doc_coverage import (
    CROSS_DOC_COVERAGE_THRESHOLD,
    CoverageVerdictLiteral,
    CrossDocCoverageEvalCase,
    CrossDocCoverageEvalRunner,
    CrossDocCoverageVerifier,
    coverage_accuracy,
)
from ctrldoc.eval.harness import load_jsonl_cases, run_eval
from ctrldoc.ops.coverage import (
    COVERAGE_ENTAILMENT_THRESHOLD,
    COVERAGE_VERDICT_ACCURACY_THRESHOLD,
    CoverageConfig,
    CoverageResult,
    TransportCoverageVerifier,
    coverage,
    list_check,
)

CCOV_EVAL_PATH = Path(__file__).parent / "eval" / "cross_doc_coverage_eval.jsonl"


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
        self._default = default or NLIScore(entailment=0.10, contradiction=0.10, neutral=0.80)
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._table.get((premise, hypothesis), self._default)


def _high_entail() -> NLIScore:
    return NLIScore(entailment=0.92, contradiction=0.03, neutral=0.05)


def _high_contradict() -> NLIScore:
    return NLIScore(entailment=0.04, contradiction=0.90, neutral=0.06)


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


# ---------------------------------------------------------------------------
# Config — rejects nonsense thresholds eagerly
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_config_default_entailment_threshold_matches_module_constant() -> None:
    """Default config exposes `COVERAGE_ENTAILMENT_THRESHOLD`."""
    cfg = CoverageConfig()
    assert cfg.entailment_threshold == COVERAGE_ENTAILMENT_THRESHOLD


@pytest.mark.family_determinism
def test_config_rejects_threshold_outside_unit_interval() -> None:
    """`entailment_threshold` outside (0, 1) is a construction-time error."""
    with pytest.raises(ValueError):
        CoverageConfig(entailment_threshold=0.0)
    with pytest.raises(ValueError):
        CoverageConfig(entailment_threshold=1.0)
    with pytest.raises(ValueError):
        CoverageConfig(entailment_threshold=-0.1)


# ---------------------------------------------------------------------------
# CoverageResult shape
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_result_is_frozen_and_strict() -> None:
    """Result rejects extra fields and pinned attributes are read-only."""
    r = CoverageResult(verdicts=["Covered"], scorer_calls=1)
    with pytest.raises(ValidationError):
        r.verdicts = ["Missing"]  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CoverageResult(verdicts=["Covered"], scorer_calls=1, stray="oops")  # type: ignore[call-arg]


@pytest.mark.family_determinism
def test_result_verdicts_only_accept_literals() -> None:
    """Pydantic literal-narrowing rejects unknown verdict strings."""
    with pytest.raises(ValidationError):
        CoverageResult(verdicts=["MaybeCovered"], scorer_calls=1)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Empty-target short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_empty_target_returns_empty_verdicts() -> None:
    """Empty target list means there's nothing to grade; no scorer calls."""
    scorer = _DictScorer({})
    result = coverage(source=[_claim()], target=[], scorer=scorer)
    assert result.verdicts == []
    assert result.scorer_calls == 0
    assert scorer.calls == []


@pytest.mark.family_determinism
def test_empty_source_marks_every_target_missing() -> None:
    """No sources → no flow can reach any target → all Missing."""
    scorer = _DictScorer({})
    targets = [_claim(predicate="A"), _claim(predicate="B")]
    result = coverage(source=[], target=targets, scorer=scorer)
    assert result.verdicts == ["Missing", "Missing"]
    assert result.scorer_calls == 0


# ---------------------------------------------------------------------------
# Hard verdicts: high entailment → Covered, neutral → Missing
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_high_entailment_yields_covered() -> None:
    """A single source with strong entailment covers the single target."""
    src = _claim(subject="A", predicate="entails", object_="B")
    tgt = _claim(subject="A", predicate="implies", object_="B")
    scorer = _DictScorer(
        {("A entails B", "A implies B"): _high_entail()},
    )
    result = coverage(source=[src], target=[tgt], scorer=scorer)
    assert result.verdicts == ["Covered"]
    assert result.scorer_calls == 1


@pytest.mark.family_verifier_calibration
def test_neutral_yields_missing() -> None:
    """Neutral-dominated NLI verdict → no real source beats slack → Missing."""
    src = _claim(subject="A", predicate="describes", object_="X")
    tgt = _claim(subject="B", predicate="describes", object_="Y")
    scorer = _DictScorer(
        {("A describes X", "B describes Y"): _low_entail()},
    )
    result = coverage(source=[src], target=[tgt], scorer=scorer)
    assert result.verdicts == ["Missing"]


@pytest.mark.family_verifier_calibration
def test_strong_contradiction_yields_missing_not_covered() -> None:
    """A confidently contradicted target gets Missing under the
    Covered/Missing surface — never accidentally Covered."""
    src = _claim(subject="data", predicate="may be shared", object_="with affiliates")
    tgt = _claim(
        subject="data",
        predicate="may be shared",
        object_="with affiliates",
        polarity="negative",
    )
    rendered_src = "data may be shared with affiliates"
    rendered_tgt = "data does not may be shared with affiliates"
    scorer = _DictScorer({(rendered_src, rendered_tgt): _high_contradict()})
    result = coverage(source=[src], target=[tgt], scorer=scorer)
    assert result.verdicts == ["Missing"]


# ---------------------------------------------------------------------------
# Many-to-one transport: one target covered by union of sources
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_many_to_one_picks_best_source() -> None:
    """When several sources each partially support a target, the best one
    wins and the verdict is Covered."""
    src_a = _claim(subject="X", predicate="does", object_="A")
    src_b = _claim(subject="X", predicate="does", object_="B")
    tgt = _claim(subject="X", predicate="does", object_="A")
    scorer = _DictScorer(
        {
            ("X does A", "X does A"): _high_entail(),  # strong match
            ("X does B", "X does A"): _low_entail(),  # weak match
        },
    )
    result = coverage(source=[src_a, src_b], target=[tgt], scorer=scorer)
    assert result.verdicts == ["Covered"]
    # Both source candidates scored exactly once.
    assert result.scorer_calls == 2


# ---------------------------------------------------------------------------
# Linear cost contract — scorer calls equal |sources| * |targets|
# ---------------------------------------------------------------------------


@pytest.mark.family_performance_cost
def test_scorer_calls_equal_source_times_target() -> None:
    """The transport reduction asks NLI on every (source, target) pair exactly once."""
    sources = [_claim(subject="s", predicate="p", object_=f"o{i}") for i in range(4)]
    targets = [_claim(subject="s", predicate="p", object_=f"t{j}") for j in range(3)]
    scorer = _DictScorer({})
    result = coverage(source=sources, target=targets, scorer=scorer)
    assert result.scorer_calls == 4 * 3
    assert len(scorer.calls) == 4 * 3


@pytest.mark.family_performance_cost
def test_scorer_called_once_per_distinct_pair() -> None:
    """The same (source, target) is never scored twice across one call."""
    sources = [_claim(predicate=f"p{i}") for i in range(3)]
    targets = [_claim(predicate=f"q{j}") for j in range(2)]
    scorer = _DictScorer({})
    coverage(source=sources, target=targets, scorer=scorer)
    distinct = set(scorer.calls)
    assert len(distinct) == len(scorer.calls)


# ---------------------------------------------------------------------------
# Determinism — repeat runs produce byte-identical verdict lists
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_repeat_calls_produce_identical_verdicts() -> None:
    """Identical input + identical scorer → identical verdict list."""
    sources = [_claim(predicate=f"p{i}") for i in range(3)]
    targets = [_claim(predicate=f"q{j}") for j in range(3)]
    scorer1 = _DictScorer({})
    scorer2 = _DictScorer({})
    r1 = coverage(source=sources, target=targets, scorer=scorer1)
    r2 = coverage(source=sources, target=targets, scorer=scorer2)
    assert r1.verdicts == r2.verdicts


# ---------------------------------------------------------------------------
# list_check — same algorithm, different surface
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_list_check_per_item_verdicts() -> None:
    """`list_check` mirrors `coverage` semantics — items map to verdicts."""
    items = [
        _claim(subject="A", predicate="p", object_="X"),
        _claim(subject="B", predicate="p", object_="Y"),
    ]
    doc = [
        _claim(subject="A", predicate="p", object_="X"),
    ]
    scorer = _DictScorer(
        {
            ("A p X", "A p X"): _high_entail(),
            ("A p X", "B p Y"): _low_entail(),
        },
    )
    result = list_check(items=items, doc=doc, scorer=scorer)
    assert result.verdicts == ["Covered", "Missing"]
    assert result.scorer_calls == 1 * 2


# ---------------------------------------------------------------------------
# Verifier shape — implements `CrossDocCoverageVerifier` protocol
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_transport_verifier_implements_protocol() -> None:
    """`TransportCoverageVerifier` is a structural `CrossDocCoverageVerifier`."""
    verifier: CrossDocCoverageVerifier = TransportCoverageVerifier(scorer=_DictScorer({}))
    # Round-trip an empty-target call through the protocol surface to
    # exercise it concretely (not just isinstance against the runtime
    # Protocol guard, which would only check method-name presence).
    out = verifier.verdicts(source=[], target=[])
    assert out == []


# ---------------------------------------------------------------------------
# Release-gate eval — per-claim accuracy ≥ 0.85 on the shipped fixture
# ---------------------------------------------------------------------------


class _GoldAlignedScorer:
    """Deterministic NLI oracle keyed on the shipped eval fixture.

    For every gold-Covered target claim the verifier ought to find at
    least one source with high entailment; the scorer assigns
    high-entailment scores to the *best-token-overlap* source for each
    Covered target, and leaves everything else neutral. Polarity-flip
    contradiction is surfaced as a confident contradiction score. This
    isolates the transport reduction's behaviour from the quality of any
    real NLI backend — the eval substrate's per-claim accuracy gate is
    being asserted *of the reduction*, not of the model.
    """

    def __init__(self, cases: list[CrossDocCoverageEvalCase]) -> None:
        # Pre-compute a table: (premise, hypothesis) → NLIScore.
        from ctrldoc.ops.coverage import _render_claim

        self._table: dict[tuple[str, str], NLIScore] = {}
        self.calls: list[tuple[str, str]] = []
        for case in cases:
            for tc in case.target_claims:
                tgt_text = _render_claim(tc.claim)
                if tc.gold_verdict == "Covered":
                    # Pick the source with the maximum token-overlap
                    # against the target — that's the "best" supporter.
                    best_src_text = self._best_source(case, tc.claim)
                    if best_src_text is not None:
                        self._table[(best_src_text, tgt_text)] = NLIScore(
                            entailment=0.95, contradiction=0.02, neutral=0.03
                        )
                elif tc.gold_verdict == "Missing":
                    # Detect polarity-flip contradiction so a Missing
                    # gold sits on the negative cell of the 3-way space.
                    if tc.claim.polarity == "negative":
                        for sc in case.source_claims:
                            if (
                                sc.polarity == "affirmative"
                                and self._token_overlap(sc, tc.claim) >= 0.5
                            ):
                                src_text = _render_claim(sc)
                                self._table[(src_text, tgt_text)] = NLIScore(
                                    entailment=0.02,
                                    contradiction=0.94,
                                    neutral=0.04,
                                )

    @staticmethod
    def _tokens(claim: ClaimTuple) -> frozenset[str]:
        return frozenset(
            t
            for t in (claim.subject + " " + claim.predicate + " " + claim.object).lower().split()
            if t
        )

    @classmethod
    def _token_overlap(cls, a: ClaimTuple, b: ClaimTuple) -> float:
        ta = cls._tokens(a)
        tb = cls._tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _best_source(self, case: CrossDocCoverageEvalCase, target: ClaimTuple) -> str | None:
        from ctrldoc.ops.coverage import _render_claim

        if not case.source_claims:
            return None
        # Stable tiebreak on rendered text to keep determinism.
        ranked = sorted(
            case.source_claims,
            key=lambda sc: (-self._token_overlap(sc, target), _render_claim(sc)),
        )
        return _render_claim(ranked[0])

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        return self._table.get(
            (premise, hypothesis),
            NLIScore(entailment=0.15, contradiction=0.10, neutral=0.75),
        )


@pytest.mark.family_verifier_calibration
def test_transport_coverage_clears_release_gate_on_eval_fixture() -> None:
    """The §6.6 per-claim accuracy ≥ 0.85 gate holds on the 12-case fixture.

    With a gold-aligned NLI oracle isolating the transport reduction, the
    verifier must achieve aggregate per-target-claim accuracy at or
    above `CROSS_DOC_COVERAGE_THRESHOLD = 0.85`.
    """
    cases = load_jsonl_cases(CCOV_EVAL_PATH, case_model=CrossDocCoverageEvalCase)
    scorer = _GoldAlignedScorer(cases)
    verifier = TransportCoverageVerifier(scorer=scorer)
    runner = CrossDocCoverageEvalRunner(verifier=verifier)

    report = run_eval(
        set_name="ops_coverage_release_gate",
        cases=cases,
        runner=runner,
        thresholds={"accuracy": CROSS_DOC_COVERAGE_THRESHOLD},
    )
    # Aggregate per-claim accuracy across the whole fixture (the per-case
    # `accuracy` metric the runner emits is itself per-claim accuracy
    # within a case; the harness averages across cases, which is the
    # spec's "per-claim accuracy" rolled up).
    predicted: list[CoverageVerdictLiteral] = []
    gold: list[CoverageVerdictLiteral] = []
    for case in cases:
        out = verifier.verdicts(
            source=list(case.source_claims),
            target=[tc.claim for tc in case.target_claims],
        )
        predicted.extend(out)
        gold.extend(tc.gold_verdict for tc in case.target_claims)
    metrics = coverage_accuracy(predicted=predicted, gold=gold)
    assert metrics["accuracy"] >= CROSS_DOC_COVERAGE_THRESHOLD, metrics
    # The eval harness's threshold gate must also pass on this fixture.
    assert report.passed is True
    assert len(report.results) == len(cases)
    # And the module's release-gate constant matches the eval substrate's.
    assert COVERAGE_VERDICT_ACCURACY_THRESHOLD == CROSS_DOC_COVERAGE_THRESHOLD
