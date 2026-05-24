"""Tier-2 NLI edge inference — `entails` / `contradicts` over universal claims.

The Tier-2 NLI edge layer is the §6.5 producer of probabilistic edges
on the per-doc claim graph. It consumes the universal `ClaimTuple` list
emitted by the Tier-2 SVO extractor (S-128) and emits `TypedEdge` rows
of type `entails` or `contradicts` between pairs whose NLI scorer
reports high mass on the corresponding label.

The cost contract the inferer must obey is the §6.5 candidate-retrieval
budget: at most `k_candidates * N` pairs are presented to the NLI
backend, where `N` is the number of claims and `k_candidates` is the
configurable retrieval fanout (default 5). Quadratic enumeration is
explicitly forbidden — the test set drives N = 8 claims through the
inferer and asserts the backend received at most 40 calls.

Edges carry:

* `confidence` = the raw top-label probability from the scorer
  (calibration is the job of the §6.5 isotonic-regression layer landing
  later in Phase 17, not of the Tier-2 NLI inferer itself).
* `raw_score` = the same top-label probability, persisted separately so
  the calibration layer can fit isotonic regression against it.
* `source = "nli"` — the §6.5 provenance tag.
* `citations` = the universal-tuple-text surfaces of both endpoints, so
  the trace renderer can show the premise + hypothesis the verdict came
  from.

SPEC-REF: §6.5 (probabilistic edges + calibration — Tier-2 NLI)
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.tier2_nli import (
    DEFAULT_K_CANDIDATES,
    NLI_CONTRADICT_THRESHOLD,
    NLI_ENTAIL_THRESHOLD,
    Tier2NLIConfig,
    Tier2NLIEdgeInferer,
    Tier2NLIExtraction,
    render_claim_text,
)

# ---------------------------------------------------------------------------
# Stub scorer — deterministic, records every call
# ---------------------------------------------------------------------------


class _DictScorer:
    """A `CalibrationScorer` keyed on `(premise, hypothesis)`.

    Missing keys default to a uniform-ish neutral score so the
    candidate-budget tests can drive N claims without enumerating every
    pair's hypothetical NLI verdict.
    """

    def __init__(self, table: dict[tuple[str, str], NLIScore]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        if (premise, hypothesis) in self._table:
            return self._table[(premise, hypothesis)]
        # Default: neutral with mild confidence so nothing crosses the
        # entailment / contradiction threshold by accident.
        return NLIScore(entailment=0.20, contradiction=0.20, neutral=0.60)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _claim(
    subject: str,
    predicate: str,
    obj: str,
    *,
    polarity: str = "affirmative",
    modality: str = "asserted",
    qualifier: str = "",
) -> ClaimTuple:
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier,
    )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_empty_input_returns_no_edges() -> None:
    """Zero claims in → zero edges out, no scorer calls made."""
    scorer = _DictScorer({})
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer([])

    assert isinstance(out, Tier2NLIExtraction)
    assert out.edges == []
    assert out.scorer_calls == 0
    assert scorer.calls == []


@pytest.mark.family_determinism
def test_single_claim_emits_no_edges() -> None:
    """A single claim has no peers to pair with — no edges, no calls."""
    scorer = _DictScorer({})
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer([_claim("the system", "uses", "tls")])

    assert out.edges == []
    assert out.scorer_calls == 0


@pytest.mark.family_verifier_calibration
def test_high_entailment_score_emits_entails_edge() -> None:
    """A scorer returning entailment ≥ threshold yields an `entails` edge."""
    a = _claim("the system", "uses", "tls 1.3")
    b = _claim("the system", "uses", "tls")
    p, h = render_claim_text(a), render_claim_text(b)

    scorer = _DictScorer({(p, h): NLIScore(entailment=0.92, contradiction=0.02, neutral=0.06)})
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer([a, b])

    entails = [e for e in out.edges if e.type == "entails"]
    assert len(entails) == 1
    edge = entails[0]
    assert edge.source == "nli"
    assert edge.confidence == pytest.approx(0.92)
    assert edge.raw_score == pytest.approx(0.92)
    assert edge.paraphrase_votes is None
    # Citations carry both endpoints' rendered text via a synthetic span.
    assert len(edge.citations) == 2
    assert any(c.text == p for c in edge.citations)
    assert any(c.text == h for c in edge.citations)


@pytest.mark.family_verifier_calibration
def test_high_contradiction_score_emits_contradicts_edge() -> None:
    """A scorer returning contradiction ≥ threshold yields a `contradicts` edge."""
    a = _claim("the cache", "is", "warm")
    b = _claim("the cache", "is", "cold")
    p, h = render_claim_text(a), render_claim_text(b)

    scorer = _DictScorer({(p, h): NLIScore(entailment=0.05, contradiction=0.88, neutral=0.07)})
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer([a, b])

    contras = [e for e in out.edges if e.type == "contradicts"]
    assert len(contras) == 1
    edge = contras[0]
    assert edge.source == "nli"
    assert edge.confidence == pytest.approx(0.88)


@pytest.mark.family_verifier_calibration
def test_neutral_high_score_emits_no_edge() -> None:
    """A scorer dominated by `neutral` produces no entailment / contradiction edge."""
    a = _claim("alice", "wrote", "the rfc")
    b = _claim("bob", "reviewed", "the rfc")
    p, h = render_claim_text(a), render_claim_text(b)

    scorer = _DictScorer({(p, h): NLIScore(entailment=0.10, contradiction=0.05, neutral=0.85)})
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer([a, b])

    assert out.edges == []


@pytest.mark.family_verifier_calibration
def test_below_threshold_emits_no_edge() -> None:
    """An entailment score just under the threshold does not emit an edge."""
    a = _claim("the proxy", "forwards", "every request")
    b = _claim("the proxy", "forwards", "requests")
    p, h = render_claim_text(a), render_claim_text(b)

    scorer = _DictScorer(
        {
            (p, h): NLIScore(
                entailment=NLI_ENTAIL_THRESHOLD - 0.01,
                contradiction=0.02,
                neutral=1.0 - (NLI_ENTAIL_THRESHOLD - 0.01) - 0.02,
            )
        }
    )
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer([a, b])

    assert out.edges == []


# ---------------------------------------------------------------------------
# Candidate-budget contract — the §6.5 `<= 5 * N` pairs rule
# ---------------------------------------------------------------------------


@pytest.mark.family_performance_cost
def test_candidate_budget_holds_at_default_k() -> None:
    """For N claims, scorer is called at most `k_candidates * N` times (default 5N)."""
    # 8 distinct claims — quadratic enumeration would be 8 * 7 = 56 calls.
    claims = [
        _claim("the system", "uses", "tls 1.3"),
        _claim("the system", "uses", "tls"),
        _claim("the cache", "is", "warm"),
        _claim("the cache", "is", "cold"),
        _claim("alice", "wrote", "the rfc"),
        _claim("bob", "reviewed", "the rfc"),
        _claim("the proxy", "forwards", "requests"),
        _claim("the worker", "retries", "on 503 responses"),
    ]
    scorer = _DictScorer({})
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer(claims)

    n = len(claims)
    max_calls = DEFAULT_K_CANDIDATES * n
    assert out.scorer_calls == len(scorer.calls)
    assert out.scorer_calls <= max_calls, (
        f"scorer called {out.scorer_calls} times for N={n}; budget is {max_calls}"
    )
    # Also: must beat the quadratic baseline strictly when N > k+1.
    assert out.scorer_calls < n * (n - 1)


@pytest.mark.family_performance_cost
def test_candidate_budget_honoured_with_custom_k() -> None:
    """A smaller k_candidates strictly tightens the call budget."""
    claims = [
        _claim("alpha", "is", "fast"),
        _claim("beta", "is", "slow"),
        _claim("gamma", "is", "warm"),
        _claim("delta", "is", "cold"),
        _claim("epsilon", "is", "eventually consistent"),
        _claim("zeta", "is", "linearizable"),
    ]
    scorer = _DictScorer({})
    inferer = Tier2NLIEdgeInferer(scorer=scorer, config=Tier2NLIConfig(k_candidates=2))

    out = inferer.infer(claims)

    assert out.scorer_calls <= 2 * len(claims)


# ---------------------------------------------------------------------------
# Candidate retrieval — token-overlap ranking
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_candidate_retrieval_prefers_token_overlap() -> None:
    """High-overlap pairs land inside the candidate set; disjoint pairs are
    pruned. The proof is that the scorer is asked about the overlap pair
    and not asked about an obvious zero-overlap pair when k is tight."""
    target = _claim("the worker", "retries", "on 503 responses")
    near = _claim("the worker", "retries", "on transient errors")
    far = _claim("the chairman", "presented", "the agenda")

    # k=1 means each claim gets exactly one candidate. The token-overlap
    # ranker should pair (target, near) — they share `the worker` and
    # `retries` — and never pair (target, far).
    scorer = _DictScorer({})
    inferer = Tier2NLIEdgeInferer(scorer=scorer, config=Tier2NLIConfig(k_candidates=1))

    inferer.infer([target, near, far])

    target_text = render_claim_text(target)
    near_text = render_claim_text(near)
    far_text = render_claim_text(far)

    # Among the scorer's recorded calls, the (target, near) pair must
    # appear (either direction) and (target, far) must not.
    pair_in_calls = lambda x, y: (  # noqa: E731
        (x, y) in scorer.calls or (y, x) in scorer.calls
    )
    assert pair_in_calls(target_text, near_text)
    assert not pair_in_calls(target_text, far_text)


# ---------------------------------------------------------------------------
# Determinism + ordering
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_repeated_runs_produce_identical_output() -> None:
    """The inferer is deterministic given a deterministic scorer."""
    claims = [
        _claim("the system", "uses", "tls 1.3"),
        _claim("the system", "uses", "tls"),
        _claim("the proxy", "forwards", "requests"),
    ]
    p, h = render_claim_text(claims[0]), render_claim_text(claims[1])
    scorer = _DictScorer({(p, h): NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07)})
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out_a = inferer.infer(claims)
    out_b = inferer.infer(claims)

    assert [e.model_dump() for e in out_a.edges] == [e.model_dump() for e in out_b.edges]


@pytest.mark.family_determinism
def test_edges_are_sorted_for_stable_diffs() -> None:
    """Edges sort by `(type, src_id, dst_id)` so diffs are reviewer-friendly."""
    a = _claim("the system", "uses", "tls 1.3")
    b = _claim("the system", "uses", "tls")
    c = _claim("the cache", "is", "warm")
    d = _claim("the cache", "is", "cold")
    pa, pb = render_claim_text(a), render_claim_text(b)
    pc, pd = render_claim_text(c), render_claim_text(d)
    scorer = _DictScorer(
        {
            (pa, pb): NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07),
            (pc, pd): NLIScore(entailment=0.04, contradiction=0.88, neutral=0.08),
        }
    )
    inferer = Tier2NLIEdgeInferer(scorer=scorer)

    out = inferer.infer([a, b, c, d])

    keys = [(e.type, e.src_id, e.dst_id) for e in out.edges]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Render contract
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_render_claim_text_includes_negation_for_negative_polarity() -> None:
    """A negative-polarity claim renders with explicit negation in surface text."""
    pos = _claim("the worker", "retries", "on errors")
    neg = _claim(
        "the worker",
        "retries",
        "on errors",
        polarity="negative",
    )

    pos_text = render_claim_text(pos)
    neg_text = render_claim_text(neg)

    assert pos_text != neg_text
    assert "not" in neg_text.lower()


@pytest.mark.family_determinism
def test_render_claim_text_includes_qualifier_when_present() -> None:
    """The qualifier surface flows into the rendered string when set."""
    q = _claim(
        "the consensus protocol",
        "elects",
        "a leader",
        qualifier="within five seconds",
    )

    text = render_claim_text(q)

    assert "within five seconds" in text


@pytest.mark.family_determinism
def test_default_thresholds_match_spec_defaults() -> None:
    """Threshold defaults stay in the (0, 1) interval and are non-degenerate."""
    assert 0.0 < NLI_ENTAIL_THRESHOLD < 1.0
    assert 0.0 < NLI_CONTRADICT_THRESHOLD < 1.0
    assert NLI_ENTAIL_THRESHOLD >= 0.5
    assert NLI_CONTRADICT_THRESHOLD >= 0.5
    assert DEFAULT_K_CANDIDATES >= 1
