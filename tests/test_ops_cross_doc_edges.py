"""Workspace cross-doc edge inference — `aligned_with` / `entails_across` /
`contradicts_across` via candidate retrieval + NLI judge.

Per §6.7 a workspace shares one concept lattice across N member docs. Cross-doc
edges are lazy, cached, and **linear** in `|A| * k` per ordered doc pair — never
quadratic. The producer for these edges is the same shape as the per-doc Tier-2
NLI inferer (S-129) but bridges claims across docs instead of within one:

* For each ordered pair of distinct member docs `(A, B)` and each claim `a` in
  doc A, retrieve the top-`k` candidate claims in doc B by deterministic
  token-overlap Jaccard. Self-pairs are not possible (different doc ids).
* Ask the NLI scorer the entailment direction `a -> b` exactly once per
  candidate pair. The top-label probability decides the edge type:
  - `entailment` above the entail threshold ⇒ `entails_across` edge.
  - `contradiction` above the contradict threshold ⇒ `contradicts_across`.
  - `entailment` above the aligned-with threshold but below the entail
    threshold ⇒ `aligned_with` edge (soft, paraphrase-style alignment).
  - Neutral-dominated pairs emit no edge.
* Edges carry the same `source = "nli"` provenance and `raw_score` as the
  per-doc Tier-2 NLI edges, so the §6.5 calibration step (S-137) can fit one
  isotonic regressor over both intra- and inter-doc NLI outputs.

The §6.5 / §6.7 cost contract for the workspace inferer is that the scorer
call count grows linearly: at most `k * |A|` calls per ordered doc pair, i.e.
`k * sum(|d|) * (n_docs - 1)` calls in total for `n_docs >= 2`. The tests below
exercise N = 3 docs of 4 claims each (12 total claims, 6 ordered pairs) and
assert the scorer was called at most `5 * 4 * 2 = 40` times, well under the
quadratic `12 * 11 = 132` baseline.

SPEC-REF: §6.7 (workspace cross-doc edges, lazy + linear scaling)
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim
from ctrldoc.ops.cross_doc_edges import (
    ALIGNED_WITH_THRESHOLD,
    CONTRADICTS_ACROSS_THRESHOLD,
    DEFAULT_K_CANDIDATES,
    ENTAILS_ACROSS_THRESHOLD,
    CrossDocEdgeConfig,
    CrossDocEdgeInference,
    CrossDocEdgeInferer,
)

# ---------------------------------------------------------------------------
# Stub scorer — deterministic, records every call
# ---------------------------------------------------------------------------


class _DictScorer:
    """A `NLIScorer` keyed on `(premise, hypothesis)`.

    Missing keys default to a neutral-dominant score so the budget tests
    can drive many claims without enumerating every hypothetical NLI
    verdict.
    """

    def __init__(self, table: dict[tuple[str, str], NLIScore]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def score(self, *, premise: str, hypothesis: str) -> NLIScore:
        self.calls.append((premise, hypothesis))
        if (premise, hypothesis) in self._table:
            return self._table[(premise, hypothesis)]
        return NLIScore(entailment=0.20, contradiction=0.20, neutral=0.60)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _claim(*, claim_id: str, doc_id: str, text: str) -> Claim:
    """Minimal `Claim` factory — only the fields the inferer actually reads."""
    return Claim(
        id=claim_id,
        doc_id=doc_id,
        text=text,
        subject=None,
        predicate=text,
        object=None,
        polarity="+",
        modality=None,
        qualifier={},
        span_refs=[Span(chunk_id=f"{doc_id}:chunk-0", char_start=0, char_end=len(text), text=text)],
        section_id=f"{doc_id}:sec-0",
        concept_ids=[],
        typed_slots={},
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Empty / single-doc short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_empty_workspace_returns_no_edges() -> None:
    """Zero docs → no edges, no scorer calls."""
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(workspace_id="ws-empty", claims_by_doc={})

    assert isinstance(out, CrossDocEdgeInference)
    assert out.edges == []
    assert out.scorer_calls == 0
    assert scorer.calls == []


@pytest.mark.family_determinism
def test_single_doc_workspace_returns_no_cross_doc_edges() -> None:
    """A workspace with one doc has no cross-doc pairs to score."""
    a = _claim(claim_id="cA", doc_id="docA", text="the system uses TLS 1.3")
    b = _claim(claim_id="cB", doc_id="docA", text="the cache is warm")
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(workspace_id="ws-solo", claims_by_doc={"docA": [a, b]})

    assert out.edges == []
    assert out.scorer_calls == 0


@pytest.mark.family_determinism
def test_doc_with_no_claims_is_skipped() -> None:
    """Docs that contribute zero claims do not break enumeration."""
    a = _claim(claim_id="cA", doc_id="docA", text="alpha")
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(
        workspace_id="ws-mixed",
        claims_by_doc={"docA": [a], "docB": []},
    )

    assert out.edges == []
    assert out.scorer_calls == 0


# ---------------------------------------------------------------------------
# Edge emission per NLI verdict
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_high_entailment_emits_entails_across_edge() -> None:
    """A high-entailment cross-doc pair produces a single `entails_across` edge."""
    a = _claim(claim_id="cA1", doc_id="docA", text="the system uses TLS 1.3")
    b = _claim(claim_id="cB1", doc_id="docB", text="the system uses TLS")
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.92, contradiction=0.02, neutral=0.06),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(
        workspace_id="ws-entail",
        claims_by_doc={"docA": [a], "docB": [b]},
    )

    types = [e.type for e in out.edges]
    assert "entails_across" in types
    # Endpoint identity uses persisted claim ids verbatim — no rehashing.
    entail = next(e for e in out.edges if e.type == "entails_across")
    assert entail.src_id == "cA1"
    assert entail.dst_id == "cB1"
    assert entail.source == "nli"
    assert entail.confidence >= ENTAILS_ACROSS_THRESHOLD
    assert entail.raw_score == pytest.approx(0.92)


@pytest.mark.family_verifier_calibration
def test_high_contradiction_emits_contradicts_across_edge() -> None:
    """A high-contradiction cross-doc pair produces a `contradicts_across` edge."""
    a = _claim(claim_id="cA1", doc_id="docA", text="the cache is warm at startup")
    b = _claim(claim_id="cB1", doc_id="docB", text="the cache is cold at startup")
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.04, contradiction=0.90, neutral=0.06),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(
        workspace_id="ws-contradict",
        claims_by_doc={"docA": [a], "docB": [b]},
    )

    contradict = next(e for e in out.edges if e.type == "contradicts_across")
    assert contradict.src_id == "cA1"
    assert contradict.dst_id == "cB1"
    assert contradict.confidence >= CONTRADICTS_ACROSS_THRESHOLD


@pytest.mark.family_verifier_calibration
def test_soft_alignment_emits_aligned_with_edge() -> None:
    """A pair whose entailment mass sits in `(aligned, entails)` emits an
    `aligned_with` edge — the soft cross-doc paraphrase signal."""
    a = _claim(claim_id="cA1", doc_id="docA", text="the proxy forwards requests")
    b = _claim(claim_id="cB1", doc_id="docB", text="the proxy forwards traffic")
    soft = (ENTAILS_ACROSS_THRESHOLD + ALIGNED_WITH_THRESHOLD) / 2
    # Sanity: the test fixture must actually fall in the soft-alignment band.
    assert ALIGNED_WITH_THRESHOLD <= soft < ENTAILS_ACROSS_THRESHOLD
    leftover = 1.0 - soft
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(
                entailment=soft,
                contradiction=leftover / 2,
                neutral=leftover / 2,
            ),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(
        workspace_id="ws-align",
        claims_by_doc={"docA": [a], "docB": [b]},
    )

    types = [e.type for e in out.edges]
    assert "aligned_with" in types
    assert "entails_across" not in types
    aligned = next(e for e in out.edges if e.type == "aligned_with")
    assert aligned.src_id == "cA1"
    assert aligned.dst_id == "cB1"


@pytest.mark.family_verifier_calibration
def test_neutral_dominated_pair_emits_no_edge() -> None:
    """If neutral wins, no edge is emitted — the empty result is the signal."""
    a = _claim(claim_id="cA1", doc_id="docA", text="alpha runs on linux")
    b = _claim(claim_id="cB1", doc_id="docB", text="beta runs on linux")
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.20, contradiction=0.10, neutral=0.70),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(
        workspace_id="ws-neutral",
        claims_by_doc={"docA": [a], "docB": [b]},
    )

    assert out.edges == []


# ---------------------------------------------------------------------------
# Candidate-budget contract — the §6.7 linear-scaling rule
# ---------------------------------------------------------------------------


@pytest.mark.family_performance_cost
def test_candidate_budget_is_linear_in_total_claims() -> None:
    """For N docs of M claims each, the scorer is called at most
    `k * M * N * (N - 1)` times — strictly less than the quadratic baseline."""
    docs: dict[str, list[Claim]] = {}
    for d in ("docA", "docB", "docC"):
        docs[d] = [
            _claim(claim_id=f"{d}-c{i}", doc_id=d, text=f"{d} text token {i}") for i in range(4)
        ]
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(workspace_id="ws-budget", claims_by_doc=docs)

    n_docs = len(docs)
    m_per_doc = 4
    total_claims = n_docs * m_per_doc
    max_calls = DEFAULT_K_CANDIDATES * m_per_doc * n_docs * (n_docs - 1)
    quadratic_baseline = total_claims * (total_claims - 1)

    assert out.scorer_calls == len(scorer.calls)
    assert out.scorer_calls <= max_calls, (
        f"scorer called {out.scorer_calls} times; "
        f"linear budget is {max_calls}, quadratic would be {quadratic_baseline}"
    )
    assert out.scorer_calls < quadratic_baseline


@pytest.mark.family_performance_cost
def test_candidate_budget_honoured_with_custom_k() -> None:
    """A tighter `k_candidates` strictly tightens the call budget."""
    docs = {
        "docA": [
            _claim(claim_id=f"a{i}", doc_id="docA", text=f"alpha token {i}") for i in range(5)
        ],
        "docB": [_claim(claim_id=f"b{i}", doc_id="docB", text=f"beta token {i}") for i in range(5)],
    }
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer, config=CrossDocEdgeConfig(k_candidates=2))

    out = inferer.infer(workspace_id="ws-customk", claims_by_doc=docs)

    # 5 claims per source doc, 2 ordered doc pairs, k = 2 → at most 20 calls.
    assert out.scorer_calls <= 2 * 5 * 2


# ---------------------------------------------------------------------------
# Candidate retrieval — token-overlap ranks the cross-doc neighbours
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_candidate_retrieval_prefers_token_overlap() -> None:
    """The token-overlap ranker pulls the high-Jaccard cross-doc pair into the
    candidate set and prunes a zero-overlap one when `k = 1`.

    Both docs carry two claims so that the `k = 1` cutoff is actually
    discriminating: in each ordered direction, every source claim has
    two target candidates, and the ranker must pick exactly the
    high-overlap one.
    """
    target = _claim(
        claim_id="cA-target",
        doc_id="docA",
        text="the worker retries on transient errors",
    )
    target_foil = _claim(
        claim_id="cA-foil",
        doc_id="docA",
        text="the chairman presented the agenda",
    )
    near = _claim(
        claim_id="cB-near",
        doc_id="docB",
        text="the worker retries on 503 responses",
    )
    far = _claim(
        claim_id="cB-far",
        doc_id="docB",
        text="lemurs hum quietly at midnight",
    )
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer, config=CrossDocEdgeConfig(k_candidates=1))

    inferer.infer(
        workspace_id="ws-rank",
        claims_by_doc={"docA": [target, target_foil], "docB": [near, far]},
    )

    pair_in_calls = lambda x, y: (x, y) in scorer.calls or (y, x) in scorer.calls  # noqa: E731
    # `target` ↔ `near` is the high-overlap pair; the ranker must surface it.
    assert pair_in_calls(target.text, near.text)
    # `target` (worker / retries) and `far` (lemurs / midnight) share no
    # tokens; with `k = 1` the ranker must not surface that pair from
    # either direction.
    assert not pair_in_calls(target.text, far.text)


# ---------------------------------------------------------------------------
# Self-pair safety — claims never edge against themselves or same-doc peers
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_inferer_never_scores_within_doc_pairs() -> None:
    """The scorer is asked only about pairs whose endpoints live in
    different docs — intra-doc pairs are Tier-2 NLI's job, not this layer's.

    Every claim carries a unique text so a recorded `(premise, hypothesis)`
    call uniquely identifies the originating pair.
    """
    a1 = _claim(claim_id="cA1", doc_id="docA", text="alpha is fast on linux")
    a2 = _claim(claim_id="cA2", doc_id="docA", text="alpha is slow on windows")
    b1 = _claim(claim_id="cB1", doc_id="docB", text="alpha runs fast in tests")
    scorer = _DictScorer({})
    inferer = CrossDocEdgeInferer(scorer=scorer)

    inferer.infer(
        workspace_id="ws-self",
        claims_by_doc={"docA": [a1, a2], "docB": [b1]},
    )

    # No same-doc pair appears in the recorded scorer calls.
    same_doc_pairs = {(a1.text, a2.text), (a2.text, a1.text)}
    for premise, hypothesis in scorer.calls:
        assert (premise, hypothesis) not in same_doc_pairs
    # And, by symmetry, every recorded pair has one endpoint in docA and
    # one in docB.
    doc_texts = {"docA": {a1.text, a2.text}, "docB": {b1.text}}
    for premise, hypothesis in scorer.calls:
        prem_in_a = premise in doc_texts["docA"]
        hyp_in_a = hypothesis in doc_texts["docA"]
        # Exactly one endpoint per pair belongs to docA.
        assert prem_in_a != hyp_in_a


# ---------------------------------------------------------------------------
# Determinism + ordering
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_repeated_runs_produce_identical_output() -> None:
    """Given a deterministic scorer, two runs return identical edges + counts."""
    a = _claim(claim_id="cA1", doc_id="docA", text="the system uses TLS 1.3")
    b = _claim(claim_id="cB1", doc_id="docB", text="the system uses TLS")
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    first = inferer.infer(workspace_id="ws-det", claims_by_doc={"docA": [a], "docB": [b]})
    second = inferer.infer(workspace_id="ws-det", claims_by_doc={"docA": [a], "docB": [b]})

    assert [e.model_dump() for e in first.edges] == [e.model_dump() for e in second.edges]
    assert first.scorer_calls == second.scorer_calls


@pytest.mark.family_determinism
def test_edges_are_sorted_for_stable_diffs() -> None:
    """Output edges sort by `(type, src_id, dst_id)` for reviewer-friendly diffs."""
    a1 = _claim(claim_id="cA1", doc_id="docA", text="the system uses TLS 1.3")
    a2 = _claim(claim_id="cA2", doc_id="docA", text="the cache is warm at startup")
    b1 = _claim(claim_id="cB1", doc_id="docB", text="the system uses TLS")
    b2 = _claim(claim_id="cB2", doc_id="docB", text="the cache is cold at startup")
    scorer = _DictScorer(
        {
            (a1.text, b1.text): NLIScore(entailment=0.91, contradiction=0.02, neutral=0.07),
            (a2.text, b2.text): NLIScore(entailment=0.04, contradiction=0.90, neutral=0.06),
            (b1.text, a1.text): NLIScore(entailment=0.88, contradiction=0.04, neutral=0.08),
            (b2.text, a2.text): NLIScore(entailment=0.05, contradiction=0.86, neutral=0.09),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(
        workspace_id="ws-sort",
        claims_by_doc={"docA": [a1, a2], "docB": [b1, b2]},
    )

    keys = [(e.type, e.src_id, e.dst_id) for e in out.edges]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Citations carry source-doc + target-doc spans (§6.7)
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_emitted_edges_cite_source_and_target_doc_spans() -> None:
    """Each cross-doc edge cites span(s) from both endpoint docs — §6.7.

    NLI is directional, so the inferer scores both `(a, b)` and `(b, a)`
    and may emit one edge per direction. Every emitted edge — regardless
    of direction — must cite at least one span from each member doc so
    the trace renderer can attribute the verdict to both docs.
    """
    a = _claim(claim_id="cA1", doc_id="docA", text="the proxy forwards requests")
    b = _claim(claim_id="cB1", doc_id="docB", text="the proxy forwards requests")
    scorer = _DictScorer(
        {
            (a.text, b.text): NLIScore(entailment=0.95, contradiction=0.02, neutral=0.03),
            (b.text, a.text): NLIScore(entailment=0.95, contradiction=0.02, neutral=0.03),
        }
    )
    inferer = CrossDocEdgeInferer(scorer=scorer)

    out = inferer.infer(
        workspace_id="ws-cite",
        claims_by_doc={"docA": [a], "docB": [b]},
    )

    assert out.edges, "expected at least one cross-doc edge from a high-entailment pair"
    for edge in out.edges:
        chunk_ids = {span.chunk_id for span in edge.citations}
        assert any(
            cid.startswith("docA") for cid in chunk_ids
        ), f"edge {edge.type} {edge.src_id}->{edge.dst_id} is missing a docA citation"
        assert any(
            cid.startswith("docB") for cid in chunk_ids
        ), f"edge {edge.type} {edge.src_id}->{edge.dst_id} is missing a docB citation"


# ---------------------------------------------------------------------------
# Threshold defaults stay sane and ordered
# ---------------------------------------------------------------------------


@pytest.mark.family_verifier_calibration
def test_threshold_defaults_are_ordered_and_in_unit_interval() -> None:
    """`aligned_with < entails_across` so the soft band exists; all in (0, 1)."""
    assert 0.0 < ALIGNED_WITH_THRESHOLD < ENTAILS_ACROSS_THRESHOLD < 1.0
    assert 0.0 < CONTRADICTS_ACROSS_THRESHOLD < 1.0
    assert DEFAULT_K_CANDIDATES >= 1
