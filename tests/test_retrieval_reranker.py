"""Contract tests for the cross-encoder reranker.

`Reranker` reduces a candidate list (e.g. top-50 from RRF) to a
shorter top-k by scoring `(query, candidate.text)` pairs jointly.
The contract ships with two reference impls — a passthrough
identity and a lexical (Jaccard) ranker. A production BGE backend
lives behind the same protocol.

SPEC-REF: §4.3 (reranker)
"""

from __future__ import annotations

import pytest

from ctrldoc.retrieval.reranker import (
    Candidate,
    IdentityReranker,
    LexicalReranker,
    Reranker,
)


def _candidates(*pairs: tuple[str, str]) -> list[Candidate]:
    return [Candidate(chunk_id=c, text=t) for c, t in pairs]


# --- protocol conformance ---


def test_identity_satisfies_protocol() -> None:
    assert isinstance(IdentityReranker(), Reranker)


def test_lexical_satisfies_protocol() -> None:
    assert isinstance(LexicalReranker(), Reranker)


# --- Candidate ---


def test_candidate_is_frozen() -> None:
    c = Candidate(chunk_id="c1", text="hello")
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        c.text = "tampered"  # type: ignore[misc]


# --- IdentityReranker ---


def test_identity_preserves_input_order() -> None:
    cands = _candidates(("c1", "alpha"), ("c2", "beta"), ("c3", "gamma"))
    out = IdentityReranker().rerank("query", cands, k=3)
    assert [c[0] for c in out] == ["c1", "c2", "c3"]


def test_identity_truncates_to_k() -> None:
    cands = _candidates(("c1", "a"), ("c2", "b"), ("c3", "c"))
    out = IdentityReranker().rerank("query", cands, k=2)
    assert [c[0] for c in out] == ["c1", "c2"]


def test_identity_k_zero_returns_empty() -> None:
    cands = _candidates(("c1", "a"))
    assert IdentityReranker().rerank("q", cands, k=0) == []


def test_identity_k_negative_rejected() -> None:
    with pytest.raises(ValueError):
        IdentityReranker().rerank("q", _candidates(("c1", "a")), k=-1)


def test_identity_empty_candidates() -> None:
    assert IdentityReranker().rerank("q", [], k=5) == []


# --- LexicalReranker (Jaccard) ---


def test_lexical_higher_overlap_ranks_higher() -> None:
    cands = _candidates(
        ("c-low", "completely unrelated text"),
        ("c-mid", "alpha some other words"),
        ("c-high", "alpha beta gamma matches"),
    )
    out = LexicalReranker().rerank("alpha beta gamma", cands, k=3)
    assert [c[0] for c in out] == ["c-high", "c-mid", "c-low"]


def test_lexical_score_is_jaccard() -> None:
    cands = _candidates(("c1", "alpha beta"))
    out = LexicalReranker().rerank("alpha beta", cands, k=1)
    # All tokens overlap; Jaccard = 2 / 2 = 1.0.
    assert out[0][1] == pytest.approx(1.0)


def test_lexical_zero_overlap_scores_zero() -> None:
    cands = _candidates(("c1", "completely unrelated"))
    out = LexicalReranker().rerank("alpha", cands, k=1)
    assert out[0][1] == pytest.approx(0.0)


def test_lexical_is_case_insensitive() -> None:
    cands = _candidates(("c1", "Alpha Beta"))
    out = LexicalReranker().rerank("alpha beta", cands, k=1)
    assert out[0][1] == pytest.approx(1.0)


def test_lexical_truncates_to_k() -> None:
    cands = _candidates(
        ("c1", "alpha"),
        ("c2", "alpha beta"),
        ("c3", "alpha beta gamma"),
    )
    out = LexicalReranker().rerank("alpha beta gamma", cands, k=2)
    assert len(out) == 2
    # c3 has the highest overlap, then c2.
    assert [c[0] for c in out] == ["c3", "c2"]


def test_lexical_empty_query_returns_zero_scores() -> None:
    cands = _candidates(("c1", "alpha"))
    out = LexicalReranker().rerank("", cands, k=1)
    assert out[0][1] == pytest.approx(0.0)


def test_lexical_empty_candidates() -> None:
    assert LexicalReranker().rerank("anything", [], k=5) == []


def test_lexical_deterministic_tie_break_uses_input_order() -> None:
    cands = _candidates(
        ("c1", "alpha"),
        ("c2", "alpha"),
        ("c3", "alpha"),
    )
    out = LexicalReranker().rerank("alpha", cands, k=3)
    # All identical scores → input order preserved.
    assert [c[0] for c in out] == ["c1", "c2", "c3"]
