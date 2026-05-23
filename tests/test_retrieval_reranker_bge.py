"""Integration tests for the BGE-reranker-v2-m3 backend.

These tests download a ~500MB cross-encoder model on first run and
skip cleanly when `transformers` (and torch) are not installed.
The model is cached after the first download, so subsequent runs
are fast.

SPEC-REF: §4.3 (reranker)
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers", reason="transformers optional; install ctrldoc[models] to run")
pytest.importorskip("torch", reason="torch optional; install ctrldoc[models] to run")

from ctrldoc.retrieval.reranker import Candidate, Reranker
from ctrldoc.retrieval.reranker_bge import BGEReranker


@pytest.fixture(scope="module")
def reranker() -> BGEReranker:
    return BGEReranker()


@pytest.mark.slow
def test_satisfies_protocol(reranker: BGEReranker) -> None:
    assert isinstance(reranker, Reranker)


@pytest.mark.slow
def test_higher_relevance_ranks_higher(reranker: BGEReranker) -> None:
    cands = [
        Candidate(chunk_id="c-off", text="The mitochondrion is the powerhouse of the cell."),
        Candidate(
            chunk_id="c-on", text="Paris is the capital city of France and sits on the Seine."
        ),
        Candidate(
            chunk_id="c-near", text="France is a country in western Europe with many regions."
        ),
    ]
    out = reranker.rerank("What is the capital of France?", cands, k=3)
    ranked = [hit[0] for hit in out]
    assert ranked[0] == "c-on"
    # All three hits returned, ordered by relevance.
    assert set(ranked) == {"c-off", "c-on", "c-near"}


@pytest.mark.slow
def test_truncates_to_k(reranker: BGEReranker) -> None:
    cands = [
        Candidate(chunk_id="c1", text="alpha beta gamma"),
        Candidate(chunk_id="c2", text="delta epsilon"),
        Candidate(chunk_id="c3", text="zeta eta"),
    ]
    out = reranker.rerank("alpha", cands, k=2)
    assert len(out) == 2


@pytest.mark.slow
def test_k_zero_returns_empty(reranker: BGEReranker) -> None:
    cands = [Candidate(chunk_id="c1", text="alpha")]
    assert reranker.rerank("query", cands, k=0) == []


@pytest.mark.slow
def test_k_negative_rejected(reranker: BGEReranker) -> None:
    with pytest.raises(ValueError):
        reranker.rerank("q", [Candidate(chunk_id="c1", text="alpha")], k=-1)


@pytest.mark.slow
def test_empty_candidates_returns_empty(reranker: BGEReranker) -> None:
    assert reranker.rerank("anything", [], k=5) == []


@pytest.mark.slow
def test_scores_are_finite_floats(reranker: BGEReranker) -> None:
    import math

    cands = [
        Candidate(chunk_id="c1", text="Paris is the capital of France."),
        Candidate(chunk_id="c2", text="Berlin is the capital of Germany."),
    ]
    out = reranker.rerank("capital of France", cands, k=2)
    assert len(out) == 2
    for _, score in out:
        assert isinstance(score, float)
        assert math.isfinite(score)


@pytest.mark.slow
def test_deterministic_for_same_input(reranker: BGEReranker) -> None:
    cands = [
        Candidate(chunk_id="c1", text="alpha beta gamma"),
        Candidate(chunk_id="c2", text="delta epsilon zeta"),
    ]
    out1 = reranker.rerank("alpha", cands, k=2)
    out2 = reranker.rerank("alpha", cands, k=2)
    assert out1 == out2
