"""Contract tests for the Embedder protocol + deterministic reference.

The protocol pins the surface that every embedding backend (the
production BGE-M3 via Ollama, or any future replacement) must
satisfy. `HashEmbedder` is the reference — deterministic,
dependency-free, dimensional, unit-normalised. Downstream slices
(vector index, retrieval) can compile and test against it without
any network or model dependency.

SPEC-REF: §4.1 (ingest step 5 — embed), §4.2 (dense vectors)
"""

from __future__ import annotations

import math

import pytest

from ctrldoc.ingest.embedder import Embedder, HashEmbedder


def test_hash_embedder_satisfies_protocol() -> None:
    assert isinstance(HashEmbedder(dimension=16), Embedder)


def test_dimension_is_pinned() -> None:
    e = HashEmbedder(dimension=64)
    assert e.dimension == 64


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_dimension_rejected(bad: int) -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimension=bad)


# --- embed ---


def test_embed_returns_correct_dimension() -> None:
    e = HashEmbedder(dimension=32)
    v = e.embed("hello world")
    assert len(v) == 32
    assert all(isinstance(x, float) for x in v)


def test_embed_is_deterministic() -> None:
    e = HashEmbedder(dimension=16)
    assert e.embed("hello") == e.embed("hello")


def test_embed_differs_per_input() -> None:
    e = HashEmbedder(dimension=64)
    assert e.embed("hello") != e.embed("goodbye")


def test_embed_unit_normalised() -> None:
    e = HashEmbedder(dimension=32)
    v = e.embed("any text")
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_embed_empty_string_returns_zero_vector() -> None:
    e = HashEmbedder(dimension=8)
    v = e.embed("")
    assert v == [0.0] * 8


def test_embed_handles_unicode() -> None:
    e = HashEmbedder(dimension=16)
    a = e.embed("héllo 漢字 🚀")
    assert len(a) == 16
    assert any(x != 0.0 for x in a)
    assert e.embed("héllo 漢字 🚀") == a


# --- embed_batch ---


def test_embed_batch_matches_individual() -> None:
    e = HashEmbedder(dimension=16)
    texts = ["one", "two", "three"]
    batched = e.embed_batch(texts)
    individual = [e.embed(t) for t in texts]
    assert batched == individual


def test_embed_batch_empty_returns_empty() -> None:
    assert HashEmbedder(dimension=16).embed_batch([]) == []


def test_embed_batch_preserves_input_order() -> None:
    e = HashEmbedder(dimension=8)
    out = e.embed_batch(["a", "b", "c"])
    assert out[0] == e.embed("a")
    assert out[2] == e.embed("c")


# --- seed ---


def test_different_seeds_produce_different_vectors() -> None:
    a = HashEmbedder(dimension=16, seed=1).embed("hello")
    b = HashEmbedder(dimension=16, seed=2).embed("hello")
    assert a != b
