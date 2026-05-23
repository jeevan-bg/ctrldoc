"""Integration tests for the BGE-M3 embedder via Ollama.

These tests hit a real `http://127.0.0.1:11434` Ollama instance
with the `bge-m3` model already pulled. They skip cleanly when
the `ollama` SDK is not installed or no Ollama service is
reachable on the loopback port.

SPEC-REF: §4.1 (ingest step 5 — embed), §4.2 (dense vectors)
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request

import pytest

pytest.importorskip("ollama", reason="ollama optional; install ctrldoc[models] to run")

from ctrldoc.ingest.embedder import Embedder
from ctrldoc.ingest.embedder_ollama import OllamaEmbedder


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _ollama_reachable(), reason="no local Ollama service reachable")


@pytest.fixture(scope="module")
def embedder() -> OllamaEmbedder:
    return OllamaEmbedder()


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_satisfies_protocol(embedder: OllamaEmbedder) -> None:
    assert isinstance(embedder, Embedder)


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_dimension_is_bge_m3_native(embedder: OllamaEmbedder) -> None:
    assert embedder.dimension == 1024


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_embed_returns_unit_vector(embedder: OllamaEmbedder) -> None:
    v = embedder.embed("hello world")
    assert len(v) == embedder.dimension
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_embed_empty_returns_zero_vector(embedder: OllamaEmbedder) -> None:
    v = embedder.embed("")
    assert v == [0.0] * embedder.dimension


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_semantic_similarity_orders_correctly(embedder: OllamaEmbedder) -> None:
    near = embedder.embed("The cat sat on the mat.")
    paraphrase = embedder.embed("A cat is sitting on the mat.")
    off = embedder.embed("Quantum chromodynamics describes the strong force.")

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    sim_paraphrase = cos(near, paraphrase)
    sim_off = cos(near, off)
    assert sim_paraphrase > sim_off + 0.1


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_embed_batch_matches_single(embedder: OllamaEmbedder) -> None:
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    singles = [embedder.embed(t) for t in texts]
    batch = embedder.embed_batch(texts)
    assert len(batch) == len(texts)
    for a, b in zip(singles, batch, strict=True):
        # Pointwise close — Ollama embeddings are deterministic per request.
        assert all(math.isclose(x, y, abs_tol=1e-5) for x, y in zip(a, b, strict=True))


@pytest.mark.slow
@pytest.mark.requires_ollama
def test_embed_batch_empty(embedder: OllamaEmbedder) -> None:
    assert embedder.embed_batch([]) == []
