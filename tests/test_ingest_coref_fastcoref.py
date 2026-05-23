"""Integration tests for the `fastcoref` coreference resolver.

These tests download a ~360MB coref model on first run and skip
cleanly when `fastcoref` (and its transformers + torch deps) are
not installed. The model is cached after the first download, so
subsequent runs are fast.

SPEC-REF: §4.1 (ingest step 2 — coref)
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastcoref", reason="fastcoref optional; install ctrldoc[ingest] to run")

from ctrldoc.ingest.coref import CorefResolver
from ctrldoc.ingest.coref_fastcoref import FastCorefResolver


@pytest.fixture(scope="module")
def resolver() -> FastCorefResolver:
    return FastCorefResolver()


@pytest.mark.slow
def test_satisfies_protocol(resolver: FastCorefResolver) -> None:
    assert isinstance(resolver, CorefResolver)


@pytest.mark.slow
def test_pronoun_resolved_to_canonical_mention(resolver: FastCorefResolver) -> None:
    text = "Sarah went to the store. She bought milk and bread."
    out = resolver.resolve(text)
    # After resolution the pronoun "She" should be rewritten to "Sarah".
    assert "Sarah bought milk" in out


@pytest.mark.slow
def test_empty_text_returns_empty(resolver: FastCorefResolver) -> None:
    assert resolver.resolve("") == ""


@pytest.mark.slow
def test_whitespace_only_returns_passthrough(resolver: FastCorefResolver) -> None:
    text = "   \n  \t"
    assert resolver.resolve(text) == text


@pytest.mark.slow
def test_no_anaphora_returns_unchanged(resolver: FastCorefResolver) -> None:
    text = "The capital of France is Paris. The capital of Germany is Berlin."
    assert resolver.resolve(text) == text


@pytest.mark.slow
def test_multi_cluster_resolution(resolver: FastCorefResolver) -> None:
    text = "Alice met Bob at the conference. He thanked her for the introduction."
    out = resolver.resolve(text)
    # "He" should resolve to "Bob" (canonical proper-noun mention of that cluster);
    # the exact form fastcoref picks for the second cluster varies, but the
    # masculine pronoun should no longer appear as a standalone subject.
    assert "Bob thanked" in out


@pytest.mark.slow
def test_deterministic_for_same_input(resolver: FastCorefResolver) -> None:
    text = "Sarah went to the store. She bought bread."
    assert resolver.resolve(text) == resolver.resolve(text)


@pytest.mark.slow
def test_length_non_decreasing(resolver: FastCorefResolver) -> None:
    """Replacing pronouns with longer canonical mentions never shortens text."""
    text = "Sarah went to the store. She bought milk and bread."
    out = resolver.resolve(text)
    assert len(out) >= len(text)
