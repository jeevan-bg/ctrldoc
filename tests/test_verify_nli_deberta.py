"""Integration tests for the DeBERTa-v3-large-mnli NLI backend.

These tests download a ~750MB cross-encoder model on first run
and skip cleanly when `transformers` (and torch) are not installed.
The model is cached after the first download, so subsequent runs
are fast.

SPEC-REF: §4.4 (verifier step 3 — NLI check)
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("transformers", reason="transformers optional; install ctrldoc[models] to run")
pytest.importorskip("torch", reason="torch optional; install ctrldoc[models] to run")

from ctrldoc.verify.nli import NLIChecker
from ctrldoc.verify.nli_deberta import DeBERTaNLIChecker


@pytest.fixture(scope="module")
def checker() -> DeBERTaNLIChecker:
    return DeBERTaNLIChecker()


@pytest.mark.slow
def test_satisfies_protocol(checker: DeBERTaNLIChecker) -> None:
    assert isinstance(checker, NLIChecker)


@pytest.mark.slow
def test_entailment_detected(checker: DeBERTaNLIChecker) -> None:
    premise = "A dog is running in the park with its owner on a sunny afternoon."
    hypothesis = "A dog is in the park."
    result = checker.check(premise, hypothesis)
    assert result.label == "entailment"
    assert 0.0 <= result.score <= 1.0


@pytest.mark.slow
def test_contradiction_detected(checker: DeBERTaNLIChecker) -> None:
    premise = "All swans observed in this region are white."
    hypothesis = "Some swans in this region are black."
    result = checker.check(premise, hypothesis)
    assert result.label == "contradiction"
    assert 0.0 <= result.score <= 1.0


@pytest.mark.slow
def test_neutral_detected(checker: DeBERTaNLIChecker) -> None:
    premise = "A woman is reading a book in a cafe."
    hypothesis = "The woman is a librarian."
    result = checker.check(premise, hypothesis)
    assert result.label == "neutral"
    assert 0.0 <= result.score <= 1.0


@pytest.mark.slow
def test_score_is_finite_float(checker: DeBERTaNLIChecker) -> None:
    result = checker.check("Paris is the capital of France.", "Paris is in France.")
    assert isinstance(result.score, float)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.slow
def test_empty_hypothesis_is_neutral(checker: DeBERTaNLIChecker) -> None:
    result = checker.check("A premise.", "")
    assert result.label == "neutral"
    assert result.score == 0.0


@pytest.mark.slow
def test_empty_premise_is_neutral(checker: DeBERTaNLIChecker) -> None:
    result = checker.check("", "A hypothesis.")
    assert result.label == "neutral"
    assert result.score == 0.0


@pytest.mark.slow
def test_deterministic_for_same_input(checker: DeBERTaNLIChecker) -> None:
    r1 = checker.check("A dog is running.", "A dog is moving.")
    r2 = checker.check("A dog is running.", "A dog is moving.")
    assert r1 == r2
