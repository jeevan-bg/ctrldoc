"""Contract tests for the NLI checker.

`NLIChecker.check(premise, hypothesis)` returns an `NLIResult` with
`label ∈ {entailment, neutral, contradiction}` and a score in
`[0, 1]`. The MVP ships a deterministic token-overlap heuristic; the
DeBERTa backend lands in S-051b.

SPEC-REF: §4.4 (verifier step 3 — NLI check)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.verify.nli import (
    HeuristicNLIChecker,
    NLIChecker,
    NLILabel,
    NLIResult,
)


def test_heuristic_satisfies_protocol() -> None:
    assert isinstance(HeuristicNLIChecker(), NLIChecker)


# --- NLIResult ---


def test_result_is_frozen() -> None:
    r = NLIResult(label="entailment", score=0.9)
    with pytest.raises(ValidationError):
        r.label = "neutral"  # type: ignore[misc]


def test_result_score_bounded() -> None:
    with pytest.raises(ValidationError):
        NLIResult(label="entailment", score=1.1)
    with pytest.raises(ValidationError):
        NLIResult(label="entailment", score=-0.1)


@pytest.mark.parametrize("label", ["entailment", "neutral", "contradiction"])
def test_result_accepts_all_three_labels(label: NLILabel) -> None:
    NLIResult(label=label, score=0.5)


def test_result_unknown_label_rejected() -> None:
    with pytest.raises(ValidationError):
        NLIResult(label="maybe", score=0.5)  # type: ignore[arg-type]


# --- HeuristicNLIChecker ---


def test_subset_hypothesis_returns_entailment() -> None:
    result = HeuristicNLIChecker().check(
        premise="Aurora uses consistent hashing for routing.",
        hypothesis="Aurora uses consistent hashing.",
    )
    assert result.label == "entailment"
    assert result.score == pytest.approx(1.0)


def test_partial_overlap_returns_neutral() -> None:
    result = HeuristicNLIChecker().check(
        premise="Aurora uses consistent hashing.",
        hypothesis="Aurora supports transactions.",
    )
    assert result.label == "neutral"


def test_zero_overlap_returns_neutral_with_zero_score() -> None:
    result = HeuristicNLIChecker().check(
        premise="Aurora distributes keys across nodes.",
        hypothesis="Completely unrelated sentence here.",
    )
    assert result.label == "neutral"
    assert result.score == pytest.approx(0.0)


def test_empty_hypothesis_returns_neutral_with_zero() -> None:
    result = HeuristicNLIChecker().check(premise="any premise", hypothesis="")
    assert result.label == "neutral"
    assert result.score == pytest.approx(0.0)


def test_empty_premise_returns_neutral_with_zero() -> None:
    result = HeuristicNLIChecker().check(premise="", hypothesis="hypothesis")
    assert result.label == "neutral"
    assert result.score == pytest.approx(0.0)


def test_score_is_token_overlap_fraction() -> None:
    # 2 of 3 hypothesis tokens overlap with the premise.
    result = HeuristicNLIChecker().check(
        premise="alpha beta gamma",
        hypothesis="alpha beta delta",
    )
    assert result.label == "neutral"
    assert result.score == pytest.approx(2 / 3)


def test_case_insensitive_overlap() -> None:
    result = HeuristicNLIChecker().check(
        premise="Alpha Beta Gamma",
        hypothesis="alpha beta",
    )
    assert result.label == "entailment"
    assert result.score == pytest.approx(1.0)


def test_determinism() -> None:
    nli = HeuristicNLIChecker()
    args = ("alpha beta gamma", "alpha beta")
    assert nli.check(*args) == nli.check(*args)


def test_entailment_threshold_is_configurable() -> None:
    # Default threshold is 0.999; a custom threshold lets partial overlap entail.
    result = HeuristicNLIChecker(entailment_threshold=0.5).check(
        premise="alpha beta gamma delta",
        hypothesis="alpha beta gamma",
    )
    assert result.label == "entailment"


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        HeuristicNLIChecker(entailment_threshold=-0.1)
    with pytest.raises(ValueError):
        HeuristicNLIChecker(entailment_threshold=1.1)
