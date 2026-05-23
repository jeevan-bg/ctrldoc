"""Contract tests for the claim decomposer.

`ClaimDecomposer.decompose(answer)` turns an answer into a list of
atomic claim strings. The heuristic reference splits on sentence
terminators; the Anthropic backend constrains output to JSON.

SPEC-REF: §4.4 (verifier step 1)
"""

from __future__ import annotations

import pytest

from ctrldoc.verify.claim_decomposer import (
    ClaimDecomposer,
    HeuristicClaimDecomposer,
)


def test_heuristic_satisfies_protocol() -> None:
    assert isinstance(HeuristicClaimDecomposer(), ClaimDecomposer)


def test_empty_input_returns_empty_list() -> None:
    assert HeuristicClaimDecomposer().decompose("") == []
    assert HeuristicClaimDecomposer().decompose("   \n  ") == []


def test_single_sentence_returns_one_claim() -> None:
    claims = HeuristicClaimDecomposer().decompose("Aurora uses consistent hashing.")
    assert claims == ["Aurora uses consistent hashing."]


def test_two_sentences_split_into_two_claims() -> None:
    claims = HeuristicClaimDecomposer().decompose(
        "Aurora uses consistent hashing. ShardRing maps keys to nodes."
    )
    assert claims == [
        "Aurora uses consistent hashing.",
        "ShardRing maps keys to nodes.",
    ]


def test_handles_multiple_terminators() -> None:
    claims = HeuristicClaimDecomposer().decompose("Yes. Really? Indeed!")
    assert claims == ["Yes.", "Really?", "Indeed!"]


def test_strips_surrounding_whitespace() -> None:
    claims = HeuristicClaimDecomposer().decompose("  Alpha.   Beta.   \n")
    assert claims == ["Alpha.", "Beta."]


def test_skips_empty_segments() -> None:
    claims = HeuristicClaimDecomposer().decompose("A.    B.")
    assert claims == ["A.", "B."]


def test_unterminated_text_becomes_single_claim() -> None:
    claims = HeuristicClaimDecomposer().decompose("no terminator")
    assert claims == ["no terminator"]


def test_drops_duplicate_adjacent_claims() -> None:
    claims = HeuristicClaimDecomposer().decompose("Alpha. Alpha. Beta.")
    assert claims == ["Alpha.", "Beta."]


def test_deterministic() -> None:
    d = HeuristicClaimDecomposer()
    text = "Alpha. Beta. Gamma."
    assert d.decompose(text) == d.decompose(text)


@pytest.mark.parametrize(
    "text,expected_count",
    [
        ("A. B. C.", 3),
        ("Single statement.", 1),
        ("Question? Answer.", 2),
    ],
)
def test_claim_count_matches_terminators(text: str, expected_count: int) -> None:
    assert len(HeuristicClaimDecomposer().decompose(text)) == expected_count
