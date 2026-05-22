"""Contract tests for the single-source-of-truth tokenizer.

The whole stack uses one tokenizer for chunking, budget accounting,
and cache-prefix sizing. Drift between callers is forbidden, so this
module exists exactly once.

SPEC-REF: §4.7 (tokenizer)
"""

from __future__ import annotations

import pytest

from ctrldoc import tokenizer


def test_tokenizer_name_is_pinned() -> None:
    assert tokenizer.TOKENIZER_NAME == "cl100k_base"


def test_count_tokens_empty_string_is_zero() -> None:
    assert tokenizer.count_tokens("") == 0


def test_count_tokens_monotonic_with_text_length() -> None:
    short = tokenizer.count_tokens("hello world")
    longer = tokenizer.count_tokens("hello world " * 50)
    assert short > 0
    assert longer > short


def test_encode_decode_round_trip() -> None:
    text = "ctrldoc — verify cl100k_base round-trip with unicode: éφ漢字"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert tokenizer.decode(tokens) == text


def test_count_tokens_matches_encode_length() -> None:
    text = "Pillar 1 — Stateless Tasks. Pillar 2 — Shared Prompt Cache."
    assert tokenizer.count_tokens(text) == len(tokenizer.encode(text))


def test_known_cl100k_short_counts() -> None:
    # These counts are stable for the cl100k_base BPE; they pin the
    # tokenizer identity, so a silent swap to a different encoding
    # would fail this test.
    assert tokenizer.count_tokens("hello") == 1
    assert tokenizer.count_tokens("hello world") == 2


def test_encoding_object_is_cached() -> None:
    a = tokenizer._encoding()
    b = tokenizer._encoding()
    assert a is b


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "the quick brown fox jumps over the lazy dog",
        "数字 + emoji 🚀 + RTL: مرحبا",
        "code: `def foo(x): return x + 1`",
    ],
)
def test_count_tokens_is_non_negative(text: str) -> None:
    assert tokenizer.count_tokens(text) >= 0
