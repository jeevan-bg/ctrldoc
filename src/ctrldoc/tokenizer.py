"""Single source of truth for token counts.

Every layer that needs a token count — chunkers, retrieval budgets,
prompt-cache size accounting, evidence-pack assembly — must call into
this module. Drift between two tokenizers in the same process is a
correctness bug.

SPEC-REF: §4.7 (tokenizer)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import tiktoken

TOKENIZER_NAME: Final[str] = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    """Return the number of `cl100k_base` tokens in `text`."""
    if not text:
        return 0
    return len(_encoding().encode(text))


def encode(text: str) -> list[int]:
    """Encode `text` to a list of `cl100k_base` token ids."""
    return _encoding().encode(text)


def decode(tokens: list[int]) -> str:
    """Decode a list of `cl100k_base` token ids back to text."""
    return _encoding().decode(tokens)


__all__ = ["TOKENIZER_NAME", "count_tokens", "decode", "encode"]
