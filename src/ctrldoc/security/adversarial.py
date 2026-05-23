# ruff: noqa: RUF001 — this module intentionally enumerates
# ambiguous codepoints (Cyrillic / Greek lookalikes of Latin letters).
# The whole point of the table is that the strings *are* ambiguous.
"""Adversarial detection + sanitisation primitives.

Source documents reaching the substrate can carry attacks that a
naïve consumer would mis-interpret:

  - Zero-width characters (U+200B, U+200C, U+200D, U+FEFF) padded
    into otherwise-benign text to defeat token-overlap matchers.
  - Bidi overrides (U+202A..U+202E, U+2066..U+2069) that flip the
    rendered direction of nearby text without changing tokenisation.
  - Homoglyphs — Cyrillic / Greek lookalikes for Latin letters that
    let an attacker write a "claim" that visually matches evidence
    while sharing no real tokens with it.
  - Prompt-injection strings inside chunks ("ignore previous
    instructions", "system:", "[[INSTRUCTION]]"…) that try to steer
    a downstream LLM out of its role.

The module exposes detectors (returning bool / list[AdversarialMarker])
and a `normalize_for_comparison` helper that strips zero-width
characters and folds the most common homoglyphs to their Latin
counterparts. The detectors are deterministic and used only by
tests and trace logging — they don't gate the production pipeline,
since the substrate's safety story is "never trust source content"
rather than "filter it cleanly."

SPEC-REF: §8.5 (adversarial tests), §8.6 family 8
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ZERO_WIDTH_CODEPOINTS: frozenset[str] = frozenset(
    {
        "​",  # ZERO WIDTH SPACE
        "‌",  # ZERO WIDTH NON-JOINER
        "‍",  # ZERO WIDTH JOINER
        "⁠",  # WORD JOINER
        "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)

BIDI_OVERRIDE_CODEPOINTS: frozenset[str] = frozenset(
    {
        "‪",  # LRE
        "‫",  # RLE
        "‬",  # PDF (pop directional formatting)
        "‭",  # LRO
        "‮",  # RLO — right-to-left override
        "⁦",  # LRI
        "⁧",  # RLI
        "⁨",  # FSI
        "⁩",  # PDI
    }
)

# Fold the most common Cyrillic / Greek lookalikes into Latin. The
# dictionary is intentionally short — exhaustive normalisation is a
# job for `unicodedata`/`confusables`, and we only need to defeat
# the textbook attacks the test suite exercises.
HOMOGLYPH_FOLD: dict[str, str] = {
    # Cyrillic lowercase
    "а": "a",  # U+0430
    "е": "e",  # U+0435
    "о": "o",  # U+043E
    "р": "p",  # U+0440
    "с": "c",  # U+0441
    "у": "y",  # U+0443
    "х": "x",  # U+0445
    "і": "i",  # U+0456 (Ukrainian)
    "ј": "j",  # U+0458
    # Cyrillic uppercase
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "Х": "X",
    # Greek lowercase
    "α": "a",  # U+03B1
    "ο": "o",  # U+03BF
    "ρ": "p",  # U+03C1
    "ν": "v",  # U+03BD
}


PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore (the )?(previous|prior|above) (instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"disregard (the )?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"system\s*[:>]", re.IGNORECASE),
    re.compile(r"\[\[\s*(system|instruction|user)\s*\]\]", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"new instructions?\s*[:>]", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"</?\s*(system|instruction)\s*>", re.IGNORECASE),
)


MarkerKind = Literal[
    "zero_width",
    "bidi_override",
    "homoglyph",
    "prompt_injection",
]


@dataclass(frozen=True)
class AdversarialMarker:
    """One adversarial signal detected in a piece of text."""

    kind: MarkerKind
    excerpt: str
    char_index: int


def contains_zero_width(text: str) -> bool:
    """True iff `text` contains any zero-width or word-joiner codepoint."""
    return any(ch in ZERO_WIDTH_CODEPOINTS for ch in text)


def contains_bidi_override(text: str) -> bool:
    """True iff `text` contains any bidi-override / isolate codepoint."""
    return any(ch in BIDI_OVERRIDE_CODEPOINTS for ch in text)


def contains_homoglyphs(text: str) -> bool:
    """True iff `text` contains any of the catalogued lookalike codepoints."""
    return any(ch in HOMOGLYPH_FOLD for ch in text)


def detect_prompt_injection(text: str) -> list[AdversarialMarker]:
    """Return one marker per matched injection pattern, in input order."""
    markers: list[AdversarialMarker] = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            markers.append(
                AdversarialMarker(
                    kind="prompt_injection",
                    excerpt=match.group(0),
                    char_index=match.start(),
                )
            )
    markers.sort(key=lambda m: m.char_index)
    return markers


def detect_adversarial_markers(text: str) -> list[AdversarialMarker]:
    """Return every adversarial signal — characterwise + pattern — in input order."""
    markers: list[AdversarialMarker] = []
    for index, ch in enumerate(text):
        if ch in ZERO_WIDTH_CODEPOINTS:
            markers.append(AdversarialMarker(kind="zero_width", excerpt=ch, char_index=index))
        elif ch in BIDI_OVERRIDE_CODEPOINTS:
            markers.append(AdversarialMarker(kind="bidi_override", excerpt=ch, char_index=index))
        elif ch in HOMOGLYPH_FOLD:
            markers.append(AdversarialMarker(kind="homoglyph", excerpt=ch, char_index=index))
    markers.extend(detect_prompt_injection(text))
    markers.sort(key=lambda m: m.char_index)
    return markers


def normalize_for_comparison(text: str) -> str:
    """Strip zero-width / bidi codepoints and fold catalogued homoglyphs.

    The result is suitable for comparing two strings for equivalence;
    it is **not** intended to be re-emitted as-is — the substrate's
    contract is that source text is preserved verbatim. Only matchers
    that explicitly want a canonical form should call this.
    """
    out: list[str] = []
    for ch in text:
        if ch in ZERO_WIDTH_CODEPOINTS or ch in BIDI_OVERRIDE_CODEPOINTS:
            continue
        out.append(HOMOGLYPH_FOLD.get(ch, ch))
    return "".join(out)


__all__ = [
    "BIDI_OVERRIDE_CODEPOINTS",
    "HOMOGLYPH_FOLD",
    "PROMPT_INJECTION_PATTERNS",
    "ZERO_WIDTH_CODEPOINTS",
    "AdversarialMarker",
    "MarkerKind",
    "contains_bidi_override",
    "contains_homoglyphs",
    "contains_zero_width",
    "detect_adversarial_markers",
    "detect_prompt_injection",
    "normalize_for_comparison",
]
