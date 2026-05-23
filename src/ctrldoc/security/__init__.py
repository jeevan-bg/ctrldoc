"""Security helpers — adversarial detection, sanitisation primitives.

The substrate's guarantee is that the LLM never executes instructions
embedded in source documents. This package provides the small set of
deterministic checks the test suite uses to assert that property
(§8.5 + §8.6 family 8).
"""

from __future__ import annotations

from ctrldoc.security.adversarial import (
    BIDI_OVERRIDE_CODEPOINTS,
    HOMOGLYPH_FOLD,
    PROMPT_INJECTION_PATTERNS,
    ZERO_WIDTH_CODEPOINTS,
    AdversarialMarker,
    contains_bidi_override,
    contains_homoglyphs,
    contains_zero_width,
    detect_adversarial_markers,
    detect_prompt_injection,
    normalize_for_comparison,
)

__all__ = [
    "BIDI_OVERRIDE_CODEPOINTS",
    "HOMOGLYPH_FOLD",
    "PROMPT_INJECTION_PATTERNS",
    "ZERO_WIDTH_CODEPOINTS",
    "AdversarialMarker",
    "contains_bidi_override",
    "contains_homoglyphs",
    "contains_zero_width",
    "detect_adversarial_markers",
    "detect_prompt_injection",
    "normalize_for_comparison",
]
