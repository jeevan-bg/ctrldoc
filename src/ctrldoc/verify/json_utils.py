"""Shared JSON-prelude helpers for LLM-backed wrappers.

Models constrained to "JSON only" still occasionally wrap their
output in a Markdown code fence (` ```json ... ``` ` or a bare
` ``` ... ``` ` block). The helpers below normalise that drift
before the caller hands the body to `json.loads`. The contract is
deliberately conservative — anything that does not start with a
fence is returned unchanged, so a refusal or hallucinated prose
flows through to the caller's parse step and surfaces the right
error rather than getting silently mangled.

SPEC-REF: §6.5
"""

from __future__ import annotations

_FENCE = "```"


def strip_code_fence(text: str) -> str:
    """Return the inner body of a fenced block, or the input unchanged.

    The accepted shapes are:

    - ``\\`\\`\\`json\\n<body>\\n\\`\\`\\``` — language-tagged fence.
    - ``\\`\\`\\`\\n<body>\\n\\`\\`\\``` — bare fence.
    - ``\\`\\`\\`json\\n<body>`` — leading fence without a trailing fence.

    Surrounding whitespace is trimmed. Idempotent: applying the
    helper twice yields the same body as one application.
    """
    stripped = text.strip()
    if not stripped.startswith(_FENCE):
        return stripped
    # Drop the opening fence (plus an optional language tag) up to the
    # first newline. If there is no newline at all the body is empty.
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.endswith(_FENCE):
        body = body[: -len(_FENCE)]
    return body.strip()


__all__ = ["strip_code_fence"]
