"""Coreference resolver protocol and identity reference.

Coref rewrites text so pronouns and other anaphoric references point at
their canonical mention ("Claude was created by Anthropic. It is
helpful." → "Claude was created by Anthropic. Claude is helpful.").
The Protocol defines the contract every backend must satisfy. The
`IdentityCorefResolver` reference is a passthrough — useful for unit
tests and as a stand-in for downstream slices until the production
backend (fastcoref) lands.

SPEC-REF: §4.1 (ingest step 2 — coref)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ctrldoc.ingest.parser import ParsedSection


@runtime_checkable
class CorefResolver(Protocol):
    """Map text → text with anaphoric references resolved to canonical mentions."""

    def resolve(self, text: str) -> str: ...


class IdentityCorefResolver:
    """Passthrough resolver — returns the input unchanged.

    Acts as the behavioural oracle for the fastcoref-backed resolver
    (S-034b). Useful in tests and as a stand-in while the production
    backend is unavailable.
    """

    def resolve(self, text: str) -> str:
        return text


def resolve_sections(
    sections: Iterable[ParsedSection],
    resolver: CorefResolver,
) -> list[ParsedSection]:
    """Apply `resolver` to each section's body and return new `ParsedSection`s.

    Structural fields (id, parent_id, title, char range) pass through
    unchanged so the chunker and storage layers see the same tree shape;
    only `text` is rewritten.
    """
    out: list[ParsedSection] = []
    for section in sections:
        resolved_text = resolver.resolve(section.text)
        out.append(section.model_copy(update={"text": resolved_text}))
    return out


__all__ = ["CorefResolver", "IdentityCorefResolver", "resolve_sections"]
