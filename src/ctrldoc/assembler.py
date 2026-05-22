"""Doc-skeleton + entity-glossary assembler.

Produces the cacheable prefix `{system_prompt, doc_skeleton,
entity_glossary}` that rides on every sub-task. Output is fully
deterministic so the Anthropic prompt cache keys on the same byte
sequence across N parallel sub-tasks.

SPEC-REF: §3.1 (cacheable prefix), §4.2 (skeleton + glossary), §4.1
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ctrldoc.models import Section
from ctrldoc.store import Store

_MAX_HEADING_DEPTH = 6  # markdown supports `#`..`######`


class CacheablePrefix(BaseModel):
    """The three-part prefix that every sub-task shares.

    `render()` concatenates the parts into the single text blob handed
    to the Anthropic API. Splitting them on this model lets the
    orchestrator surface counts and cache-control markers without
    re-parsing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str
    doc_skeleton: str
    entity_glossary: str

    def render(self) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt.rstrip())
        if self.doc_skeleton:
            parts.append("# Document skeleton\n\n" + self.doc_skeleton.rstrip())
        if self.entity_glossary:
            parts.append("# Entity glossary\n\n" + self.entity_glossary.rstrip())
        return "\n\n".join(parts) + ("\n" if parts else "")


def assemble_skeleton(store: Store) -> str:
    """Render the section tree as a depth-prefixed markdown outline.

    Sections that point at a non-existent parent are dropped — we
    don't fabricate parents on the fly. Sibling order follows the
    order returned by `store.iter_sections()`, which on every backend
    is insertion order.
    """
    sections = list(store.iter_sections())
    if not sections:
        return ""

    by_id: dict[str, Section] = {s.id: s for s in sections}
    children: dict[str | None, list[Section]] = {}
    for section in sections:
        # Skip orphans whose parent_id refers to a non-existent section.
        if section.parent_id is not None and section.parent_id not in by_id:
            continue
        children.setdefault(section.parent_id, []).append(section)

    lines: list[str] = []

    def visit(section: Section, depth: int) -> None:
        heading_level = min(depth + 1, _MAX_HEADING_DEPTH)
        lines.append(f"{'#' * heading_level} {section.title}")
        if section.summary:
            lines.append(section.summary)
        lines.append("")
        for child in children.get(section.id, []):
            visit(child, depth + 1)

    for root in children.get(None, []):
        visit(root, 0)

    return "\n".join(lines).rstrip() + "\n"


def assemble_glossary(store: Store) -> str:
    """Render the entity glossary as a sorted, deterministic text block."""
    entities = sorted(store.iter_entities(), key=lambda e: e.id)
    if not entities:
        return ""
    lines: list[str] = []
    for entity in entities:
        if entity.aliases:
            aliases = ", ".join(entity.aliases)
            lines.append(f"- **{entity.id}** [{entity.type}] — aliases: {aliases}")
        else:
            lines.append(f"- **{entity.id}** [{entity.type}]")
    return "\n".join(lines) + "\n"


def assemble_cacheable_prefix(store: Store, *, system_prompt: str) -> CacheablePrefix:
    """Bundle the system prompt + skeleton + glossary into one frozen record."""
    return CacheablePrefix(
        system_prompt=system_prompt,
        doc_skeleton=assemble_skeleton(store),
        entity_glossary=assemble_glossary(store),
    )


__all__ = [
    "CacheablePrefix",
    "assemble_cacheable_prefix",
    "assemble_glossary",
    "assemble_skeleton",
]
