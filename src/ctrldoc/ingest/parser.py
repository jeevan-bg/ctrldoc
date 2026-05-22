"""Parser protocol and native Markdown parser.

The parser is the first step of L0 ingest. It reads source text and
emits a flat list of `ParsedSection` records — each with a stable
canonical id, the parent id, the heading text, and the body that
belongs to this section *excluding* child-section bodies.

The MD parser is dependency-free; PDF and code parsers (S-031, S-032)
satisfy the same protocol so the chunker can consume any source format
uniformly.

SPEC-REF: §4.1
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, NonNegativeInt

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*\s*$")


class ParsedSection(BaseModel):
    """One node in the parsed document tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    parent_id: str | None
    title: str
    text: str
    char_start: NonNegativeInt
    char_end: NonNegativeInt


@runtime_checkable
class Parser(Protocol):
    """Source-format → list of `ParsedSection`."""

    def parse(self, source: str | Path) -> list[ParsedSection]: ...


class MarkdownParser:
    """Native Markdown parser — splits a document by ATX headings."""

    def parse(self, source: str | Path) -> list[ParsedSection]:
        text = _read_source(source)
        if not text.strip():
            return []

        # Step 1: collect (depth, title, heading_char_start, heading_char_end)
        # for every heading line, plus the character range of the section body.
        headings: list[tuple[int, str, int, int]] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            stripped = line.rstrip("\n").rstrip("\r")
            match = _HEADING_RE.match(stripped)
            if match is not None:
                depth = len(match.group(1))
                title = match.group(2).strip()
                headings.append((depth, title, offset, offset + len(line)))
            offset += len(line)

        if not headings:
            # No headings → one preamble section spanning the whole doc.
            return [
                ParsedSection(
                    id=_make_id("", None, 1),
                    parent_id=None,
                    title="",
                    text=text.strip(),
                    char_start=0,
                    char_end=len(text),
                )
            ]

        # Step 2: walk headings, attaching each body to its heading and
        # resolving parents by depth on a stack.
        results: list[ParsedSection] = []
        stack: list[tuple[int, str]] = []  # (depth, parsed_section_id)
        used_ids: set[str] = set()

        # Preamble: any text before the first heading becomes an untitled root.
        first_heading_start = headings[0][2]
        preamble_text = text[:first_heading_start].strip()
        if preamble_text:
            preamble_id = _allocate_id(used_ids, "", None, 1)
            results.append(
                ParsedSection(
                    id=preamble_id,
                    parent_id=None,
                    title="",
                    text=preamble_text,
                    char_start=0,
                    char_end=first_heading_start,
                )
            )

        for idx, (depth, title, heading_start, body_start) in enumerate(headings):
            # Body runs from after the heading line to the next heading start
            # (or EOF).
            next_heading_start = headings[idx + 1][2] if idx + 1 < len(headings) else len(text)
            body = text[body_start:next_heading_start].strip()

            # Resolve parent: pop the stack until we find a depth strictly
            # less than this heading's depth.
            while stack and stack[-1][0] >= depth:
                stack.pop()
            parent_id = stack[-1][1] if stack else None
            section_id = _allocate_id(used_ids, title, parent_id, depth)

            results.append(
                ParsedSection(
                    id=section_id,
                    parent_id=parent_id,
                    title=title,
                    text=body,
                    char_start=heading_start,
                    char_end=next_heading_start,
                )
            )
            stack.append((depth, section_id))

        return results


def _read_source(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    return source


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _make_id(title: str, parent_id: str | None, depth: int) -> str:
    base = _slugify(title) if title else "preamble"
    if parent_id is None:
        return f"sec/{base}"
    return f"{parent_id}/{base}"


def _allocate_id(used: set[str], title: str, parent_id: str | None, depth: int) -> str:
    base = _make_id(title, parent_id, depth)
    candidate = base
    n = 2
    while candidate in used:
        candidate = f"{base}-{n}"
        n += 1
    used.add(candidate)
    return candidate


__all__ = ["MarkdownParser", "ParsedSection", "Parser"]
