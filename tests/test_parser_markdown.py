"""Contract tests for the Parser protocol and Markdown parser.

The parser is the first step of L0 ingest. It reads source bytes and
emits a flat list of `ParsedSection` records that the chunker (S-033)
then breaks into leaf chunks. IDs are deterministic across re-parses
so a re-ingest is idempotent.

SPEC-REF: §4.1 (ingest step 1)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ctrldoc.ingest.parser import MarkdownParser, ParsedSection, Parser


def parse(md: str) -> list[ParsedSection]:
    return MarkdownParser().parse(md)


def test_satisfies_protocol() -> None:
    assert isinstance(MarkdownParser(), Parser)


def test_empty_input_returns_no_sections() -> None:
    assert parse("") == []
    assert parse("   \n\n  \n") == []


def test_text_with_no_headings_becomes_single_preamble() -> None:
    sections = parse("just a paragraph\nwith two lines\n")
    assert len(sections) == 1
    assert sections[0].parent_id is None
    assert "paragraph" in sections[0].text
    assert sections[0].title == ""  # untitled preamble


def test_single_heading_section() -> None:
    md = "# Introduction\n\nIntro body.\n"
    sections = parse(md)
    assert len(sections) == 1
    assert sections[0].title == "Introduction"
    assert sections[0].parent_id is None
    assert "Intro body." in sections[0].text


def test_hierarchy_parents_resolved() -> None:
    md = (
        "# Outer\n\nOuter body.\n\n"
        "## Inner\n\nInner body.\n\n"
        "### Leaf\n\nLeaf body.\n\n"
        "# Sibling\n\nSibling body.\n"
    )
    sections = parse(md)
    titles = [s.title for s in sections]
    parents = {s.title: s.parent_id for s in sections}
    by_title = {s.title: s for s in sections}
    assert titles == ["Outer", "Inner", "Leaf", "Sibling"]
    assert parents["Outer"] is None
    assert parents["Inner"] == by_title["Outer"].id
    assert parents["Leaf"] == by_title["Inner"].id
    assert parents["Sibling"] is None


def test_sibling_at_same_level_has_no_parent_relationship() -> None:
    md = "# A\n\na body\n\n## A1\n\na1 body\n\n## A2\n\na2 body\n"
    sections = parse(md)
    by_title = {s.title: s for s in sections}
    assert by_title["A1"].parent_id == by_title["A"].id
    assert by_title["A2"].parent_id == by_title["A"].id


def test_preamble_section_when_text_precedes_first_heading() -> None:
    md = "leading paragraph\n\n# First\n\nbody\n"
    sections = parse(md)
    assert sections[0].parent_id is None
    assert sections[0].title == ""
    assert "leading paragraph" in sections[0].text
    assert sections[1].title == "First"


def test_section_text_excludes_child_section_bodies() -> None:
    md = "# Outer\n\nouter prose\n\n## Inner\n\ninner prose\n"
    sections = parse(md)
    outer = next(s for s in sections if s.title == "Outer")
    inner = next(s for s in sections if s.title == "Inner")
    assert "outer prose" in outer.text
    assert "inner prose" not in outer.text
    assert "inner prose" in inner.text


def test_char_range_matches_source() -> None:
    md = "# Title\n\nbody.\n"
    section = parse(md)[0]
    assert md[section.char_start : section.char_end].startswith("# Title")


def test_ids_are_deterministic() -> None:
    md = "# A\n\nbody\n\n## B\n\nbody\n"
    first = [s.id for s in parse(md)]
    second = [s.id for s in parse(md)]
    assert first == second


def test_ids_are_unique() -> None:
    md = "# Dup\n\n## Dup\n\n## Dup\n\n# Dup\n"
    sections = parse(md)
    ids = [s.id for s in sections]
    assert len(ids) == len(set(ids))


def test_heading_levels_skip_levels_handled() -> None:
    """`# A` then `### Deep` should still parent Deep under A even though level 2 is skipped."""
    md = "# A\n\nbody\n\n### Deep\n\nbody\n"
    sections = parse(md)
    by_title = {s.title: s for s in sections}
    assert by_title["Deep"].parent_id == by_title["A"].id


def test_heading_with_inline_markdown_keeps_title_text() -> None:
    md = "# **Bold** Title\n\nbody\n"
    section = parse(md)[0]
    # We keep the raw title; downstream layers can render or strip markdown.
    assert section.title == "**Bold** Title"


def test_path_input_is_accepted(tmp_path: Path) -> None:
    md_file = tmp_path / "doc.md"
    md_file.write_text("# Hello\n\nbody.\n", encoding="utf-8")
    sections = MarkdownParser().parse(md_file)
    assert len(sections) == 1
    assert sections[0].title == "Hello"


def test_parsed_section_is_frozen() -> None:
    section = parse("# X\n\nbody")[0]
    with pytest.raises(ValidationError):
        section.title = "tampered"  # type: ignore[misc]
