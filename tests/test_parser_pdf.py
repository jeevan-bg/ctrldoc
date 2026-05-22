"""Contract tests for the PDF parser.

PDFParser converts a PDF to markdown via `pymupdf4llm.to_markdown`
and then reuses the Markdown parser to extract a section tree.
Tests build small PDFs on the fly so no binary fixtures are
committed to the repo.

SPEC-REF: §4.1 (ingest step 1)
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ctrldoc.ingest.parser import Parser
from ctrldoc.ingest.pdf import PDFParser


def _build_pdf(
    path: Path, lines: list[tuple[str, int, bool]], *, page_break_on_heading: bool = False
) -> Path:
    """Write a PDF whose pages contain the given (text, fontsize, bold) lines.

    Each tuple becomes one text line, top-down. `bold` toggles a bold
    font so pymupdf4llm picks the line up as a heading. When
    `page_break_on_heading` is set, every bold line starts a new page —
    that gives pymupdf4llm's font-size statistics a clearer signal.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    for text, fontsize, bold in lines:
        if page_break_on_heading and bold and y > 72.0:
            page = doc.new_page()
            y = 72.0
        font = "Helvetica-Bold" if bold else "Helvetica"
        page.insert_text((72, y), text, fontsize=fontsize, fontname=font)
        y += fontsize * 1.6
    doc.save(path)
    doc.close()
    return path


def test_satisfies_protocol() -> None:
    assert isinstance(PDFParser(), Parser)


def test_single_heading_pdf(tmp_path: Path) -> None:
    pdf = _build_pdf(
        tmp_path / "doc.pdf",
        [
            ("Sample Heading", 18, True),
            ("This is the body text for the heading.", 12, False),
        ],
    )
    sections = PDFParser().parse(pdf)
    assert len(sections) >= 1
    titles = [s.title for s in sections]
    # pymupdf4llm wraps bold headings in `**...**`; we keep that literally.
    assert any("Sample Heading" in t for t in titles)


def test_multi_section_pdf_keeps_tree_shape(tmp_path: Path) -> None:
    pdf = _build_pdf(
        tmp_path / "doc.pdf",
        [
            ("Outer Title", 28, True),
            ("Outer body line one.", 10, False),
            ("Outer body line two.", 10, False),
            ("Inner Title", 20, True),
            ("Inner body line one.", 10, False),
            ("Inner body line two.", 10, False),
        ],
        page_break_on_heading=True,
    )
    sections = PDFParser().parse(pdf)
    # We expect at least two sections, all reachable through parent links.
    assert len(sections) >= 2
    ids = {s.id for s in sections}
    for s in sections:
        if s.parent_id is not None:
            assert s.parent_id in ids


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PDFParser().parse(tmp_path / "does_not_exist.pdf")


def test_str_path_accepted(tmp_path: Path) -> None:
    pdf = _build_pdf(
        tmp_path / "doc.pdf",
        [
            ("Heading", 18, True),
            ("Body.", 12, False),
        ],
    )
    sections = PDFParser().parse(str(pdf))
    assert len(sections) >= 1


def test_empty_pdf_returns_no_sections(tmp_path: Path) -> None:
    pdf = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    sections = PDFParser().parse(pdf)
    # A page with no visible text should not invent sections.
    assert sections == []


def test_deterministic_across_calls(tmp_path: Path) -> None:
    pdf = _build_pdf(
        tmp_path / "doc.pdf",
        [
            ("A Heading", 24, True),
            ("first paragraph", 10, False),
            ("B Heading", 24, True),
            ("second paragraph", 10, False),
        ],
        page_break_on_heading=True,
    )
    first = [(s.id, s.title) for s in PDFParser().parse(pdf)]
    second = [(s.id, s.title) for s in PDFParser().parse(pdf)]
    assert first == second
