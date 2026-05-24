"""Parser dispatch routes a source path to the right L0 parser.

`ctrldoc.ingest.parser_dispatch.get_parser(path)` is the single
entry point the CLI ingest path uses to pick a parser by extension:

- `.pdf` → `PDFParser`
- `.md`, `.markdown`, `.txt` → `MarkdownParser`

Unknown extensions raise `UnsupportedSourceExtensionError`. The
helper is case-insensitive so `Doc.PDF` routes the same as `doc.pdf`.

SPEC-REF: §5.1
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.ingest.parser import MarkdownParser, Parser
from ctrldoc.ingest.parser_dispatch import (
    UnsupportedSourceExtensionError,
    get_parser,
)
from ctrldoc.ingest.pdf import PDFParser

# family_referential_integrity — the dispatch surface is part of the
# CLI ↔ pipeline wiring contract.


def test_dispatch_pdf_extension_routes_to_pdf_parser(tmp_path: Path) -> None:
    parser = get_parser(tmp_path / "bishop.pdf")
    assert isinstance(parser, PDFParser)


def test_dispatch_md_extension_routes_to_markdown_parser(tmp_path: Path) -> None:
    parser = get_parser(tmp_path / "notes.md")
    assert isinstance(parser, MarkdownParser)


def test_dispatch_markdown_extension_routes_to_markdown_parser(
    tmp_path: Path,
) -> None:
    parser = get_parser(tmp_path / "spec.markdown")
    assert isinstance(parser, MarkdownParser)


def test_dispatch_txt_extension_routes_to_markdown_parser(tmp_path: Path) -> None:
    parser = get_parser(tmp_path / "plain.txt")
    assert isinstance(parser, MarkdownParser)


def test_dispatch_is_case_insensitive(tmp_path: Path) -> None:
    assert isinstance(get_parser(tmp_path / "Doc.PDF"), PDFParser)
    assert isinstance(get_parser(tmp_path / "Notes.MD"), MarkdownParser)
    assert isinstance(get_parser(tmp_path / "spec.MARKDOWN"), MarkdownParser)
    assert isinstance(get_parser(tmp_path / "plain.TXT"), MarkdownParser)


def test_dispatch_unknown_extension_raises(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedSourceExtensionError) as exc:
        get_parser(tmp_path / "diagram.png")
    assert ".png" in str(exc.value)


def test_dispatch_missing_extension_raises(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedSourceExtensionError):
        get_parser(tmp_path / "README")


def test_dispatch_accepts_string_path(tmp_path: Path) -> None:
    assert isinstance(get_parser(str(tmp_path / "doc.md")), MarkdownParser)
    assert isinstance(get_parser(str(tmp_path / "doc.pdf")), PDFParser)


def test_returned_parsers_satisfy_protocol(tmp_path: Path) -> None:
    assert isinstance(get_parser(tmp_path / "x.md"), Parser)
    assert isinstance(get_parser(tmp_path / "x.pdf"), Parser)
