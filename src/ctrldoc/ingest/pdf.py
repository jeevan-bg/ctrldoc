"""PDF parser — converts a PDF to markdown then reuses MarkdownParser.

`pymupdf4llm.to_markdown` does the heavy lifting (heading detection,
table-to-markdown rendering, list reconstruction). We then feed that
markdown into the existing `MarkdownParser` so chunks/sections share
the same tree shape regardless of source format.

SPEC-REF: §4.1 (ingest step 1)
"""

from __future__ import annotations

from pathlib import Path

import pymupdf4llm  # type: ignore[import-untyped]

from ctrldoc.ingest.parser import MarkdownParser, ParsedSection


class PDFParser:
    """Parse a PDF into the same `ParsedSection` tree the chunker expects."""

    def parse(self, source: str | Path) -> list[ParsedSection]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        markdown = pymupdf4llm.to_markdown(str(path))
        if not markdown.strip():
            return []
        return MarkdownParser().parse(markdown)


__all__ = ["PDFParser"]
