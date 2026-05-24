"""Source-extension → `Parser` routing for the L0 ingest path.

The CLI ingest entry point cannot pin a single parser any longer:
production-hardening UAT exercises both Markdown notes and PDF
chapters via the same `ctrldoc ingest` command. `get_parser(path)`
returns the right parser for the source's file extension, keeping
the dispatch one-pass and the unknown-extension surface explicit
(no silent fallback to Markdown).

Routing table:

- ``.pdf``                          → :class:`PDFParser`
- ``.md``, ``.markdown``, ``.txt``  → :class:`MarkdownParser`

Extension matching is case-insensitive. Unknown extensions raise
:class:`UnsupportedSourceExtensionError` so callers fail loudly at
ingest time rather than silently producing an empty section tree.

SPEC-REF: §5.1
"""

from __future__ import annotations

from pathlib import Path

from ctrldoc.ingest.parser import MarkdownParser, Parser
from ctrldoc.ingest.pdf import PDFParser


class UnsupportedSourceExtensionError(ValueError):
    """Raised when no parser is registered for a source path's extension."""


_MARKDOWN_EXTS: frozenset[str] = frozenset({".md", ".markdown", ".txt"})
_PDF_EXTS: frozenset[str] = frozenset({".pdf"})


def get_parser(path: str | Path) -> Parser:
    """Return the parser registered for ``path``'s file extension.

    Both ``str`` and :class:`pathlib.Path` inputs are accepted. The
    file does not need to exist on disk — only the suffix is read.
    """
    suffix = Path(path).suffix.lower()
    if not suffix:
        raise UnsupportedSourceExtensionError(
            f"source path has no extension: {path!r}; "
            f"supported extensions: {sorted(_PDF_EXTS | _MARKDOWN_EXTS)}"
        )
    if suffix in _PDF_EXTS:
        return PDFParser()
    if suffix in _MARKDOWN_EXTS:
        return MarkdownParser()
    raise UnsupportedSourceExtensionError(
        f"no parser registered for extension {suffix!r}; "
        f"supported extensions: {sorted(_PDF_EXTS | _MARKDOWN_EXTS)}"
    )


__all__ = ["UnsupportedSourceExtensionError", "get_parser"]
