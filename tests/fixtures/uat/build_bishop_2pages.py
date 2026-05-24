"""Deterministic builder for `bishop_2pages.pdf`.

Re-run this script if the committed PDF is missing or its
`pymupdf` build version changes incompatibly. The output is a
small two-page PDF whose first page is a chapter heading + body
paragraph and whose second page is a sub-heading + body paragraph
— a minimal Bishop-style technical-chapter shape used by the
production-hardening UAT ingest-completeness gate.

Usage:

    .venv/bin/python tests/fixtures/uat/build_bishop_2pages.py

The script is idempotent — it overwrites the existing
`bishop_2pages.pdf` with byte-identical content on the same
`pymupdf` version.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

_LINES: list[tuple[str, int, bool]] = [
    ("Chapter 5 Neural Networks", 20, True),
    ("Feed-forward neural networks represent nonlinear functions", 12, False),
    ("from a set of input variables to a set of output variables.", 12, False),
    ("They are controlled by a vector of adjustable parameters.", 12, False),
    ("5.1 Feed-forward Network Functions", 16, True),
    ("The linear models for regression and classification are based", 12, False),
    ("on linear combinations of fixed nonlinear basis functions.", 12, False),
    ("Their analytical and computational properties are well known.", 12, False),
]


def build(path: Path) -> Path:
    """Write the two-page Bishop-style PDF to ``path``."""
    doc = pymupdf.open()
    # Page 1: chapter heading + body.
    page = doc.new_page()
    y = 72.0
    for text, fontsize, bold in _LINES[:4]:
        font = "Helvetica-Bold" if bold else "Helvetica"
        page.insert_text((72, y), text, fontsize=fontsize, fontname=font)
        y += fontsize * 1.6
    # Page 2: section heading + body.
    page = doc.new_page()
    y = 72.0
    for text, fontsize, bold in _LINES[4:]:
        font = "Helvetica-Bold" if bold else "Helvetica"
        page.insert_text((72, y), text, fontsize=fontsize, fontname=font)
        y += fontsize * 1.6
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = Path(__file__).parent / "bishop_2pages.pdf"
    build(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
