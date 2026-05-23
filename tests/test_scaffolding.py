"""Sanity check that the repository scaffolding loads.

SPEC-REF: §12 (Build Order — Day 1 Checklist)
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_package_imports() -> None:
    import ctrldoc

    assert ctrldoc.__version__ == "0.2.0"


@pytest.mark.family_synthetic_gold
def test_synthetic_fixture_exists(synthetic_doc_path: Path, synthetic_gold_path: Path) -> None:
    assert synthetic_doc_path.exists(), "synthetic gold doc missing"
    assert synthetic_gold_path.exists(), "synthetic gold annotations missing"
    text = synthetic_doc_path.read_text(encoding="utf-8")
    assert len(text) > 1000, "synthetic doc should be substantial"
