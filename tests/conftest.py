"""Shared pytest fixtures for ctrldoc tests.

SPEC-REF: §8 (Testing Strategy)
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def synthetic_doc_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "synthetic" / "gold_doc.md"


@pytest.fixture(scope="session")
def synthetic_gold_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "synthetic" / "gold.yaml"
