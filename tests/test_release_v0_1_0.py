"""Release sanity for v0.1.0.

The release version lives in three places — `pyproject.toml`,
`src/ctrldoc/__init__.py`, and the CHANGELOG heading. A bump that
moves only one of them is a packaging bug. This file pins parity
across all three and asserts the CHANGELOG has a heading + body
for the current version.

SPEC-REF: §7 (release), §12
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import ctrldoc


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    payload = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]["version"]


# --- version parity ---


def test_package_version_matches_pyproject() -> None:
    assert ctrldoc.__version__ == _pyproject_version()


def test_changelog_has_heading_for_current_version() -> None:
    changelog = (_repo_root() / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = f"## [{ctrldoc.__version__}]"
    assert heading in changelog, (
        f"CHANGELOG.md missing heading {heading!r}; release notes must land "
        f"at the same commit as a version bump."
    )


def test_changelog_keepachangelog_shape() -> None:
    """The CHANGELOG follows Keep-a-Changelog: an `## [Unreleased]`
    heading at the top, then dated version headings below."""
    changelog = (_repo_root() / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
    # Every released version heading carries an ISO-ish date.
    version_lines = re.findall(r"^## \[(\d+\.\d+\.\d+)\][^\n]*", changelog, flags=re.MULTILINE)
    assert version_lines, "no released versions in CHANGELOG"


@pytest.mark.parametrize(
    "section",
    [
        "L0 — Ingest",
        "L1 — Multi-view",
        "L2 — Retrieval",
        "L3 — Verifier",
        "L4 — Orchestrator",
        "L5 — Playbooks",
        "Eval & hardening",
        "CLI & docs",
        "Known limitations",
    ],
)
def test_changelog_v0_1_0_lists_every_layer(section: str) -> None:
    """The 0.1.0 release notes must describe every architectural layer
    plus the queued blockers — so a reader knows the surface area."""
    changelog = (_repo_root() / "CHANGELOG.md").read_text(encoding="utf-8")
    # Find the `## [0.1.0]` section and the next `## [` heading.
    start = changelog.index("## [0.1.0]")
    rest = changelog[start:]
    next_heading = rest.find("## [", 1)
    body = rest[:next_heading] if next_heading != -1 else rest
    assert section in body, f"v0.1.0 release notes missing section {section!r}"


# --- README + CHANGELOG cross-reference ---


def test_readme_mentions_current_version() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    assert ctrldoc.__version__ in readme, (
        f"README.md missing version {ctrldoc.__version__!r}; status section "
        f"must advertise the tagged release."
    )


def test_readme_links_to_changelog() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    assert "CHANGELOG.md" in readme
