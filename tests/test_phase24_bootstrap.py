"""Phase 24 bootstrap gate.

S-148 opens the Phase 24 production-hardening arc by bumping the
local budget cap to $5.00 and documenting the Homebrew Python 3.11
install command in `README.md`. Both invariants are pinned here so a
future regression that drops either is caught immediately.

The budget assertion reads `.ctrldoc/BUDGET.md` (the workflow file is
gitignored by design — see `.gitignore`). When the file is absent
(fresh clone / CI without the workflow directory) the test skips with
a clear reason; locally and during the arc it is the gate.

SPEC-REF: §9
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

BUDGET_FILE = REPO_ROOT / ".ctrldoc" / "BUDGET.md"
README = REPO_ROOT / "README.md"

EXPECTED_MAX_COST_USD = 5.00
INSTALL_COMMAND = 'python3.11 -m pip install -e ".[dev,index,ingest]"'


# --------------------------------------------------------------------------
# Budget cap
# --------------------------------------------------------------------------


def _parse_max_cost(text: str) -> float | None:
    """Return the `max_cost_usd` value declared in BUDGET.md, or None.

    Matches the row produced by the `Limits` block, tolerant of
    surrounding markdown bullet syntax and the parenthetical note
    after the number.
    """
    pattern = re.compile(
        r"`?max_cost_usd`?\s*:\s*([0-9]+\.?[0-9]*)",
    )
    m = pattern.search(text)
    if m is None:
        return None
    return float(m.group(1))


def test_budget_max_cost_pinned_to_phase24_cap() -> None:
    """`.ctrldoc/BUDGET.md` must declare `max_cost_usd: 5.00` (Phase 24)."""
    if not BUDGET_FILE.exists():
        pytest.skip(
            "BUDGET.md absent — `.ctrldoc/` is gitignored. The cap is a "
            "local workflow invariant; fresh clones skip this gate."
        )
    text = BUDGET_FILE.read_text(encoding="utf-8")
    value = _parse_max_cost(text)
    assert value is not None, (
        "BUDGET.md does not declare a `max_cost_usd` value the parser "
        "can read; check the `Limits` block formatting."
    )
    assert value == pytest.approx(
        EXPECTED_MAX_COST_USD
    ), f"max_cost_usd is {value}, expected {EXPECTED_MAX_COST_USD} (Phase 24 cap per S-148)."


# --------------------------------------------------------------------------
# README install documentation
# --------------------------------------------------------------------------


def test_readme_documents_homebrew_311_install_command() -> None:
    """`README.md` must carry the exact pip command S-148 standardised."""
    text = README.read_text(encoding="utf-8")
    assert INSTALL_COMMAND in text, (
        f"README.md is missing the Phase 24 install command "
        f"`{INSTALL_COMMAND}`; this is the line that produces a "
        f"console-script with a Python 3.11 shebang."
    )


def test_readme_mentions_homebrew_python_311_requirement() -> None:
    """The README must explain *why* 3.11 (sqlite extensions / Homebrew)."""
    text = README.read_text(encoding="utf-8").lower()
    assert "homebrew" in text, (
        "README.md does not mention Homebrew — readers cannot tell why "
        "the install command pins `python3.11` rather than the system "
        "Python."
    )
    assert "3.11" in text, "README.md does not state the 3.11 requirement."
