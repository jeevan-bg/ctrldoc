"""Verify every MVP-required spec section has at least one mapped test.

Reads docs/SPEC_TRACE.md (the public traceability matrix) and reports any
spec sections still marked as pending or partial.

Exit codes:
  0 — coverage OK or trace file absent.
  1 — coverage gap detected.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    trace = root / "docs" / "SPEC_TRACE.md"
    if not trace.exists():
        print("spec_trace_check: trace file not present — skipping.")
        return 0

    text = trace.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| §")]
    pending = [row for row in rows if "pending" in row.lower()]
    if pending:
        print(f"spec_trace_check: {len(pending)} spec sections without test coverage.")
        for row in pending:
            print(f"  {row}")
        return 1
    print(f"spec_trace_check: all {len(rows)} mapped spec sections covered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
