"""Aggregate the four cli_smoke audit runs into a single SUMMARY.md.

Reads every `runs/cli_smoke/<NN>/<run_id>/result.json` produced by
`ctrldoc audit --output-dir runs/cli_smoke/<NN>` and writes:

  - `runs/cli_smoke/SUMMARY.md` — table of per-doc verdict counts
    and a grand-total row.

Usage:

  .venv/bin/python scripts/aggregate_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

SMOKE_DIR = Path("runs/cli_smoke")
SUMMARY_PATH = SMOKE_DIR / "SUMMARY.md"

_AUDIT_LABELS = ("Covered", "Partial", "NotCovered", "Ambiguous")


def _load_result(audit_dir: Path) -> dict[str, object] | None:
    """Find the deepest result.json under the audit dir; return its payload."""
    matches = list(audit_dir.rglob("result.json"))
    if not matches:
        return None
    payload: dict[str, object] = json.loads(matches[0].read_text(encoding="utf-8"))
    return payload


def main() -> int:
    audit_dirs = sorted(p for p in SMOKE_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    rows: list[tuple[str, dict[str, object]]] = []
    for audit_dir in audit_dirs:
        result = _load_result(audit_dir)
        if result is None:
            rows.append((audit_dir.name, {"error": "no result.json found"}))
            continue
        rows.append((audit_dir.name, result))

    lines: list[str] = []
    lines.append("# ctrldoc — smoke audit summary")
    lines.append("")
    lines.append("Four `ctrldoc audit --profile thrifty` runs against")
    lines.append("`/Users/jeevan/Downloads/THREAT_MODEL_v1_4.md`, one per")
    lines.append("phase-0 checklist doc. Per-item judging routed through")
    lines.append("the local Qwen2.5-7B Ollama tier; no Opus calls fired.")
    lines.append("")
    lines.append("## Per-audit summary")
    lines.append("")
    lines.append(
        "| # | checklist | items | "
        + " | ".join(_AUDIT_LABELS)
        + " | result.json |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    totals = dict.fromkeys(_AUDIT_LABELS, 0)
    grand_total_items = 0
    for name, result in rows:
        if "error" in result:
            lines.append(
                f"| {name} | _(audit failed)_ | 0 | 0 | 0 | 0 | 0 | "
                f"{result['error']} |"
            )
            continue
        checklist = str(result.get("checklist_path", "")).split("/")[-1]
        summary = result.get("summary", {})
        items_total = int(result.get("items_total", 0))
        grand_total_items += items_total
        row = [
            f"| {name}",
            f"`{checklist}`",
            str(items_total),
        ]
        for label in _AUDIT_LABELS:
            count = int(summary.get(label, 0)) if isinstance(summary, dict) else 0
            totals[label] += count
            row.append(str(count))
        result_path_obj = result.get("run_id", "")
        row.append(f"runs/cli_smoke/{name}/{result_path_obj}/result.json")
        lines.append(" | ".join(row) + " |")
    lines.append(
        f"| **Total** | **all 4** | **{grand_total_items}** | "
        + " | ".join(f"**{totals[label]}**" for label in _AUDIT_LABELS)
        + " | |"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Ambiguous` includes items where Ollama emitted")
    lines.append("  malformed JSON (the sequential runner's fallback)")
    lines.append("  and items the model judged genuinely ambiguous.")
    lines.append("- No Opus calls fired in any run (thrifty profile keeps")
    lines.append("  Opus reserved for synthesis, which `coverage_audit`")
    lines.append("  does not invoke).")
    SUMMARY_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
