"""CLI helpers for the anomaly-scan subcommand.

`render_scan_markdown(queue, target_path, profile, run_id)` renders
the `AnomalyScanPlaybook` output as a Markdown triage queue grouped
by detector, sorted within each group by severity
(critical → warn → info). A summary table at the foot counts
findings per detector and per severity.

`AnomalyScanPlaybook` itself is dependency-light (just
`HedgeWordDetector` + `EmptySummaryDetector` in the MVP) and is
fully deterministic — the scan subcommand works in any profile,
including `heuristic`.

SPEC-REF: §5.5 (UC5 anomaly_scan), §6 (CLI)
"""

from __future__ import annotations

from pathlib import Path

from ctrldoc.models import Finding, SeverityLiteral
from ctrldoc.playbooks.anomaly import AnomalyQueue

_SEVERITY_ORDER: tuple[SeverityLiteral, ...] = ("critical", "warn", "info")


def render_scan_markdown(
    *,
    queue: AnomalyQueue,
    target_path: Path,
    profile: str,
    run_id: str,
) -> str:
    """Render an `AnomalyQueue` as a Markdown triage report.

    Layout: header → per-detector groups → summary table.
    Findings inside each detector group are sorted by severity
    (critical first), then by chunk_id for stable ordering.
    """
    findings_by_detector: dict[str, list[Finding]] = {}
    for finding in queue.findings:
        findings_by_detector.setdefault(finding.ctrldoc, []).append(finding)
    for bucket in findings_by_detector.values():
        bucket.sort(key=lambda f: (_SEVERITY_ORDER.index(f.severity), f.location.chunk_id))

    lines: list[str] = []
    lines.append("# ctrldoc — anomaly scan report")
    lines.append("")
    lines.append(f"- **Target**: `{target_path}`")
    lines.append(f"- **Profile**: `{profile}`")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append(f"- **Total findings**: {len(queue.findings)}")
    lines.append("")

    if not findings_by_detector:
        lines.append("_(no anomalies detected)_")
        lines.append("")
    else:
        for detector_name in sorted(findings_by_detector):
            bucket = findings_by_detector[detector_name]
            lines.append(f"## Detector `{detector_name}` ({len(bucket)})")
            lines.append("")
            for finding in bucket:
                snippet = finding.location.text.strip().replace("\n", " ")
                if len(snippet) > 240:
                    snippet = snippet[:237] + "…"
                lines.append(
                    f"- **{finding.severity}** — {finding.claim} "
                    f"`[{finding.location.chunk_id}]` {snippet}"
                )
            lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Detector | critical | warn | info | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    if findings_by_detector:
        crit_total = warn_total = info_total = 0
        for detector_name in sorted(findings_by_detector):
            bucket = findings_by_detector[detector_name]
            crit = sum(1 for f in bucket if f.severity == "critical")
            warn = sum(1 for f in bucket if f.severity == "warn")
            info = sum(1 for f in bucket if f.severity == "info")
            crit_total += crit
            warn_total += warn
            info_total += info
            lines.append(f"| `{detector_name}` | {crit} | {warn} | {info} | {len(bucket)} |")
        lines.append(
            f"| **Total** | **{crit_total}** | **{warn_total}** | **{info_total}** | "
            f"**{len(queue.findings)}** |"
        )
    else:
        lines.append("| _(none)_ | 0 | 0 | 0 | 0 |")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_scan_markdown"]
