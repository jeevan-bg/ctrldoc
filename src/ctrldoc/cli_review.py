"""CLI helpers for the analytical-review subcommand.

Two pieces:

  - `LLMLensSweeper` — implements the `LensSweeper` protocol against
    a `BundleRetriever`. For each lens, it retrieves an evidence pack
    keyed on the lens name + description, then asks the bundle's
    local `TaskClient` (Ollama Qwen in thrifty) to enumerate issues
    visible through that lens. The model returns
    `{"findings": [{claim, severity, citation_chunk_id}]}`; the
    sweeper resolves each citation back to the corresponding span
    in the evidence pack and emits a `Finding` tagged with the
    lens id.

  - `render_review_markdown(report, ...)` — renders a
    `ReviewReport` as a Markdown document grouped by lens, with the
    synthesis narrative up front and per-finding citation snippets.

SPEC-REF: §5.4 (analytical review), §6 (CLI)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.cli_audit import BundleRetriever
from ctrldoc.models import Finding, SeverityLiteral
from ctrldoc.orch.task import StatelessTaskRunner, TaskInput
from ctrldoc.playbooks.review import Lens, ReviewReport

_LENS_SWEEPER_SYSTEM_PROMPT = (
    "You are a strict analytical reviewer. Given a LENS and EVIDENCE "
    "spans from the document, enumerate concrete issues visible through "
    "that lens. Return one JSON object of shape:\n"
    '  {"findings": [{"claim": "<one-sentence issue>",\n'
    '                 "severity": "info"|"warn"|"critical",\n'
    '                 "citation_chunk_id": "<chunk_id copied from EVIDENCE>"}]}\n\n'
    "Each finding must cite exactly one chunk_id that appears in the "
    "EVIDENCE. If no issues are visible through the lens, return "
    '{"findings": []}. No prose outside the JSON object."'
)


class _SweptFinding(BaseModel):
    """One row the model emits per finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    severity: SeverityLiteral
    citation_chunk_id: str


class _SweptFindings(BaseModel):
    """Container — the model returns `{"findings": [...]}` per lens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: list[_SweptFinding] = Field(default_factory=list)


class LLMLensSweeper:
    """`LensSweeper` backed by a `BundleRetriever` + `TaskClient`.

    One retrieval + one task call per lens. Each call sees only the
    cacheable prefix and the rendered evidence pack — never the raw
    doc.
    """

    def __init__(
        self,
        *,
        prefix: CacheablePrefix,
        retriever: BundleRetriever,
        task_runner: StatelessTaskRunner,
    ) -> None:
        self._prefix = prefix
        self._retriever = retriever
        self._task_runner = task_runner

    def sweep(self, lens: Lens) -> list[Finding]:
        # Use just the lens name as the retrieval query; the description
        # goes into the task input. Avoids `:` which Tantivy parses as a
        # field separator in BM25 queries.
        pack = self._retriever.retrieve(lens.name)
        if not pack.spans:
            return []
        evidence_text = "\n\n".join(f"[{s.chunk_id}] {s.text}" for s in pack.spans)
        span_by_id = {s.chunk_id: s for s in pack.spans}
        task_input = (
            f"LENS:\n  id: {lens.id}\n  name: {lens.name}\n  description: {lens.description}\n"
        )
        task = TaskInput(
            prefix=self._prefix,
            evidence_pack=evidence_text,
            task_input=task_input,
        )
        result = self._task_runner.run(task, output_model=_SweptFindings)
        findings: list[Finding] = []
        for row in result.findings:
            span = span_by_id.get(row.citation_chunk_id)
            if span is None:
                continue  # model hallucinated a chunk_id; drop it
            findings.append(
                Finding(
                    ctrldoc=lens.id,
                    location=span,
                    claim=row.claim,
                    severity=row.severity,
                )
            )
        return findings


# --- review report renderer ---


_SEVERITY_ORDER: tuple[SeverityLiteral, ...] = ("critical", "warn", "info")


def render_review_markdown(
    *,
    report: ReviewReport,
    target_path: Path,
    profile: str,
    run_id: str,
) -> str:
    """Render an `AnalyticalReviewPlaybook` `ReviewReport` as Markdown.

    Layout: header → synthesis narrative (headline + summary +
    structured sections) → per-lens groups with each lens's
    findings ordered by severity (critical → warn → info) → a
    summary table at the foot.
    """
    findings_by_lens: dict[str, list[Finding]] = {}
    for finding in report.findings:
        findings_by_lens.setdefault(finding.ctrldoc, []).append(finding)
    for bucket in findings_by_lens.values():
        bucket.sort(key=lambda f: _SEVERITY_ORDER.index(f.severity))

    lines: list[str] = []
    lines.append("# ctrldoc — analytical review")
    lines.append("")
    lines.append(f"- **Document type**: {report.doc_type}")
    lines.append(f"- **Target**: `{target_path}`")
    lines.append(f"- **Profile**: `{profile}`")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append("")
    lines.append("## Narrative")
    lines.append("")
    if report.narrative.headline:
        lines.append(f"### {report.narrative.headline}")
        lines.append("")
    if report.narrative.summary:
        lines.append(report.narrative.summary)
        lines.append("")
    if report.narrative.sections:
        lines.append("### Synthesised sections")
        lines.append("")
        for i, section in enumerate(report.narrative.sections, start=1):
            lines.append(f"{i}. {section}")
        lines.append("")
    if not any((report.narrative.headline, report.narrative.summary, report.narrative.sections)):
        lines.append("_(synthesis returned an empty narrative)_")
        lines.append("")

    lines.append("## Findings by lens")
    lines.append("")
    if not findings_by_lens:
        lines.append("_(no findings)_")
        lines.append("")
    else:
        for lens_id in sorted(findings_by_lens):
            bucket = findings_by_lens[lens_id]
            lines.append(f"### `{lens_id}` ({len(bucket)})")
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
    lines.append("| Lens | Findings |")
    lines.append("|---|---:|")
    if findings_by_lens:
        for lens_id in sorted(findings_by_lens):
            lines.append(f"| `{lens_id}` | {len(findings_by_lens[lens_id])} |")
        lines.append(f"| **Total** | **{sum(len(v) for v in findings_by_lens.values())}** |")
    else:
        lines.append("| _(none)_ | 0 |")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "LLMLensSweeper",
    "render_review_markdown",
]
