"""CLI helpers for the QA subcommand.

Two pieces:

  - `VerifierRetriever` — adapts the shared `BundleRetriever` (from
    `cli_audit.py`) to the verifier's `Retriever` protocol, which
    expects `retrieve(claim_text, depth)` returning a
    `RetrievedEvidence` (text + citation spans). Depth is currently
    a no-op: both `normal` and `broad` route through the same
    bundle retrieval so a follow-up slice can widen `broad` (e.g.
    raise `top_k_after_rerank`) without changing the call sites.

  - `render_qa_markdown(report, ...)` — renders the QA playbook
    output as a Markdown answer + per-claim verification table.
    The table carries verified state, confidence, NLI and judge
    scores, and the chunk-id citations the verifier accepted.

SPEC-REF: §5.1 (QA playbook), §6 (CLI)
"""

from __future__ import annotations

from pathlib import Path

from ctrldoc.cli_audit import BundleRetriever
from ctrldoc.playbooks.qa import AnswerReport
from ctrldoc.verify.claim_verifier import (
    RetrievalDepth,
    RetrievedEvidence,
)


class VerifierRetriever:
    """Adapter — `BundleRetriever` → verifier `Retriever` protocol.

    The verifier hands the claim text in. The bundle retriever
    treats it as a query, runs the planner + executor + RRF +
    reranker, and returns an `EvidencePack`. We flatten its spans
    into the labelled-text format the judge prompt already uses
    (`[chunk_id] text`) and pass the spans through as citations.
    """

    def __init__(self, *, bundle_retriever: BundleRetriever) -> None:
        self._bundle_retriever = bundle_retriever

    def retrieve(self, claim_text: str, *, depth: RetrievalDepth) -> RetrievedEvidence:
        del depth  # documented no-op (see module docstring)
        pack = self._bundle_retriever.retrieve(claim_text)
        if not pack.spans:
            return RetrievedEvidence(text="", citations=[])
        text = "\n\n".join(f"[{s.chunk_id}] {s.text}" for s in pack.spans)
        return RetrievedEvidence(text=text, citations=list(pack.spans))


def render_qa_markdown(
    *,
    report: AnswerReport,
    target_path: Path,
    profile: str,
    run_id: str,
) -> str:
    """Render an `AnswerReport` as a Markdown report.

    Header → answer block → per-claim verification table → an
    appendix listing every cited span. Citations are rendered as
    `[chunk_id]` handles matching the prompt-time labelling.
    """
    lines: list[str] = []
    lines.append("# ctrldoc — QA report")
    lines.append("")
    lines.append(f"- **Query**: {report.query}")
    lines.append(f"- **Target**: `{target_path}`")
    lines.append(f"- **Profile**: `{profile}`")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    answer = report.answer.strip() or "_(empty answer — verifier refused every claim)_"
    lines.append(answer)
    lines.append("")
    lines.append("## Claim verification")
    lines.append("")
    if not report.claims:
        lines.append("_(no claims to verify — generation returned an empty answer)_")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    lines.append("| # | Verified | Conf | NLI | Judge | Citations | Claim |")
    lines.append("|---:|:--:|---:|---:|---:|---|---|")
    for i, claim in enumerate(report.claims, start=1):
        verified_mark = "yes" if claim.verified else "no"
        cites = ", ".join(f"`[{s.chunk_id}]`" for s in claim.citations) or "_(none)_"
        claim_clip = claim.text.replace("|", "\\|").replace("\n", " ").strip()
        if len(claim_clip) > 200:
            claim_clip = claim_clip[:197] + "…"
        lines.append(
            f"| {i} | {verified_mark} | "
            f"{claim.confidence:.2f} | {claim.nli_score:.2f} | "
            f"{claim.judge_score:.2f} | {cites} | {claim_clip} |"
        )
    lines.append("")

    # Appendix: full citation snippets for every claim that retrieved any.
    appendix_claims = [c for c in report.claims if c.citations]
    if appendix_claims:
        lines.append("## Citation snippets")
        lines.append("")
        for i, claim in enumerate(appendix_claims, start=1):
            lines.append(f"### Claim {i} citations")
            lines.append("")
            for span in claim.citations:
                snippet = span.text.strip().replace("\n", " ")
                if len(snippet) > 240:
                    snippet = snippet[:237] + "…"
                lines.append(f"- `[{span.chunk_id}]` {snippet}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "VerifierRetriever",
    "render_qa_markdown",
]
