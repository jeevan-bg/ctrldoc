"""UC2 — coverage audit playbook.

Each `ChecklistItem` carries a `topic_key`. Items sharing a key form
a cluster: one retrieval, one batched judge call (§5.2). The batch
emits one `Verdict` per item (`Covered`, `Partial`, `NotCovered`,
`Ambiguous`) with citation chunk ids; the playbook resolves those
ids back to `Span` records from the cluster's evidence pack. The
returned `CoverageReport` preserves the caller's input order so the
verdict list lines up with the checklist.

The clustering is driven by the caller — usually a pre-processing
step that uses topic embeddings or an LLM pass to group items. The
playbook itself just iterates whatever `topic_key`s appear.

SPEC-REF: §5.2
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import EvidencePack, Span, Verdict, VerdictLiteral
from ctrldoc.orch.batch import BatchedTaskInput, BatchedTaskRunner, BatchItem


class ChecklistItem(BaseModel):
    """One requirement to evaluate against the target doc."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str
    topic_key: str


class CoverageReport(BaseModel):
    """Aggregate verdicts in original checklist order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdicts: list[Verdict]


@runtime_checkable
class CoverageRetriever(Protocol):
    """Fetches the evidence pack for one topic cluster."""

    def retrieve(self, topic_key: str) -> EvidencePack: ...


_VerdictLabel = Literal["Covered", "Partial", "NotCovered", "Ambiguous"]


class _BatchedVerdict(BaseModel):
    """One verdict the model emits in the batched judging step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: _VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    citation_chunk_ids: list[str]


class CoverageAuditPlaybook:
    """Cluster items by `topic_key`, batch-judge each cluster, aggregate."""

    def __init__(
        self,
        *,
        prefix: CacheablePrefix,
        retriever: CoverageRetriever,
        batched_runner: BatchedTaskRunner,
    ) -> None:
        self._prefix = prefix
        self._retriever = retriever
        self._batched_runner = batched_runner

    def run(self, items: list[ChecklistItem]) -> CoverageReport:
        if not items:
            return CoverageReport(verdicts=[])

        seen_ids: set[str] = set()
        for item in items:
            if item.id in seen_ids:
                raise ValueError(f"duplicate checklist item id: {item.id!r}")
            seen_ids.add(item.id)

        # First-appearance order keeps the retrieval call sequence deterministic.
        clusters: dict[str, list[ChecklistItem]] = {}
        for item in items:
            clusters.setdefault(item.topic_key, []).append(item)

        verdicts_by_id: dict[str, Verdict] = {}
        for topic_key, cluster_items in clusters.items():
            pack = self._retriever.retrieve(topic_key)
            evidence_text = _render_evidence(pack)
            batched_input = BatchedTaskInput(
                prefix=self._prefix,
                evidence_pack=evidence_text,
                items=[BatchItem(id=item.id, task_input=item.text) for item in cluster_items],
            )
            results = self._batched_runner.run(batched_input, output_model=_BatchedVerdict)
            for item, result in zip(cluster_items, results, strict=True):
                verdicts_by_id[item.id] = Verdict(
                    item_id=item.id,
                    verdict=_as_verdict_literal(result.verdict),
                    citations=_resolve_citations(result.citation_chunk_ids, pack),
                    confidence=result.confidence,
                )

        return CoverageReport(verdicts=[verdicts_by_id[item.id] for item in items])


def _render_evidence(pack: EvidencePack) -> str:
    """Render an evidence pack as labelled spans for the judge prompt."""
    if not pack.spans:
        return ""
    lines = [f"[{span.chunk_id}] {span.text}" for span in pack.spans]
    return "\n\n".join(lines)


def _resolve_citations(chunk_ids: list[str], pack: EvidencePack) -> list[Span]:
    """Filter the pack's spans to those whose chunk_id appears in the cited set.

    Unknown chunk_ids (model hallucinations) are silently dropped — the
    playbook trusts the spans in the pack but not the model's free-text
    references.
    """
    cited = set(chunk_ids)
    return [span for span in pack.spans if span.chunk_id in cited]


def _as_verdict_literal(value: _VerdictLabel) -> VerdictLiteral:
    return value


__all__ = [
    "ChecklistItem",
    "CoverageAuditPlaybook",
    "CoverageReport",
    "CoverageRetriever",
]
