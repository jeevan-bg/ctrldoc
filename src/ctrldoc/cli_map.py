"""CLI helpers for the concept-relation-map subcommand.

Four pieces:

  - `StoreEntityConceptExtractor` — deterministic: pulls entities
    from the bundle's store and emits each as a `Concept`. Caps
    to `max_concepts` (default 10) by mention count, descending.
    Avoids O(M²) LLM blowup when the doc has dozens of entities.

  - `BundleCoOccurrenceRetriever` — wraps the shared
    `BundleRetriever` (S-113). For each (c_i, c_j) pair, retrieves
    against `f"{c_i.name} {c_j.name}"` and returns the resulting
    evidence pack. Empty packs short-circuit the playbook.

  - `LLMRelationClassifier` — asks the bundle's local task client
    to emit `{"type": <RelationTypeLiteral>|"unrelated",
    "confidence": <0..1>, "citation_chunk_ids": [str]}`. Returns
    `None` when the model marks the pair as `unrelated`.

  - `render_map_markdown(graph, ...)` — emits a Markdown adjacency
    table + Mermaid graph block (typed edges between concept
    nodes), keyed off the playbook's `RelationGraph`.

SPEC-REF: §5.6 (relation_map), §6 (CLI)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.cli_audit import BundleRetriever
from ctrldoc.models import EvidencePack, RelationTypeLiteral, Span
from ctrldoc.ops.map import (
    Concept,
    RelationClassification,
    RelationGraph,
)
from ctrldoc.orch.task import StatelessTaskRunner, TaskInput
from ctrldoc.store import Store

DEFAULT_MAX_CONCEPTS = 10
_RELATION_TYPES: tuple[str, ...] = (
    *get_args(RelationTypeLiteral),
    "unrelated",
)


# --- concept extractor ---


class StoreEntityConceptExtractor:
    """Deterministic concept extractor — uses the store's entities.

    Sorts entities by mention count (descending) then by id for
    stability, then caps to `max_concepts`. Returns `[]` when the
    store has no entities (graph degenerates to nodes-only).
    """

    def __init__(self, *, store: Store, max_concepts: int = DEFAULT_MAX_CONCEPTS) -> None:
        if max_concepts <= 0:
            raise ValueError("max_concepts must be positive")
        self._store = store
        self._max_concepts = max_concepts

    def extract(self) -> list[Concept]:
        entities = list(self._store.iter_entities())
        # Most-mentioned entities first; stable on ties by id.
        entities.sort(key=lambda e: (-len(e.mention_chunk_ids), e.id))
        capped = entities[: self._max_concepts]
        return [Concept(id=e.id, name=e.aliases[0] if e.aliases else e.id) for e in capped]


# --- co-occurrence retriever ---


class BundleCoOccurrenceRetriever:
    """`CoOccurrenceRetriever` wrapping a `BundleRetriever`.

    Retrieves against the concept names joined by a space (BM25 +
    dense over the doc index). Pairs that retrieve no spans are
    treated as "doc does not co-mention" and the playbook skips
    the classifier call.
    """

    def __init__(self, *, bundle_retriever: BundleRetriever) -> None:
        self._bundle_retriever = bundle_retriever

    def retrieve(self, c_i: Concept, c_j: Concept) -> EvidencePack:
        # Strip non-alphanumerics from concept names to keep the BM25
        # query well-formed (Tantivy parses `:` as a field separator
        # and various punctuation as operators).
        query = f"{_sanitise_query(c_i.name)} {_sanitise_query(c_j.name)}".strip()
        if not query:
            return EvidencePack(query=query, spans=[], token_count=0, retrieval_plan=[])
        return self._bundle_retriever.retrieve(query)


def _sanitise_query(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z _-]+", " ", text).strip()


# --- classifier ---


class _ClassifierOutput(BaseModel):
    """Schema the LLM emits for one (c_i, c_j) pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    citation_chunk_ids: list[str] = Field(default_factory=list)


_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify the relation between two concepts using only the "
    "EVIDENCE spans. Return one JSON object of shape:\n"
    '  {"type": <one of: depends_on, contradicts, refines, instantiates, '
    "conflicts_with, prerequisite_of, alternative_to, unrelated>,\n"
    '   "confidence": <0.0-1.0>,\n'
    '   "citation_chunk_ids": [<chunk_id copied from EVIDENCE>]}\n\n'
    "Pick `unrelated` if the evidence does not establish a relation "
    "between the two concepts. Cite only chunk_ids that appear in "
    "EVIDENCE. No prose outside the JSON."
)


class LLMRelationClassifier:
    """LLM-backed relation classifier; returns `None` for `unrelated`.

    Routes through the bundle's `local` tier (Ollama Qwen in thrifty).
    Confidence is clamped to `[0, 1]` and citation chunk_ids are
    resolved against the evidence pack's spans; hallucinated ids
    are silently dropped.
    """

    def __init__(
        self,
        *,
        prefix: CacheablePrefix,
        task_runner: StatelessTaskRunner,
    ) -> None:
        self._prefix = prefix
        self._task_runner = task_runner

    def classify(
        self,
        c_i: Concept,
        c_j: Concept,
        evidence: EvidencePack,
    ) -> RelationClassification | None:
        if not evidence.spans:
            return None
        evidence_text = "\n\n".join(f"[{s.chunk_id}] {s.text}" for s in evidence.spans)
        task_input = (
            f"CONCEPT A:\n  id: {c_i.id}\n  name: {c_i.name}\n\n"
            f"CONCEPT B:\n  id: {c_j.id}\n  name: {c_j.name}\n"
        )
        task = TaskInput(
            prefix=self._prefix,
            evidence_pack=evidence_text,
            task_input=task_input,
        )
        result = self._task_runner.run(task, output_model=_ClassifierOutput)
        if result.type not in _RELATION_TYPES:
            return None
        if result.type == "unrelated":
            return None
        span_by_id = {s.chunk_id: s for s in evidence.spans}
        citations: list[Span] = [
            span_by_id[cid] for cid in result.citation_chunk_ids if cid in span_by_id
        ]
        return RelationClassification(
            type=result.type,  # type: ignore[arg-type]  # narrowed by membership check
            citations=citations,
            confidence=max(0.0, min(1.0, result.confidence)),
        )


# --- markdown renderer ---


def render_map_markdown(
    *,
    graph: RelationGraph,
    target_path: Path,
    profile: str,
    run_id: str,
) -> str:
    """Render a `RelationGraph` as Markdown.

    Layout:

    1. Header — target, profile, run_id, node + edge counts.
    2. Adjacency table — one row per edge with src, type, dst,
       confidence, cited chunk_ids.
    3. Mermaid graph block — typed edges between concept nodes.
       Concepts referenced only as endpoints of zero edges still
       appear as standalone nodes so the graph is complete.
    """
    lines: list[str] = []
    lines.append("# ctrldoc — concept relation map")
    lines.append("")
    lines.append(f"- **Target**: `{target_path}`")
    lines.append(f"- **Profile**: `{profile}`")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append(f"- **Nodes**: {len(graph.nodes)}")
    lines.append(f"- **Edges**: {len(graph.edges)}")
    lines.append("")

    lines.append("## Adjacency table")
    lines.append("")
    if not graph.edges:
        lines.append("_(no relations detected)_")
        lines.append("")
    else:
        lines.append("| src | type | dst | confidence | citations |")
        lines.append("|---|---|---|---:|---|")
        for edge in graph.edges:
            cites = ", ".join(f"`[{s.chunk_id}]`" for s in edge.citations) or "_(none)_"
            lines.append(
                f"| `{edge.src_concept}` | `{edge.type}` | `{edge.dst_concept}` | "
                f"{edge.confidence:.2f} | {cites} |"
            )
        lines.append("")

    lines.append("## Graph")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    if not graph.nodes:
        lines.append('    _empty["(no concepts)"]')
    else:
        for concept in graph.nodes:
            safe_id = _mermaid_id(concept.id)
            safe_name = concept.name.replace('"', "'")
            lines.append(f'    {safe_id}["{safe_name}"]')
        for edge in graph.edges:
            src = _mermaid_id(edge.src_concept)
            dst = _mermaid_id(edge.dst_concept)
            lines.append(f"    {src} -- {edge.type} --> {dst}")
    lines.append("```")
    return "\n".join(lines).rstrip() + "\n"


def _mermaid_id(concept_id: str) -> str:
    """Mermaid node ids must match `[A-Za-z][\\w]*`; slug everything else."""
    safe = re.sub(r"[^0-9A-Za-z_]", "_", concept_id)
    if not safe or not safe[0].isalpha():
        safe = "c_" + safe
    return safe


__all__ = [
    "DEFAULT_MAX_CONCEPTS",
    "BundleCoOccurrenceRetriever",
    "LLMRelationClassifier",
    "StoreEntityConceptExtractor",
    "render_map_markdown",
]
