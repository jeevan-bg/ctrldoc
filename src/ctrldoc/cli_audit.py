"""CLI helpers for the coverage audit subcommand.

Three pieces:

  - `parse_checklist_markdown(text)` — deterministic Markdown-section
    parser that turns a checklist doc into a list of `ChecklistItem`s.
    Each `## H2` or `### H3` heading + its first paragraph becomes one
    item. `topic_key` is the parent section: the most recent `## H2`
    when the item is an `### H3`; the immediately preceding `# H1`
    when the item is a `## H2`; the file stem when no parent exists.

  - `BundleRetriever` — adapts a `BackendBundle`'s planner /
    embedder / store / indexes into the `QARetriever` and
    `CoverageRetriever` protocols. Runs the bundle's planner, then
    the executor, fuses step results via RRF, optionally reranks
    via the bundle's reranker, and assembles an `EvidencePack`.

  - `render_coverage_markdown(...)` — renders a `CoverageReport`
    grouped by verdict (Covered → Partial → NotCovered → Ambiguous),
    with each item's citations rendered as `[chunk_id]` references
    matching the prompt-time labelling.

SPEC-REF: §5.2 (coverage audit), §6 (CLI)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from ctrldoc.backends import BackendBundle
from ctrldoc.models import EvidencePack, Verdict
from ctrldoc.orch.batch import BatchedTaskInput, BatchItem
from ctrldoc.orch.task import StatelessTaskRunner, TaskInput, TaskOutputError
from ctrldoc.playbooks.coverage import ChecklistItem, CoverageReport
from ctrldoc.retrieval.dsl import render_plan_dsl
from ctrldoc.retrieval.evidence import build_evidence_pack
from ctrldoc.retrieval.executor import PlanExecutor
from ctrldoc.retrieval.fusion import fuse_step_results
from ctrldoc.retrieval.reranker import Candidate
from ctrldoc.store import Store
from ctrldoc.store.bm25 import BM25Index
from ctrldoc.store.vectors import VectorIndex

if TYPE_CHECKING:
    from ctrldoc.orch.batch import BatchedTaskRunner  # noqa: F401

_T = TypeVar("_T", bound=BaseModel)

# --- checklist parser ---


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "item"


def parse_checklist_markdown(
    text: str, *, fallback_topic_key: str = "checklist"
) -> list[ChecklistItem]:
    """Parse a Markdown checklist into deterministic `ChecklistItem`s.

    Each `## H2` and `### H3` becomes an item. The item text is the
    heading concatenated with the first paragraph (lines until a
    blank line or another heading). `topic_key`:

      - H3 → most recent H2 (slugged); falls back to the most
        recent H1 if no H2; finally `fallback_topic_key`.
      - H2 → most recent H1 (slugged); else `fallback_topic_key`.
      - H1 and deeper levels are not emitted as items.

    Item ids are unique slugs of `<topic_key>-<heading>`; duplicates
    get a `-<n>` suffix.
    """
    items: list[ChecklistItem] = []
    h1_slug: str | None = None
    h2_slug: str | None = None
    lines = text.splitlines()
    seen_ids: set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _HEADING_RE.match(line.rstrip())
        if not match:
            i += 1
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        if level == 1:
            h1_slug = _slugify(heading)
            h2_slug = None
            i += 1
            continue
        if level == 2:
            topic_key = h1_slug or fallback_topic_key
            h2_slug = _slugify(heading)
        elif level == 3:
            topic_key = h2_slug or h1_slug or fallback_topic_key
        else:
            i += 1
            continue
        # Collect first paragraph (until blank line or next heading).
        j = i + 1
        paragraph_lines: list[str] = []
        while j < len(lines):
            nxt = lines[j].rstrip()
            if not nxt.strip():
                break
            if _HEADING_RE.match(nxt):
                break
            paragraph_lines.append(nxt.strip())
            j += 1
        paragraph = " ".join(paragraph_lines).strip()
        item_text = f"{heading}: {paragraph}" if paragraph else heading
        base_id = _slugify(f"{topic_key}-{heading}")
        item_id = base_id
        suffix = 2
        while item_id in seen_ids:
            item_id = f"{base_id}-{suffix}"
            suffix += 1
        seen_ids.add(item_id)
        items.append(ChecklistItem(id=item_id, text=item_text, topic_key=topic_key))
        i = j
    return items


# --- bundle retriever ---


class BundleRetriever:
    """Implements `QARetriever` and `CoverageRetriever` against a bundle.

    The retriever runs the bundle's planner against the cacheable
    prefix and the query, executes the resulting plan via
    `PlanExecutor`, fuses chunk-id lists across views with RRF,
    then reranks the top hits via the bundle's reranker before
    handing the ranked list to `build_evidence_pack`.

    Heuristic profile (no LLM): the bundle's `HeuristicPlanner`
    produces a dense + lexical search plan deterministically; the
    `IdentityReranker` is a no-op. Thrifty / production swap in
    the production planner / `BGEReranker`.
    """

    def __init__(
        self,
        *,
        bundle: BackendBundle,
        store: Store,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        prefix_skeleton: str,
        prefix_glossary: str,
        top_k_after_rerank: int = 8,
    ) -> None:
        self._bundle = bundle
        self._store = store
        self._vector_index = vector_index
        self._bm25_index = bm25_index
        self._prefix_skeleton = prefix_skeleton
        self._prefix_glossary = prefix_glossary
        self._top_k_after_rerank = top_k_after_rerank
        self._executor = PlanExecutor(
            store=store,
            vector_index=vector_index,
            bm25_index=bm25_index,
            embedder=bundle.embedder,
        )

    def retrieve(self, query: str) -> EvidencePack:
        from ctrldoc.assembler import CacheablePrefix

        prefix = CacheablePrefix(
            system_prompt="You are a strict retrieval planner.",
            doc_skeleton=self._prefix_skeleton,
            entity_glossary=self._prefix_glossary,
        )
        plan = self._bundle.planner.plan(prefix, query)
        step_results = self._executor.execute(plan)
        fused = fuse_step_results(step_results)
        if not fused:
            return EvidencePack(
                query=query, spans=[], token_count=0, retrieval_plan=[render_plan_dsl(plan)]
            )
        candidates = [
            Candidate(chunk_id=cid, text=_chunk_text(self._store, cid)) for cid, _ in fused
        ]
        ranked = self._bundle.reranker.rerank(
            query=query, candidates=candidates, k=self._top_k_after_rerank
        )
        # Reranker emits `list[tuple[chunk_id, score]]` (`RerankHit`).
        ranked_chunk_ids = [chunk_id for chunk_id, _ in ranked]
        return build_evidence_pack(
            query=query,
            ranked_chunk_ids=ranked_chunk_ids,
            store=self._store,
            retrieval_plan=[render_plan_dsl(plan)],
        )


def _chunk_text(store: Store, chunk_id: str) -> str:
    chunk = store.get_chunk(chunk_id)
    return chunk.text if chunk is not None else ""


# --- per-item runner shim for small local models ---


class SequentialBatchedRunner:
    """Drop-in shape for ``BatchedTaskRunner.run(batched_input,
    output_model=...)`` that issues one `StatelessTaskRunner` call
    per item.

    Local 7B models (Qwen2.5-7B-Q4) often fail to emit the
    `BatchedTaskRunner`'s strict array shape — they echo input
    fields back into the per-item output. This shim sidesteps that
    by giving the model one item at a time with a small focused
    prompt; the per-item call still goes through the bundle's
    `local` tier (Ollama) so the budget rule holds. Slower than
    true batching, but reliable.
    """

    def __init__(
        self,
        *,
        stateless: StatelessTaskRunner,
        on_error: Callable[[BatchItem, Exception], BaseModel] | None = None,
    ) -> None:
        self._stateless = stateless
        self._on_error = on_error

    def run(self, task: BatchedTaskInput, *, output_model: type[_T]) -> list[_T]:
        results: list[_T] = []
        for item in task.items:
            single = TaskInput(
                prefix=task.prefix,
                evidence_pack=task.evidence_pack,
                task_input=item.task_input,
            )
            try:
                results.append(self._stateless.run(single, output_model=output_model))
            except TaskOutputError as exc:
                if self._on_error is None:
                    raise
                fallback = self._on_error(item, exc)
                if not isinstance(fallback, output_model):
                    raise TypeError(
                        f"on_error must return a {output_model.__name__} instance; "
                        f"got {type(fallback).__name__}"
                    ) from exc
                results.append(fallback)
        return results


# --- coverage report renderer ---


_VERDICT_ORDER: Sequence[str] = ("Covered", "Partial", "NotCovered", "Ambiguous")


def render_coverage_markdown(
    *,
    report: CoverageReport,
    items: list[ChecklistItem],
    checklist_path: Path,
    target_path: Path,
    profile: str,
    run_id: str,
) -> str:
    """Render a coverage audit report grouped by verdict.

    Summary table first (counts per verdict), then per-verdict
    sections listing each item with its citations.
    """
    items_by_id = {item.id: item for item in items}
    verdicts_by_label: dict[str, list[tuple[ChecklistItem, Verdict]]] = {
        label: [] for label in _VERDICT_ORDER
    }
    for verdict in report.verdicts:
        item = items_by_id.get(verdict.item_id)
        if item is None:
            continue
        verdicts_by_label.setdefault(verdict.verdict, []).append((item, verdict))

    lines: list[str] = []
    lines.append("# ctrldoc — coverage audit report")
    lines.append("")
    lines.append(f"- **Checklist**: `{checklist_path}`")
    lines.append(f"- **Target**: `{target_path}`")
    lines.append(f"- **Profile**: `{profile}`")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Verdict | Count |")
    lines.append("|---|---:|")
    for label in _VERDICT_ORDER:
        lines.append(f"| {label} | {len(verdicts_by_label[label])} |")
    lines.append(f"| **Total** | **{len(report.verdicts)}** |")
    lines.append("")

    for label in _VERDICT_ORDER:
        bucket = verdicts_by_label[label]
        lines.append(f"## {label} ({len(bucket)})")
        lines.append("")
        if not bucket:
            lines.append("_(none)_")
            lines.append("")
            continue
        for item, verdict in bucket:
            confidence = verdict.confidence
            lines.append(f"### `{item.id}` — confidence {confidence:.2f}")
            lines.append("")
            lines.append(item.text)
            lines.append("")
            citations = list(verdict.citations)
            if citations:
                lines.append("**Citations**:")
                lines.append("")
                for span in citations:
                    snippet = span.text.strip().replace("\n", " ")
                    if len(snippet) > 240:
                        snippet = snippet[:237] + "…"
                    lines.append(f"- `[{span.chunk_id}]` {snippet}")
                lines.append("")
            else:
                lines.append("_(no citations)_")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "BundleRetriever",
    "SequentialBatchedRunner",
    "parse_checklist_markdown",
    "render_coverage_markdown",
]
