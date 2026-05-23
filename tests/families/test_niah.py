"""Family-2 invariants: Needle-in-a-Haystack retrieval.

A document with distinctive sentinel strings injected at the start,
middle, end, and a few random positions must achieve 100% top-5
retrieval for every sentinel. The MVP asserts this via the lexical
view (BM25) and the full pipeline (planner + executor + RRF).

The dense view via `HashEmbedder` is intentionally not tested here:
that embedder is a deterministic placeholder, not a semantic encoder,
so a partial-string query and a full-chunk-text embedding are
uncorrelated. NIAH against the dense view will be validated once the
production embedder (S-036b) is wired.

SPEC-REF: §4.3, §8.6 family 2
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from ctrldoc.assembler import assemble_cacheable_prefix
from ctrldoc.ingest.coref import IdentityCorefResolver
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.ingest.ner import StubNERTagger
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.ingest.summarizer import HeuristicSummarizer
from ctrldoc.retrieval.executor import PlanExecutor
from ctrldoc.retrieval.fusion import fuse_step_results
from ctrldoc.retrieval.planner import HeuristicPlanner
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex

_SENTINELS = {
    "start": "NEEDLE_ALPHA_42",
    "mid": "NEEDLE_BETA_99",
    "end": "NEEDLE_GAMMA_777",
    "random1": "NEEDLE_DELTA_314",
    "random2": "NEEDLE_EPSILON_271",
}


def _build_haystack_doc(sentinels: dict[str, str]) -> str:
    """Build a synthetic doc with ~30 leaf sections of filler text and the
    sentinels at the documented positions."""
    section_count = 30
    rng = random.Random(0xC1)  # deterministic for the random-position picks
    random_positions = {
        sentinels["random1"]: rng.randrange(5, section_count - 5),
        sentinels["random2"]: rng.randrange(5, section_count - 5),
    }

    lines: list[str] = ["# Aurora\n", "\n", "Filler intro paragraph.\n", "\n"]
    for i in range(section_count):
        lines.append(f"## Section {i:02d}\n")
        lines.append("\n")
        body = (
            f"Filler sentence number {i}. Cache replication factor is N=3. "
            f"Operational complexity is bounded by component count. "
            f"Section {i:02d} discusses topic {i:02d}."
        )
        if i == 0:
            body = f"{sentinels['start']} appears first. {body}"
        if i == section_count // 2:
            body = f"{body} {sentinels['mid']} appears in the middle."
        if i == section_count - 1:
            body = f"{body} {sentinels['end']} appears last."
        for sentinel, pos in random_positions.items():
            if i == pos:
                body = f"{body} {sentinel} appears at a random position."
        lines.append(body + "\n")
        lines.append("\n")
    return "".join(lines)


@pytest.fixture
def kit(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=16)
    bm25_index = TantivyBM25Index(path=tmp_path / "bm25")
    embedder = HashEmbedder(dimension=16)
    return {
        "store": store,
        "vector_index": vector_index,
        "bm25_index": bm25_index,
        "embedder": embedder,
        "executor": PlanExecutor(
            store=store,
            vector_index=vector_index,
            bm25_index=bm25_index,
            embedder=embedder,
        ),
    }


@pytest.fixture
def haystack(tmp_path: Path, kit: dict) -> dict[str, str]:
    doc = tmp_path / "doc.md"
    doc.write_text(_build_haystack_doc(_SENTINELS), encoding="utf-8")
    ingest_document(
        source=doc,
        parser=MarkdownParser(),
        coref=IdentityCorefResolver(),
        ner=StubNERTagger({}),
        ner_labels=[],
        embedder=kit["embedder"],
        summarizer=HeuristicSummarizer(),
        store=kit["store"],
        vector_index=kit["vector_index"],
        bm25_index=kit["bm25_index"],
    )
    return _SENTINELS


@pytest.mark.family_niah
def test_bm25_finds_every_sentinel_in_top_5(haystack: dict[str, str], kit: dict) -> None:
    """Lexical view alone must achieve 100% top-5 recall on injected sentinels."""
    for position, sentinel in haystack.items():
        hits = kit["bm25_index"].search(sentinel, k=5)
        chunk_ids = [chunk_id for chunk_id, _ in hits]
        assert chunk_ids, f"no BM25 hits for {position}={sentinel!r}"
        # The chunk containing the sentinel must be in the top-5.
        sentinel_chunks = [c.id for c in kit["store"].iter_chunks() if sentinel in c.text]
        assert sentinel_chunks, f"sentinel {sentinel!r} not stored in any chunk"
        assert any(chunk_id in chunk_ids for chunk_id in sentinel_chunks), (
            f"sentinel {sentinel!r} ({position}) missing from top-5 BM25 hits"
        )


@pytest.mark.family_niah
def test_full_pipeline_finds_every_sentinel_in_top_5(
    haystack: dict[str, str],
    kit: dict,
) -> None:
    """planner → executor → RRF must achieve 100% top-5 recall on sentinels."""
    prefix = assemble_cacheable_prefix(kit["store"], system_prompt="")
    planner = HeuristicPlanner(default_k=10)
    for position, sentinel in haystack.items():
        plan = planner.plan(prefix, sentinel)
        results = kit["executor"].execute(plan)
        fused = fuse_step_results(results)
        top_5 = [chunk_id for chunk_id, _ in fused[:5]]
        sentinel_chunks = [c.id for c in kit["store"].iter_chunks() if sentinel in c.text]
        assert any(chunk_id in top_5 for chunk_id in sentinel_chunks), (
            f"sentinel {sentinel!r} ({position}) missing from top-5 fused hits; top-5 was {top_5}"
        )
