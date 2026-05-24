"""Ingest pipeline wires within-doc edge inference into the L0 path.

The pipeline gains an optional `edge_inferer: WithinDocEdgeInferer | None`.
When provided, after claims land in the store the pipeline iterates the
just-persisted claims for the doc, runs the inferer, and writes every
emitted `TypedEdge` through `store.append_typed_edge`. Without the
inferer the pipeline behaves exactly as before — zero typed-edges land.

Determinism: identical inputs ⇒ identical persisted typed-edge rows
across re-runs (verified by id sets and full-row equality).

Gate (slice S-155): every emitted edge carries at least one citation
span in its `citations` list — every persisted `Claim` has `span_refs`
by §7, and the inferer threads at least one of them per endpoint into
each edge.

SPEC-REF: §6.3, §6.5
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.within_doc_edges import WithinDocEdgeInferer
from ctrldoc.ingest.coref import IdentityCorefResolver
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.ingest.ner import StubNERTagger
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.ingest.summarizer import HeuristicSummarizer
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex

pytestmark = [pytest.mark.family_referential_integrity]

_EMBED_DIM = 16
_LABELS = ["person"]

# A markdown source whose chunks contain both a `must` and a `should`
# variant of the same SVO — the Galois floor will emit an `entails`
# edge from the obligatory claim to the recommended one.
_SOURCE = """\
# Spec

The system must validate inputs. The system should validate inputs.
"""


class _CueExtractor:
    """Deterministic stub `ClaimExtractor` — one tuple per cue word."""

    def extract(self, sentence: str) -> list[ClaimTuple]:
        sentence_lower = sentence.lower()
        out: list[ClaimTuple] = []
        if "must" in sentence_lower:
            out.append(
                ClaimTuple(
                    subject="system",
                    predicate="validate",
                    object="inputs",
                    polarity="affirmative",
                    modality="obligatory",
                )
            )
        if "should" in sentence_lower:
            out.append(
                ClaimTuple(
                    subject="system",
                    predicate="validate",
                    object="inputs",
                    polarity="affirmative",
                    modality="recommended",
                )
            )
        return out


def _write_source(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "spec.md"
    src.write_text(_SOURCE, encoding="utf-8")
    return src


def _run_ingest(*, tmp_path: Path, with_inferer: bool) -> InMemoryStore:
    src = _write_source(tmp_path)
    store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=_EMBED_DIM)
    bm25_index = TantivyBM25Index(path=tmp_path / "bm25")
    ingest_document(
        source=src,
        parser=MarkdownParser(),
        coref=IdentityCorefResolver(),
        ner=StubNERTagger(by_text={}),
        ner_labels=_LABELS,
        embedder=HashEmbedder(dimension=_EMBED_DIM),
        summarizer=HeuristicSummarizer(),
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
        claim_extractor=_CueExtractor(),
        edge_inferer=WithinDocEdgeInferer(nli_scorer=None) if with_inferer else None,
        doc_id="doc-spec",
    )
    return store


def test_pipeline_without_inferer_persists_zero_typed_edges(tmp_path: Path) -> None:
    store = _run_ingest(tmp_path=tmp_path, with_inferer=False)
    assert list(store.iter_typed_edges()) == []


def test_pipeline_with_inferer_persists_galois_entails_edge(tmp_path: Path) -> None:
    store = _run_ingest(tmp_path=tmp_path, with_inferer=True)
    edges = list(store.iter_typed_edges())
    assert edges, "expected at least one Galois-inferred edge"
    # The obligatory → recommended pair should produce an `entails`.
    types = {e.type for e in edges}
    assert "entails" in types
    assert all(e.source == "heuristic" for e in edges)


def test_every_persisted_edge_cites_at_least_one_span(tmp_path: Path) -> None:
    """S-155 gate — every emitted edge cites a span."""
    store = _run_ingest(tmp_path=tmp_path, with_inferer=True)
    edges = list(store.iter_typed_edges())
    assert edges, "fixture should produce at least one edge"
    for edge in edges:
        assert len(edge.citations) >= 1, f"edge {edge} has no citations"
        for span in edge.citations:
            assert span.chunk_id, f"edge {edge} has a citation with empty chunk_id"


def test_pipeline_edge_inference_is_deterministic_across_runs(tmp_path: Path) -> None:
    a = _run_ingest(tmp_path=tmp_path / "a", with_inferer=True)
    b = _run_ingest(tmp_path=tmp_path / "b", with_inferer=True)
    a_edges = sorted(a.iter_typed_edges(), key=lambda e: (e.type, e.src_id, e.dst_id))
    b_edges = sorted(b.iter_typed_edges(), key=lambda e: (e.type, e.src_id, e.dst_id))
    assert a_edges == b_edges
