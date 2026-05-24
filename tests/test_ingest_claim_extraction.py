"""L0 ingest pipeline persists §6.2 universal claim tuples per chunk.

The pipeline accepts an optional `claim_extractor` Protocol. When
provided, it runs the extractor over every chunk's text, adapts each
emitted `ClaimTuple` into a persisted `Claim`, and writes it through
`store.append_claim`. Without an extractor the pipeline behaves
exactly as before (back-compat).

Determinism: identical inputs ⇒ identical persisted claim ids across
re-runs, in every profile. Content-hashed ids dedupe within a run
(same tuple in two chunks ⇒ two rows under distinct ids because the
chunk binding is part of the hash).

SPEC-REF: §6.2, §6.4
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.eval.claim_extraction import ClaimTuple
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
_LABELS = ["person", "organization"]


_SOURCE = """\
# Spec

The system must validate inputs. Operators should review logs.
"""


class _CueExtractor:
    """Deterministic stub `ClaimExtractor`.

    Emits one tuple per sentence that contains a target cue word so
    the pipeline test does not depend on spaCy. Real downstream code
    uses `SpacyTier2SVOExtractor`; the Protocol seam keeps tests
    fast and hermetic.
    """

    def extract(self, sentence: str) -> list[ClaimTuple]:
        sentence_lower = sentence.lower()
        tuples: list[ClaimTuple] = []
        if "must" in sentence_lower:
            tuples.append(
                ClaimTuple(
                    subject="system",
                    predicate="validate",
                    object="inputs",
                    polarity="affirmative",
                    modality="obligatory",
                )
            )
        if "should" in sentence_lower:
            tuples.append(
                ClaimTuple(
                    subject="operators",
                    predicate="review",
                    object="logs",
                    polarity="affirmative",
                    modality="recommended",
                )
            )
        return tuples


def _write_source(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "spec.md"
    src.write_text(_SOURCE, encoding="utf-8")
    return src


def _run_ingest(*, tmp_path: Path, with_extractor: bool) -> InMemoryStore:
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
        claim_extractor=_CueExtractor() if with_extractor else None,
        doc_id="doc-spec",
    )
    return store


def test_pipeline_with_no_extractor_persists_zero_claims(tmp_path: Path) -> None:
    store = _run_ingest(tmp_path=tmp_path, with_extractor=False)
    assert list(store.iter_claims()) == []


def test_pipeline_with_extractor_persists_claims(tmp_path: Path) -> None:
    store = _run_ingest(tmp_path=tmp_path, with_extractor=True)
    claims = list(store.iter_claims())
    assert claims, "extractor should have persisted at least one claim"
    predicates = sorted(c.predicate for c in claims)
    assert "validate" in predicates
    assert "review" in predicates


def test_persisted_claims_inherit_doc_id_and_section(tmp_path: Path) -> None:
    store = _run_ingest(tmp_path=tmp_path, with_extractor=True)
    chunk_section_ids = {c.section_id for c in store.iter_chunks()}
    for claim in store.iter_claims():
        assert claim.doc_id == "doc-spec"
        assert claim.section_id in chunk_section_ids


def test_persisted_claims_carry_one_span_per_chunk(tmp_path: Path) -> None:
    store = _run_ingest(tmp_path=tmp_path, with_extractor=True)
    chunk_ids = {c.id for c in store.iter_chunks()}
    for claim in store.iter_claims():
        assert len(claim.span_refs) >= 1
        for span in claim.span_refs:
            assert span.chunk_id in chunk_ids


def test_pipeline_is_byte_deterministic_across_two_runs(tmp_path: Path) -> None:
    a = _run_ingest(tmp_path=tmp_path / "a", with_extractor=True)
    b = _run_ingest(tmp_path=tmp_path / "b", with_extractor=True)
    a_ids = sorted(c.id for c in a.iter_claims())
    b_ids = sorted(c.id for c in b.iter_claims())
    assert a_ids == b_ids
    # And the full row content too — guards against any non-id field
    # picking up nondeterminism (timestamps, list order).
    a_rows = sorted(a.iter_claims(), key=lambda c: c.id)
    b_rows = sorted(b.iter_claims(), key=lambda c: c.id)
    assert a_rows == b_rows
