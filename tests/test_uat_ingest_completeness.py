"""End-to-end ingest gate over the Bishop-style 2-page PDF fixture.

`tests/fixtures/uat/bishop_2pages.pdf` is the production-hardening
UAT artifact: a small two-page PDF that exercises the full L0
ingest path via the parser dispatch helper. The gate asserts:

1. Parser dispatch routes `.pdf` to `PDFParser`.
2. Running the heuristic-profile L0 pipeline over the PDF parses
   at least one section, indexes at least one chunk, and stores
   the same chunk under both the in-memory store and the BM25
   index.
3. Re-ingest is byte-deterministic — chunk ids and section ids
   match across two runs over the same input file.

Holds the §5.1 ingest contract end-to-end without a network or
LLM round-trip.

SPEC-REF: §5.1
"""

from __future__ import annotations

from pathlib import Path

from ctrldoc.ingest.coref import IdentityCorefResolver
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.ingest.ner import StubNERTagger
from ctrldoc.ingest.parser_dispatch import get_parser
from ctrldoc.ingest.pdf import PDFParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.ingest.summarizer import HeuristicSummarizer
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex

_FIXTURE = Path(__file__).parent / "fixtures" / "uat" / "bishop_2pages.pdf"
_HEURISTIC_EMBED_DIM = 32

# Labels mirror the CLI ingest path's heuristic-profile default.
_LABELS = ["person", "organization", "location"]


# family_referential_integrity — committed-fixture contract.


def test_bishop_pdf_fixture_present() -> None:
    assert _FIXTURE.exists(), (
        "bishop_2pages.pdf missing — rebuild with "
        "`.venv/bin/python tests/fixtures/uat/build_bishop_2pages.py`"
    )
    assert _FIXTURE.stat().st_size > 500
    assert _FIXTURE.read_bytes().startswith(b"%PDF-")


def test_dispatch_picks_pdf_parser_for_bishop_fixture() -> None:
    assert isinstance(get_parser(_FIXTURE), PDFParser)


# family_synthetic_gold — end-to-end ingest gate.


def _ingest_into(tmp_path: Path) -> tuple[InMemoryStore, TantivyBM25Index]:
    parser = get_parser(_FIXTURE)
    store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=_HEURISTIC_EMBED_DIM)
    bm25_index = TantivyBM25Index(path=tmp_path / "bm25")
    stats = ingest_document(
        source=_FIXTURE,
        parser=parser,
        coref=IdentityCorefResolver(),
        ner=StubNERTagger(by_text={}),
        ner_labels=_LABELS,
        embedder=HashEmbedder(dimension=_HEURISTIC_EMBED_DIM),
        summarizer=HeuristicSummarizer(),
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )
    # Completeness floor: PDF parsed cleanly and the pipeline produced
    # at least one section and one chunk.
    assert stats.sections_parsed >= 1
    assert stats.chunks_indexed >= 1
    assert stats.embedded_tokens >= 1
    return store, bm25_index


def test_bishop_pdf_ingest_completeness(tmp_path: Path) -> None:
    store, bm25_index = _ingest_into(tmp_path)
    chunks = list(store.iter_chunks())
    sections = list(store.iter_sections())
    assert chunks, "store ended up with zero chunks"
    assert sections, "store ended up with zero sections"
    # Every chunk has at least one BM25 hit on a token drawn from
    # its own body — guards the store ↔ search-index dual-write the
    # pipeline performs.
    for chunk in chunks:
        first_token = next(
            (token for token in chunk.text.split() if token.isalnum()),
            None,
        )
        if first_token is None:
            continue
        results = bm25_index.search(first_token, k=10)
        assert any(
            hit_chunk_id == chunk.id for hit_chunk_id, _ in results
        ), f"chunk {chunk.id} not searchable in bm25 index via token {first_token!r}"


def test_bishop_pdf_ingest_is_byte_deterministic(tmp_path: Path) -> None:
    first_store, _ = _ingest_into(tmp_path / "first")
    second_store, _ = _ingest_into(tmp_path / "second")
    first_chunk_ids = sorted(c.id for c in first_store.iter_chunks())
    second_chunk_ids = sorted(c.id for c in second_store.iter_chunks())
    first_section_ids = sorted(s.id for s in first_store.iter_sections())
    second_section_ids = sorted(s.id for s in second_store.iter_sections())
    assert first_chunk_ids == second_chunk_ids
    assert first_section_ids == second_section_ids
