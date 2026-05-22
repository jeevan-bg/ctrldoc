"""Family-13 invariants: incremental update / freshness.

Editing one section must only re-index that section's chunks.
Adding a section must add only its chunks. Removing a section must
clear its chunks. The pipeline's incremental driver returns an
`IncrementalStats` summary that the test asserts against.

SPEC-REF: §4.1, §8.6 family 13
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.ingest.coref import IdentityCorefResolver
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.ingest.ner import StubNERTagger
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document, ingest_document_incremental
from ctrldoc.ingest.summarizer import HeuristicSummarizer
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex


@pytest.fixture
def pipeline_kit(tmp_path: Path):  # type: ignore[no-untyped-def]
    counter = {"n": 0}

    def make() -> dict:
        counter["n"] += 1
        store = InMemoryStore()
        return {
            "parser": MarkdownParser(),
            "coref": IdentityCorefResolver(),
            "ner": StubNERTagger({}),
            "ner_labels": ["person"],
            "embedder": HashEmbedder(dimension=16),
            "summarizer": HeuristicSummarizer(),
            "store": store,
            "vector_index": InMemoryVectorIndex(dimension=16),
            "bm25_index": TantivyBM25Index(path=tmp_path / f"bm25-{counter['n']}"),
            "store_ref": store,
        }

    return make


def _run(source: Path | str, kit: dict, *, incremental: bool = False) -> object:
    fn = ingest_document_incremental if incremental else ingest_document
    return fn(
        source=source,
        parser=kit["parser"],
        coref=kit["coref"],
        ner=kit["ner"],
        ner_labels=kit["ner_labels"],
        embedder=kit["embedder"],
        summarizer=kit["summarizer"],
        store=kit["store"],
        vector_index=kit["vector_index"],
        bm25_index=kit["bm25_index"],
    )


_BASE_DOC = """# Project

Intro paragraph.

## 1. Alpha

Alpha body sentence. Alpha second sentence.

## 2. Beta

Beta body sentence. Beta second sentence.

## 3. Gamma

Gamma body sentence. Gamma second sentence.
"""


@pytest.mark.family_incremental_update
def test_unchanged_doc_reports_no_changes(tmp_path: Path, pipeline_kit) -> None:  # type: ignore[no-untyped-def]
    kit = pipeline_kit()
    doc = tmp_path / "doc.md"
    doc.write_text(_BASE_DOC, encoding="utf-8")
    _run(doc, kit)
    chunk_ids_before = sorted(c.id for c in kit["store_ref"].iter_chunks())

    stats = _run(doc, kit, incremental=True)
    chunk_ids_after = sorted(c.id for c in kit["store_ref"].iter_chunks())
    assert stats.sections_added == 0  # type: ignore[attr-defined]
    assert stats.sections_changed == 0  # type: ignore[attr-defined]
    assert stats.sections_removed == 0  # type: ignore[attr-defined]
    assert chunk_ids_before == chunk_ids_after


@pytest.mark.family_incremental_update
def test_editing_one_section_only_reindexes_that_section(
    tmp_path: Path,
    pipeline_kit,  # type: ignore[no-untyped-def]
) -> None:
    kit = pipeline_kit()
    doc = tmp_path / "doc.md"
    doc.write_text(_BASE_DOC, encoding="utf-8")
    _run(doc, kit)

    # Snapshot chunk ids per section.
    chunks_by_section_before: dict[str, set[str]] = {}
    for c in kit["store_ref"].iter_chunks():
        chunks_by_section_before.setdefault(c.section_id, set()).add(c.id)

    edited = _BASE_DOC.replace(
        "Beta body sentence. Beta second sentence.",
        "Beta body has been edited entirely. New second sentence.",
    )
    doc.write_text(edited, encoding="utf-8")
    stats = _run(doc, kit, incremental=True)

    chunks_by_section_after: dict[str, set[str]] = {}
    for c in kit["store_ref"].iter_chunks():
        chunks_by_section_after.setdefault(c.section_id, set()).add(c.id)

    beta_section_id = next(
        s.id for s in kit["store_ref"].iter_sections() if s.title.endswith("Beta")
    )

    # Beta's chunks must change; every other section's chunks stay identical.
    assert chunks_by_section_before[beta_section_id] != chunks_by_section_after[beta_section_id]
    for section_id, before_ids in chunks_by_section_before.items():
        if section_id == beta_section_id:
            continue
        assert before_ids == chunks_by_section_after.get(section_id, set()), (
            f"unrelated section re-indexed: {section_id}"
        )

    assert stats.sections_changed == 1  # type: ignore[attr-defined]
    assert stats.sections_added == 0  # type: ignore[attr-defined]
    assert stats.sections_removed == 0  # type: ignore[attr-defined]


@pytest.mark.family_incremental_update
def test_adding_a_section_only_adds_its_chunks(
    tmp_path: Path,
    pipeline_kit,  # type: ignore[no-untyped-def]
) -> None:
    kit = pipeline_kit()
    doc = tmp_path / "doc.md"
    doc.write_text(_BASE_DOC, encoding="utf-8")
    _run(doc, kit)
    chunk_ids_before = {c.id for c in kit["store_ref"].iter_chunks()}

    extended = _BASE_DOC + "\n## 4. Delta\n\nDelta body sentence. Delta second sentence.\n"
    doc.write_text(extended, encoding="utf-8")
    stats = _run(doc, kit, incremental=True)

    chunk_ids_after = {c.id for c in kit["store_ref"].iter_chunks()}
    new_ids = chunk_ids_after - chunk_ids_before
    removed_ids = chunk_ids_before - chunk_ids_after
    assert new_ids, "no new chunks were added"
    assert removed_ids == set(), "untouched sections lost chunks"

    delta_section_id = next(
        s.id for s in kit["store_ref"].iter_sections() if s.title.endswith("Delta")
    )
    assert all(
        c.section_id == delta_section_id for c in kit["store_ref"].iter_chunks() if c.id in new_ids
    )
    assert stats.sections_added == 1  # type: ignore[attr-defined]
    assert stats.sections_changed == 0  # type: ignore[attr-defined]
    assert stats.sections_removed == 0  # type: ignore[attr-defined]


@pytest.mark.family_incremental_update
def test_removing_a_section_clears_its_chunks(
    tmp_path: Path,
    pipeline_kit,  # type: ignore[no-untyped-def]
) -> None:
    kit = pipeline_kit()
    doc = tmp_path / "doc.md"
    doc.write_text(_BASE_DOC, encoding="utf-8")
    _run(doc, kit)

    gamma_section_id = next(
        s.id for s in kit["store_ref"].iter_sections() if s.title.endswith("Gamma")
    )
    gamma_chunks_before = {
        c.id for c in kit["store_ref"].iter_chunks() if c.section_id == gamma_section_id
    }
    assert gamma_chunks_before, "test pre-condition: Gamma must have chunks initially"

    shortened = _BASE_DOC.split("## 3. Gamma")[0]
    doc.write_text(shortened, encoding="utf-8")
    stats = _run(doc, kit, incremental=True)

    chunk_ids_after = {c.id for c in kit["store_ref"].iter_chunks()}
    assert not gamma_chunks_before & chunk_ids_after, "Gamma chunks were not deleted"
    assert not any(s.id == gamma_section_id for s in kit["store_ref"].iter_sections()), (
        "Gamma section was not deleted"
    )
    assert stats.sections_removed == 1  # type: ignore[attr-defined]
    assert stats.sections_changed == 0  # type: ignore[attr-defined]
    assert stats.sections_added == 0  # type: ignore[attr-defined]
