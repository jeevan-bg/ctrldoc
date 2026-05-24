"""`SQLiteStore.append_claim` / `iter_claims` / `get_claim` round-trip.

The §6.2 universal-tuple claims persist into the `claims` table.
Idempotent by id — same `claim.id` overwrites in-place — so a
re-ingest is safe and dedupes by content-hash identity at the SQL
boundary.

SPEC-REF: §6.2, §6.4
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.models import Span
from ctrldoc.models_v1 import Claim
from ctrldoc.store.sqlite import SQLiteStore

pytestmark = [pytest.mark.family_referential_integrity]


def _claim(
    *,
    id_: str = "sha256:" + "a" * 64,
    doc_id: str = "doc-1",
    text: str = "system validate inputs",
    subject: str = "system",
    predicate: str = "validate",
    obj: str | None = "inputs",
    polarity: str = "+",
    modality: str | None = "must",
    qualifier: dict[str, object] | None = None,
    section_id: str = "sec-001",
    chunk_id: str = "chunk-001",
    confidence: float = 1.0,
) -> Claim:
    return Claim(
        id=id_,
        doc_id=doc_id,
        text=text,
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier or {},
        span_refs=[Span(chunk_id=chunk_id, char_start=0, char_end=len(text), text=text)],
        section_id=section_id,
        concept_ids=[],
        typed_slots={},
        confidence=confidence,  # type: ignore[arg-type]
    )


def test_append_claim_then_get_returns_equal_row(tmp_path: Path) -> None:
    db = SQLiteStore(tmp_path / "store.db")
    claim = _claim()
    db.append_claim(claim)
    fetched = db.get_claim(claim.id)
    assert fetched == claim


def test_iter_claims_yields_all_appended_in_id_order(tmp_path: Path) -> None:
    db = SQLiteStore(tmp_path / "store.db")
    claims = [
        _claim(id_="sha256:" + "0" * 64, text="a"),
        _claim(id_="sha256:" + "2" * 64, text="c"),
        _claim(id_="sha256:" + "1" * 64, text="b"),
    ]
    for c in claims:
        db.append_claim(c)
    fetched = list(db.iter_claims())
    assert [c.id for c in fetched] == sorted(c.id for c in claims)


def test_append_claim_is_idempotent_on_duplicate_id(tmp_path: Path) -> None:
    db = SQLiteStore(tmp_path / "store.db")
    claim = _claim()
    db.append_claim(claim)
    # Same id, different surface text — the second call must overwrite,
    # not raise. Identity is the content-hashed id.
    overwritten = claim.model_copy(update={"text": "system validate inputs (rev)"})
    db.append_claim(overwritten)
    fetched = db.get_claim(claim.id)
    assert fetched is not None
    assert fetched.text == "system validate inputs (rev)"
    # And the total row count is still one.
    assert len(list(db.iter_claims())) == 1


def test_get_claim_returns_none_for_missing_id(tmp_path: Path) -> None:
    db = SQLiteStore(tmp_path / "store.db")
    assert db.get_claim("sha256:" + "z" * 64) is None


def test_iter_claims_for_doc_filters_by_doc(tmp_path: Path) -> None:
    db = SQLiteStore(tmp_path / "store.db")
    db.append_claim(_claim(id_="sha256:" + "0" * 64, doc_id="doc-A"))
    db.append_claim(_claim(id_="sha256:" + "1" * 64, doc_id="doc-B"))
    db.append_claim(_claim(id_="sha256:" + "2" * 64, doc_id="doc-A"))
    a_only = list(db.iter_claims_for_doc("doc-A"))
    assert {c.doc_id for c in a_only} == {"doc-A"}
    assert len(a_only) == 2


def test_qualifier_and_typed_slots_round_trip_through_json(tmp_path: Path) -> None:
    db = SQLiteStore(tmp_path / "store.db")
    rich = Claim(
        id="sha256:" + "f" * 64,
        doc_id="doc-1",
        text="system must validate inputs under load",
        subject="system",
        predicate="validate",
        object="inputs",
        polarity="+",
        modality="must",
        qualifier={"text": "under load"},
        span_refs=[Span(chunk_id="chunk-001", char_start=0, char_end=10, text="0123456789")],
        section_id="sec-001",
        concept_ids=["c1", "c2"],
        typed_slots={"actor": "system", "object_type": "inputs"},
        confidence=1.0,  # type: ignore[arg-type]
    )
    db.append_claim(rich)
    fetched = db.get_claim(rich.id)
    assert fetched == rich


def test_append_claim_with_none_subject_and_object(tmp_path: Path) -> None:
    # The Claim model allows subject and object to be None — the SQL
    # schema's `subject` / `object` columns are nullable. Verify the
    # round-trip preserves NULL rather than coercing to empty string.
    db = SQLiteStore(tmp_path / "store.db")
    bare = _claim(subject=None, obj=None)  # type: ignore[arg-type]
    db.append_claim(bare)
    fetched = db.get_claim(bare.id)
    assert fetched is not None
    assert fetched.subject is None
    assert fetched.object is None


def test_clear_all_removes_claim_rows(tmp_path: Path) -> None:
    db = SQLiteStore(tmp_path / "store.db")
    db.append_claim(_claim())
    db.clear_all()
    assert list(db.iter_claims()) == []
