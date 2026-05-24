"""Adapter from extractor `ClaimTuple` to persisted `models_v1.Claim`.

The §6.2 universal tuple is what the SVO extractor emits. The §7 / §8
data model stores the same logical content with bookkeeping the
substrate needs: content-hashed id, parent doc / section / chunk
identity, span_refs into the chunk text, and the §7 modal-force
alphabet (`assert` / `must` / `may` / `should` / `shall` / `neg`)
rather than the extractor's §6.2 surface alphabet (`asserted` /
`obligatory` / `recommended` / `permitted` / `prohibited` /
`hypothetical`). This adapter is the canonical seam.

SPEC-REF: §6.2, §6.4
"""

from __future__ import annotations

import pytest

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.claim_persistence import claim_from_tuple, claim_id_for_tuple
from ctrldoc.models import Chunk
from ctrldoc.models_v1 import Claim

pytestmark = [pytest.mark.family_determinism]


# --- fixtures ----------------------------------------------------------------


def _chunk(text: str = "The system must validate inputs. It should log errors.") -> Chunk:
    return Chunk(
        id="chunk-001",
        section_id="sec-001",
        text=text,
        token_count=12,
        char_start=0,
        char_end=len(text),
        embedding_id="emb/chunk-001",
        metadata={},
    )


def _tuple(
    *,
    subject: str = "system",
    predicate: str = "validate",
    obj: str = "inputs",
    polarity: str = "affirmative",
    modality: str = "obligatory",
    qualifier: str = "",
) -> ClaimTuple:
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        qualifier=qualifier,
    )


# --- claim_id determinism ----------------------------------------------------


def test_claim_id_is_sha256_prefixed() -> None:
    cid = claim_id_for_tuple(
        doc_id="doc-1",
        chunk_id="chunk-001",
        tuple_=_tuple(),
    )
    assert cid.startswith("sha256:")
    # 64 hex chars after the prefix.
    assert len(cid) == len("sha256:") + 64


def test_claim_id_stable_across_calls() -> None:
    a = claim_id_for_tuple(doc_id="doc-1", chunk_id="chunk-001", tuple_=_tuple())
    b = claim_id_for_tuple(doc_id="doc-1", chunk_id="chunk-001", tuple_=_tuple())
    assert a == b


def test_claim_id_differs_when_doc_id_differs() -> None:
    a = claim_id_for_tuple(doc_id="doc-1", chunk_id="chunk-001", tuple_=_tuple())
    b = claim_id_for_tuple(doc_id="doc-2", chunk_id="chunk-001", tuple_=_tuple())
    assert a != b


def test_claim_id_differs_when_polarity_flips() -> None:
    a = claim_id_for_tuple(
        doc_id="doc-1", chunk_id="chunk-001", tuple_=_tuple(polarity="affirmative")
    )
    b = claim_id_for_tuple(doc_id="doc-1", chunk_id="chunk-001", tuple_=_tuple(polarity="negative"))
    assert a != b


def test_claim_id_differs_when_qualifier_differs() -> None:
    a = claim_id_for_tuple(doc_id="doc-1", chunk_id="chunk-001", tuple_=_tuple(qualifier=""))
    b = claim_id_for_tuple(
        doc_id="doc-1", chunk_id="chunk-001", tuple_=_tuple(qualifier="under high load")
    )
    assert a != b


# --- polarity mapping --------------------------------------------------------


def test_affirmative_polarity_maps_to_plus() -> None:
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=_chunk(),
        tuple_=_tuple(polarity="affirmative"),
    )
    assert claim.polarity == "+"


def test_negative_polarity_maps_to_minus() -> None:
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=_chunk(),
        tuple_=_tuple(polarity="negative"),
    )
    assert claim.polarity == "-"


# --- modality mapping --------------------------------------------------------


@pytest.mark.parametrize(
    ("surface", "persisted"),
    [
        ("asserted", "assert"),
        ("obligatory", "must"),
        ("recommended", "should"),
        ("permitted", "may"),
        ("prohibited", "neg"),
    ],
)
def test_modality_surface_to_modal_force(surface: str, persisted: str) -> None:
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=_chunk(),
        tuple_=_tuple(modality=surface),
    )
    assert claim.modality == persisted


def test_hypothetical_modality_maps_to_none() -> None:
    # §7 modal force is uncommitted; hypothetical force is meaningful at
    # extraction time but does not bind to any of the six modal-force
    # slots. `None` is the documented "modality was not bound" sentinel.
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=_chunk(),
        tuple_=_tuple(modality="hypothetical"),
    )
    assert claim.modality is None


# --- qualifier mapping -------------------------------------------------------


def test_empty_qualifier_is_empty_dict() -> None:
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=_chunk(),
        tuple_=_tuple(qualifier=""),
    )
    assert claim.qualifier == {}


def test_non_empty_qualifier_round_trips_under_text_key() -> None:
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=_chunk(),
        tuple_=_tuple(qualifier="under high load"),
    )
    assert claim.qualifier == {"text": "under high load"}


# --- span_ref grounding ------------------------------------------------------


def test_span_refs_anchor_one_span_per_chunk() -> None:
    text = "The system must validate inputs."
    chunk = _chunk(text=text)
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=chunk,
        tuple_=_tuple(),
    )
    # Exactly one span, anchoring the whole chunk — the SVO extractor is
    # sentence-level, the chunk is the smallest persisted unit, so a
    # one-span-per-chunk grounding is the conservative floor.
    assert len(claim.span_refs) == 1
    span = claim.span_refs[0]
    assert span.chunk_id == "chunk-001"
    assert span.char_start == 0
    assert span.char_end == len(text)
    assert span.text == text


# --- full shape --------------------------------------------------------------


def test_full_claim_round_trips_through_pydantic() -> None:
    chunk = _chunk()
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=chunk,
        tuple_=_tuple(),
    )
    # Round-trip — exercises the model_config(frozen=True, extra="forbid")
    # contract end-to-end.
    rehydrated = Claim.model_validate_json(claim.model_dump_json())
    assert rehydrated == claim


def test_claim_carries_doc_section_and_confidence_floor() -> None:
    claim = claim_from_tuple(
        doc_id="doc-7",
        chunk=_chunk(),
        tuple_=_tuple(),
    )
    assert claim.doc_id == "doc-7"
    assert claim.section_id == "sec-001"
    # Heuristic / Tier-2 SVO is deterministic; we ship it with a floor
    # confidence of 1.0 (pre-calibration). Tier-3 layers replace this
    # in later slices.
    assert claim.confidence == 1.0
    # Tier-2 ships no concept binding and no typed-slot induction; those
    # land in S-154/S-155.
    assert claim.concept_ids == []
    assert claim.typed_slots == {}


def test_claim_text_is_subject_predicate_object_surface() -> None:
    claim = claim_from_tuple(
        doc_id="doc-1",
        chunk=_chunk(),
        tuple_=_tuple(subject="alice", predicate="signs", obj="treaty"),
    )
    # The `text` field is the human-readable surface form; the canonical
    # logical content lives on subject / predicate / object.
    assert "alice" in claim.text
    assert "signs" in claim.text
    assert "treaty" in claim.text


# --- error surface -----------------------------------------------------------


def test_unknown_modality_raises_value_error() -> None:
    bad_tuple = _tuple()
    # Bypass the ClaimTuple validator with a model_copy that smuggles an
    # unknown modality — the adapter must reject it rather than silently
    # passing it through to the persisted alphabet.
    object.__setattr__(bad_tuple, "modality", "uncertain")
    with pytest.raises(ValueError, match="modality"):
        claim_from_tuple(doc_id="doc-1", chunk=_chunk(), tuple_=bad_tuple)
