"""Contract tests for L2..L5 output models.

These models are emitted by retrieval, verification, and the playbook
synthesisers. They carry citations and confidence scores that the rest
of the pipeline trusts, so each constraint here is load-bearing.

SPEC-REF: §4.0 (data model), §4.3 (retrieval evidence-pack cap)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.models import (
    Claim,
    EvidencePack,
    Finding,
    RelationEdge,
    Span,
    Verdict,
)


def _span(**overrides: object) -> Span:
    defaults: dict[str, object] = {
        "chunk_id": "chunk-1",
        "char_start": 0,
        "char_end": 5,
        "text": "hello",
    }
    defaults.update(overrides)
    return Span(**defaults)  # type: ignore[arg-type]


# --- EvidencePack ---


def test_evidence_pack_field_set() -> None:
    assert set(EvidencePack.model_fields) == {
        "query",
        "spans",
        "token_count",
        "retrieval_plan",
    }


def test_evidence_pack_token_count_cap() -> None:
    EvidencePack(query="q", spans=[], token_count=6000, retrieval_plan=[])
    with pytest.raises(ValidationError):
        EvidencePack(query="q", spans=[], token_count=6001, retrieval_plan=[])


def test_evidence_pack_negative_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidencePack(query="q", spans=[], token_count=-1, retrieval_plan=[])


def test_evidence_pack_round_trip() -> None:
    e = EvidencePack(
        query="what is X?",
        spans=[_span()],
        token_count=128,
        retrieval_plan=["search(query, view=dense, k=8)"],
    )
    assert EvidencePack.model_validate(e.model_dump()) == e


# --- Claim ---


def test_claim_field_set() -> None:
    assert set(Claim.model_fields) == {
        "text",
        "citations",
        "verified",
        "confidence",
        "nli_score",
        "judge_score",
    }


@pytest.mark.parametrize("field", ["confidence", "nli_score", "judge_score"])
@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_claim_scores_clamped_to_unit_interval(field: str, bad_value: float) -> None:
    kwargs = {
        "text": "x",
        "citations": [_span()],
        "verified": True,
        "confidence": 0.5,
        "nli_score": 0.5,
        "judge_score": 0.5,
    }
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        Claim(**kwargs)  # type: ignore[arg-type]


def test_claim_zero_and_one_scores_ok() -> None:
    Claim(
        text="x", citations=[_span()], verified=True, confidence=0.0, nli_score=1.0, judge_score=0.5
    )


# --- Verdict ---


def test_verdict_literal_values() -> None:
    for verdict in ("Covered", "Partial", "NotCovered", "Ambiguous"):
        Verdict(item_id="i", verdict=verdict, citations=[_span()], confidence=0.8)  # type: ignore[arg-type]


def test_verdict_unknown_literal_rejected() -> None:
    with pytest.raises(ValidationError):
        Verdict(item_id="i", verdict="MaybeCovered", citations=[], confidence=0.5)  # type: ignore[arg-type]


def test_verdict_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        Verdict(item_id="i", verdict="Covered", citations=[], confidence=1.5)


# --- Finding ---


def test_finding_field_set() -> None:
    assert set(Finding.model_fields) == {"ctrldoc", "location", "claim", "severity"}


def test_finding_severity_literal() -> None:
    for severity in ("info", "warn", "critical"):
        Finding(ctrldoc="assumptions", location=_span(), claim="x", severity=severity)  # type: ignore[arg-type]


def test_finding_unknown_severity_rejected() -> None:
    with pytest.raises(ValidationError):
        Finding(ctrldoc="assumptions", location=_span(), claim="x", severity="urgent")  # type: ignore[arg-type]


# --- RelationEdge ---


_RELATION_TYPES = (
    "depends_on",
    "contradicts",
    "refines",
    "instantiates",
    "conflicts_with",
    "prerequisite_of",
    "alternative_to",
)


@pytest.mark.parametrize("relation_type", _RELATION_TYPES)
def test_relation_edge_accepts_all_spec_types(relation_type: str) -> None:
    RelationEdge(
        src_concept="A",
        dst_concept="B",
        type=relation_type,  # type: ignore[arg-type]
        citations=[_span()],
        confidence=0.7,
    )


def test_relation_edge_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        RelationEdge(
            src_concept="A",
            dst_concept="B",
            type="related_to",  # type: ignore[arg-type]
            citations=[_span()],
            confidence=0.5,
        )


def test_relation_edge_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        RelationEdge(
            src_concept="A",
            dst_concept="B",
            type="depends_on",
            citations=[_span()],
            confidence=-0.1,
        )


# --- shared invariants ---


def _example(cls: type) -> object:
    if cls is EvidencePack:
        return EvidencePack(query="q", spans=[], token_count=0, retrieval_plan=[])
    if cls is Claim:
        return Claim(
            text="x",
            citations=[_span()],
            verified=False,
            confidence=0.5,
            nli_score=0.5,
            judge_score=0.5,
        )
    if cls is Verdict:
        return Verdict(item_id="i", verdict="Covered", citations=[_span()], confidence=0.5)
    if cls is Finding:
        return Finding(ctrldoc="assumptions", location=_span(), claim="x", severity="info")
    if cls is RelationEdge:
        return RelationEdge(
            src_concept="A", dst_concept="B", type="depends_on", citations=[_span()], confidence=0.5
        )
    raise AssertionError(cls)


@pytest.mark.parametrize("cls", [EvidencePack, Claim, Verdict, Finding, RelationEdge])
def test_models_reject_extra_fields(cls: type) -> None:
    obj = _example(cls)
    payload = obj.model_dump()  # type: ignore[attr-defined]
    payload["bogus"] = "no"
    with pytest.raises(ValidationError):
        cls.model_validate(payload)  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", [EvidencePack, Claim, Verdict, Finding, RelationEdge])
def test_models_are_frozen(cls: type) -> None:
    obj = _example(cls)
    with pytest.raises(ValidationError):
        next_field = next(iter(cls.model_fields))  # type: ignore[attr-defined]
        setattr(obj, next_field, "tampered")


@pytest.mark.parametrize("cls", [EvidencePack, Claim, Verdict, Finding, RelationEdge])
def test_models_round_trip(cls: type) -> None:
    obj = _example(cls)
    restored = cls.model_validate(obj.model_dump())  # type: ignore[attr-defined]
    assert restored == obj
