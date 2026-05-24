"""Contract tests for the v1 universal-substrate Pydantic models.

These models (`Claim`, `Concept`, `TypedEdge`, `Workspace`,
`CoverageReport`, `CoverageVerdict`) are the data-model surface that
the storage layer (`store_schema_v2`), the claim graph (L1.5), the
workspace primitive (L2.5), and the universal-transport operations
(L5) read and write. They are the in-memory mirror of the v2 SQL
schema landed in S-125, and the calibrated edges + coverage verdicts
that flow out of every cross-doc operation.

The constraints pinned here are load-bearing: every field name, the
primitive-type and edge-type literal sets, the unit-interval bound on
all confidences, polarity / modality literals, and the strict / frozen
`extra='forbid'` shape every model carries.

SPEC-REF: §7 (data model additions)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctrldoc.models import Span
from ctrldoc.models_v1 import (
    Claim,
    Concept,
    CoverageReport,
    CoverageSummary,
    CoverageVerdict,
    EdgeSourceLiteral,
    ModalityLiteral,
    PolarityLiteral,
    PrimitiveTypeLiteral,
    ProofTrace,
    TypedEdge,
    TypedEdgeTypeLiteral,
    VerdictLiteral,
    Workspace,
)
from ctrldoc.provenance import Provenance


def _span(**overrides: object) -> Span:
    defaults: dict[str, object] = {
        "chunk_id": "chunk-1",
        "char_start": 0,
        "char_end": 5,
        "text": "hello",
    }
    defaults.update(overrides)
    return Span(**defaults)  # type: ignore[arg-type]


def _provenance() -> Provenance:
    return Provenance.create(
        playbook="workspace",
        playbook_version="1.0.0",
        index_hash="sha256:deadbeef",
        models={"judge": "qwen2.5:7b-instruct"},
    )


# --- Claim ---


def test_claim_field_set() -> None:
    assert set(Claim.model_fields) == {
        "id",
        "doc_id",
        "text",
        "subject",
        "predicate",
        "object",
        "polarity",
        "modality",
        "qualifier",
        "span_refs",
        "section_id",
        "concept_ids",
        "typed_slots",
        "confidence",
    }


def test_claim_round_trip() -> None:
    claim = Claim(
        id="sha256:abc",
        doc_id="doc-1",
        text="The server must restart on signal.",
        subject="server",
        predicate="restart_on",
        object="signal",
        polarity="+",
        modality="must",
        qualifier={"timeout_s": 30},
        span_refs=[_span()],
        section_id="sec-1",
        concept_ids=["concept-server"],
        typed_slots={"timepoint": "T+0"},
        confidence=0.91,
    )
    assert claim.modality == "must"
    assert claim.qualifier == {"timeout_s": 30}
    # Frozen — mutation rejected.
    with pytest.raises(ValidationError):
        claim.text = "mutated"  # type: ignore[misc]


def test_claim_optional_fields_default_friendly() -> None:
    # subject / object / modality are optional (Span-only adapters may
    # not bind them); polarity and predicate are mandatory.
    Claim(
        id="sha256:zero",
        doc_id="d",
        text="raw",
        subject=None,
        predicate="exists",
        object=None,
        polarity="+",
        modality=None,
        qualifier={},
        span_refs=[],
        section_id="sec",
        concept_ids=[],
        typed_slots={},
        confidence=1.0,
    )


def test_claim_polarity_must_be_plus_or_minus() -> None:
    with pytest.raises(ValidationError):
        Claim(
            id="x",
            doc_id="d",
            text="t",
            subject=None,
            predicate="p",
            object=None,
            polarity="?",  # type: ignore[arg-type]
            modality=None,
            qualifier={},
            span_refs=[],
            section_id="sec",
            concept_ids=[],
            typed_slots={},
            confidence=0.5,
        )


def test_claim_modality_allowed_set() -> None:
    allowed = {"assert", "must", "may", "should", "shall", "neg"}
    # Round-trip each one to be sure.
    for modality in allowed:
        Claim(
            id="x",
            doc_id="d",
            text="t",
            subject=None,
            predicate="p",
            object=None,
            polarity="+",
            modality=modality,  # type: ignore[arg-type]
            qualifier={},
            span_refs=[],
            section_id="sec",
            concept_ids=[],
            typed_slots={},
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        Claim(
            id="x",
            doc_id="d",
            text="t",
            subject=None,
            predicate="p",
            object=None,
            polarity="+",
            modality="recommended",  # type: ignore[arg-type]
            qualifier={},
            span_refs=[],
            section_id="sec",
            concept_ids=[],
            typed_slots={},
            confidence=0.5,
        )


def test_claim_confidence_must_be_unit_interval() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            Claim(
                id="x",
                doc_id="d",
                text="t",
                subject=None,
                predicate="p",
                object=None,
                polarity="+",
                modality=None,
                qualifier={},
                span_refs=[],
                section_id="sec",
                concept_ids=[],
                typed_slots={},
                confidence=bad,
            )


def test_claim_polarity_modality_literals_exported() -> None:
    """The literal aliases are public — verifier modules import them."""
    assert PolarityLiteral is not None
    assert ModalityLiteral is not None


def test_claim_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        Claim(  # type: ignore[call-arg]
            id="x",
            doc_id="d",
            text="t",
            subject=None,
            predicate="p",
            object=None,
            polarity="+",
            modality=None,
            qualifier={},
            span_refs=[],
            section_id="sec",
            concept_ids=[],
            typed_slots={},
            confidence=0.5,
            unknown_field=123,
        )


# --- Concept ---


def test_concept_field_set() -> None:
    assert set(Concept.model_fields) == {
        "id",
        "canonical_name",
        "aliases",
        "primitive_type",
        "mention_claim_ids",
        "doc_ids",
    }


def test_concept_round_trip() -> None:
    concept = Concept(
        id="concept-server",
        canonical_name="Application server",
        aliases=["server", "app server"],
        primitive_type="Entity",
        mention_claim_ids=["claim-1", "claim-2"],
        doc_ids=["doc-1"],
    )
    assert concept.primitive_type == "Entity"


def test_concept_primitive_type_allowed_set() -> None:
    """The atomic library is closed at 10 primitives per SPEC §7."""
    allowed: set[str] = {
        "Entity",
        "Event",
        "Process",
        "Property",
        "Quantity",
        "Definition",
        "Assertion",
        "Obligation",
        "Citation",
        "Relation",
    }
    for primitive in allowed:
        Concept(
            id=f"c-{primitive}",
            canonical_name=primitive.lower(),
            aliases=[],
            primitive_type=primitive,  # type: ignore[arg-type]
            mention_claim_ids=[],
            doc_ids=[],
        )
    with pytest.raises(ValidationError):
        Concept(
            id="bad",
            canonical_name="x",
            aliases=[],
            primitive_type="Animal",  # type: ignore[arg-type]
            mention_claim_ids=[],
            doc_ids=[],
        )


def test_concept_primitive_type_literal_exported() -> None:
    assert PrimitiveTypeLiteral is not None


# --- TypedEdge ---


def test_typed_edge_field_set() -> None:
    assert set(TypedEdge.model_fields) == {
        "src_id",
        "dst_id",
        "type",
        "confidence",
        "raw_score",
        "citations",
        "source",
        "paraphrase_votes",
    }


def test_typed_edge_intra_doc_types_allowed() -> None:
    intra_types: tuple[str, ...] = (
        "entails",
        "contradicts",
        "refines",
        "instantiates",
        "depends_on",
        "prerequisite_of",
        "part_of",
        "is_a",
        "example_of",
        "alternative_to",
        "equivalent_to",
        "related_to",
    )
    for edge_type in intra_types:
        TypedEdge(
            src_id="a",
            dst_id="b",
            type=edge_type,  # type: ignore[arg-type]
            confidence=0.8,
            raw_score=0.7,
            citations=[],
            source="nli",
            paraphrase_votes=3,
        )


def test_typed_edge_cross_doc_types_allowed() -> None:
    """The cross-doc edge types from §6.7 must validate on the same model."""
    cross_types: tuple[str, ...] = (
        "aligned_with",
        "entails_across",
        "contradicts_across",
        "stronger_than",
    )
    for edge_type in cross_types:
        TypedEdge(
            src_id="a",
            dst_id="b",
            type=edge_type,  # type: ignore[arg-type]
            confidence=0.7,
            raw_score=0.65,
            citations=[],
            source="nli",
            paraphrase_votes=None,
        )


def test_typed_edge_type_rejected_when_unknown() -> None:
    with pytest.raises(ValidationError):
        TypedEdge(
            src_id="a",
            dst_id="b",
            type="dances_with",  # type: ignore[arg-type]
            confidence=0.5,
            raw_score=0.4,
            citations=[],
            source="nli",
            paraphrase_votes=None,
        )


def test_typed_edge_source_must_be_known() -> None:
    for source in ("heuristic", "nli", "llm", "induction"):
        TypedEdge(
            src_id="a",
            dst_id="b",
            type="entails",
            confidence=0.6,
            raw_score=0.5,
            citations=[],
            source=source,  # type: ignore[arg-type]
            paraphrase_votes=None,
        )
    with pytest.raises(ValidationError):
        TypedEdge(
            src_id="a",
            dst_id="b",
            type="entails",
            confidence=0.6,
            raw_score=0.5,
            citations=[],
            source="cosmic_rays",  # type: ignore[arg-type]
            paraphrase_votes=None,
        )


def test_typed_edge_confidence_unit_interval() -> None:
    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            TypedEdge(
                src_id="a",
                dst_id="b",
                type="entails",
                confidence=bad,
                raw_score=0.0,
                citations=[],
                source="nli",
                paraphrase_votes=None,
            )


def test_typed_edge_paraphrase_votes_optional() -> None:
    TypedEdge(
        src_id="a",
        dst_id="b",
        type="entails",
        confidence=0.9,
        raw_score=0.85,
        citations=[],
        source="nli",
        paraphrase_votes=None,
    )


def test_typed_edge_paraphrase_votes_nonneg() -> None:
    with pytest.raises(ValidationError):
        TypedEdge(
            src_id="a",
            dst_id="b",
            type="entails",
            confidence=0.9,
            raw_score=0.85,
            citations=[],
            source="nli",
            paraphrase_votes=-1,
        )


def test_typed_edge_type_literals_exported() -> None:
    assert TypedEdgeTypeLiteral is not None
    assert EdgeSourceLiteral is not None


# --- Workspace ---


def test_workspace_field_set() -> None:
    assert set(Workspace.model_fields) == {
        "id",
        "name",
        "doc_ids",
        "induced_schema",
        "provenance",
    }


def test_workspace_round_trip() -> None:
    workspace = Workspace(
        id="ws-1",
        name="spec-vs-impl",
        doc_ids=["doc-spec", "doc-impl"],
        induced_schema={"Entity": ["server", "client"]},
        provenance=_provenance(),
    )
    assert workspace.doc_ids == ["doc-spec", "doc-impl"]
    assert workspace.induced_schema == {"Entity": ["server", "client"]}


# --- CoverageVerdict + CoverageReport + CoverageSummary ---


def test_coverage_verdict_field_set() -> None:
    assert set(CoverageVerdict.model_fields) == {
        "target_claim_id",
        "verdict",
        "aligned_source_claims",
        "transport_cost",
        "calibrated_confidence",
        "trace",
    }


def test_coverage_verdict_literals() -> None:
    for verdict in ("Covered", "Partial", "Missing", "Contradicted"):
        CoverageVerdict(
            target_claim_id="t1",
            verdict=verdict,  # type: ignore[arg-type]
            aligned_source_claims=[],
            transport_cost=0.1,
            calibrated_confidence=0.9,
            trace=ProofTrace(steps=[]),
        )
    with pytest.raises(ValidationError):
        CoverageVerdict(
            target_claim_id="t1",
            verdict="MaybeCovered",  # type: ignore[arg-type]
            aligned_source_claims=[],
            transport_cost=0.1,
            calibrated_confidence=0.9,
            trace=ProofTrace(steps=[]),
        )


def test_coverage_verdict_confidence_unit_interval() -> None:
    with pytest.raises(ValidationError):
        CoverageVerdict(
            target_claim_id="t1",
            verdict="Covered",
            aligned_source_claims=[],
            transport_cost=0.0,
            calibrated_confidence=1.5,
            trace=ProofTrace(steps=[]),
        )


def test_coverage_summary_rates_unit_interval() -> None:
    CoverageSummary(
        covered_rate=0.4,
        partial_rate=0.3,
        missing_rate=0.2,
        contradicted_rate=0.1,
    )
    with pytest.raises(ValidationError):
        CoverageSummary(
            covered_rate=1.5,
            partial_rate=0.0,
            missing_rate=0.0,
            contradicted_rate=0.0,
        )


def test_coverage_summary_rates_must_sum_to_one() -> None:
    """Verdicts partition the target claims — rates form a probability mass."""
    with pytest.raises(ValidationError):
        CoverageSummary(
            covered_rate=0.5,
            partial_rate=0.5,
            missing_rate=0.5,
            contradicted_rate=0.0,
        )


def test_coverage_report_field_set() -> None:
    assert set(CoverageReport.model_fields) == {
        "workspace_id",
        "target_doc_id",
        "source_doc_id",
        "per_claim",
        "summary",
    }


def test_coverage_report_round_trip() -> None:
    verdict = CoverageVerdict(
        target_claim_id="t1",
        verdict="Covered",
        aligned_source_claims=["s1"],
        transport_cost=0.05,
        calibrated_confidence=0.92,
        trace=ProofTrace(steps=["retrieve", "nli_entail", "calibrate"]),
    )
    summary = CoverageSummary(
        covered_rate=1.0,
        partial_rate=0.0,
        missing_rate=0.0,
        contradicted_rate=0.0,
    )
    report = CoverageReport(
        workspace_id="ws-1",
        target_doc_id="doc-target",
        source_doc_id="doc-source",
        per_claim=[verdict],
        summary=summary,
    )
    assert report.summary.covered_rate == pytest.approx(1.0)
    assert report.per_claim[0].verdict == "Covered"


def test_coverage_verdict_literal_exported() -> None:
    assert VerdictLiteral is not None
