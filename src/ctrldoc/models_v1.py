"""v1 universal-substrate data-model contracts.

The v1 substrate adds six in-memory shapes on top of the v0.3 models
in `ctrldoc.models`: `Claim`, `Concept`, `TypedEdge`, `Workspace`,
plus `CoverageReport` / `CoverageVerdict` (with their `CoverageSummary`
and `ProofTrace` companions). They are the mirror of the SQL schema
provisioned at S-125 (`store_schema_v2`) and the calibrated-edge graph
the universal-transport operations (§6.6) traverse.

These models live in their own module so the v0.3 surface in
`ctrldoc.models` keeps working unchanged through the v1 build-out;
once the playbook layer collapses (Phase 22) the two modules are
consolidated. Until then the storage layer and the L1.5/L2.5/L5
modules import from here.

The `Claim` shape implemented here is the §7 record — the persisted
claim-graph node with `id`, `doc_id`, `span_refs`, `concept_ids`,
`typed_slots`, and `confidence`. It is a superset of the universal
claim tuple from §6.2 (`ClaimTuple` in `ctrldoc.eval.claim_extraction`)
which carries only the six logical fields needed for extraction
scoring.

SPEC-REF: §7 (data model additions)
"""

from __future__ import annotations

import builtins
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)

from ctrldoc.models import Span, UnitInterval
from ctrldoc.provenance import Provenance

# Local alias so `dict[str, _Value]` survives on classes that have an
# `object: ...` field (`Claim`). Without it mypy resolves bare `object`
# to the class attribute and rejects it as a non-type.
_Value: TypeAlias = builtins.object

PolarityLiteral: TypeAlias = Literal["+", "-"]
"""Universal-tuple polarity from §6.2: affirmative vs negated."""

ModalityLiteral: TypeAlias = Literal[
    "assert",
    "must",
    "may",
    "should",
    "shall",
    "neg",
]
"""Modal force on a claim from §7. `None` means modality was not bound."""

PrimitiveTypeLiteral: TypeAlias = Literal[
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
]
"""The closed 10-element atomic library every per-doc schema co-induces from."""

TypedEdgeTypeLiteral: TypeAlias = Literal[
    # Intra-document edges.
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
    # Cross-document edges (§6.7).
    "aligned_with",
    "entails_across",
    "contradicts_across",
    "stronger_than",
]
"""The fixed type alphabet for `TypedEdge`. Intra- and cross-doc unified."""

EdgeSourceLiteral: TypeAlias = Literal["heuristic", "nli", "llm", "induction"]
"""Provenance of a `TypedEdge`'s `raw_score`."""

VerdictLiteral: TypeAlias = Literal["Covered", "Partial", "Missing", "Contradicted"]
"""The four-class per-claim verdict the optimal-transport core emits."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProofTrace(_Strict):
    """The replayable chain that produced one verdict.

    Step strings are intentionally opaque here — each operation owns its
    own step vocabulary (e.g. `retrieve`, `nli_entail`, `transport`,
    `calibrate`). What matters at the data-model layer is that the
    trace is a list — ordered, citable, and append-only — so §13
    non-negotiable 4 (every verdict is replayable) holds.
    """

    steps: list[str]


class Claim(_Strict):
    """A persisted claim-graph node — the §7 superset of the §6.2 tuple.

    Carries the six logical slots (subject / predicate / object / polarity
    / modality / qualifier) plus the bookkeeping the universal substrate
    needs: a content-hashed `id`, the parent `doc_id` and `section_id`,
    the `span_refs` it grounds in, the `concept_ids` it binds to, the
    adapter-specific `typed_slots` co-induced under §6.4, and the
    extraction `confidence`.
    """

    id: str
    doc_id: str
    text: str
    subject: str | None
    predicate: str
    object: str | None
    polarity: PolarityLiteral
    modality: ModalityLiteral | None
    qualifier: dict[str, _Value]
    span_refs: list[Span]
    section_id: str
    concept_ids: list[str]
    typed_slots: dict[str, _Value]
    confidence: UnitInterval


class Concept(_Strict):
    """A canonical-cluster node — one row in the per-workspace concept lattice.

    `mention_claim_ids` and `doc_ids` are kept on the concept so the
    Galois subsumption walker (§6.3) can answer "which docs cover this
    concept?" without re-querying the edge table.
    """

    id: str
    canonical_name: str
    aliases: list[str]
    primitive_type: PrimitiveTypeLiteral
    mention_claim_ids: list[str]
    doc_ids: list[str]


class TypedEdge(_Strict):
    """A calibrated edge in the claim graph (§6.5).

    `raw_score` is the pre-calibration model output (NLI logit, LLM
    self-reported probability, or 1.0 for deterministic heuristics).
    `confidence` is the post-isotonic-regression calibrated probability
    the orchestrator and the verdict ledger trust. `paraphrase_votes`
    is `None` for non-NLI sources (heuristic, LLM, induction); the
    field is required to be non-negative when present.
    """

    src_id: str
    dst_id: str
    type: TypedEdgeTypeLiteral
    confidence: UnitInterval
    raw_score: float
    citations: list[Span]
    source: EdgeSourceLiteral
    paraphrase_votes: Annotated[NonNegativeInt, Field(strict=False)] | None


class Workspace(_Strict):
    """A shared latent ontology over N docs (§6.7).

    `induced_schema` is the YAML-cached schema from §6.4 — the per-doc
    typed nodes/edges merged into the workspace-level co-induced
    ontology. Stored as `dict[str, object]` so it round-trips through
    JSON without an extra Pydantic model per primitive.
    """

    id: str
    name: str
    doc_ids: list[str]
    induced_schema: dict[str, object]
    provenance: Provenance


class CoverageSummary(_Strict):
    """Aggregate rates across a `CoverageReport`'s `per_claim` verdicts.

    The four rates partition the target-doc claims (one verdict per
    claim), so they must sum to 1.0 within a small floating-point
    tolerance. The summary is what dashboards and the MCP `coverage`
    tool surface; the per-claim verdicts are the audit trail.
    """

    covered_rate: UnitInterval
    partial_rate: UnitInterval
    missing_rate: UnitInterval
    contradicted_rate: UnitInterval

    @model_validator(mode="after")
    def _rates_sum_to_one(self) -> CoverageSummary:
        total = self.covered_rate + self.partial_rate + self.missing_rate + self.contradicted_rate
        # 1e-6 tolerance is well below any reportable precision.
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"coverage rates must sum to 1.0; got "
                f"{total} = "
                f"{self.covered_rate} + {self.partial_rate} "
                f"+ {self.missing_rate} + {self.contradicted_rate}"
            )
        return self


class CoverageVerdict(_Strict):
    """One row in a `CoverageReport.per_claim` list.

    `aligned_source_claims` lists the source-doc claims the transport
    engine matched against this target claim (zero for `Missing`).
    `transport_cost` is the min-cost-flow edge weight at this assignment;
    `calibrated_confidence` is the post-paraphrase-vote calibrated
    probability the verdict ledger persists.
    """

    target_claim_id: str
    verdict: VerdictLiteral
    aligned_source_claims: list[str]
    transport_cost: float
    calibrated_confidence: UnitInterval
    trace: ProofTrace


class CoverageReport(_Strict):
    """The full output of a `coverage(target, source)` call (§6.6).

    Carries both the per-claim verdicts (audit trail) and the summary
    rates (dashboard). Pinned to a `workspace_id` so cross-doc edges
    can be replayed against the same shared ontology.
    """

    workspace_id: str
    target_doc_id: str
    source_doc_id: str
    per_claim: list[CoverageVerdict]
    summary: CoverageSummary


__all__ = [
    "Claim",
    "Concept",
    "CoverageReport",
    "CoverageSummary",
    "CoverageVerdict",
    "EdgeSourceLiteral",
    "ModalityLiteral",
    "PolarityLiteral",
    "PrimitiveTypeLiteral",
    "ProofTrace",
    "TypedEdge",
    "TypedEdgeTypeLiteral",
    "VerdictLiteral",
    "Workspace",
]
