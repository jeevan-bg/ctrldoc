"""Tests for the §6.4 residual EM loop wrapping the schema proposer.

The EM loop measures `unmatched_claim_rate` against a `SchemaProposal`
and, when the rate exceeds the configurable `tau_residual` threshold,
re-proposes a refined schema using the residual (unbound) claims as
extra evidence, then re-extracts only over the sections that contained
those residuals. The loop terminates when residual <= tau or when the
maximum iteration count is reached.

SPEC-REF: §6.4
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import pytest

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.residual_em import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TAU_RESIDUAL,
    BoundClaim,
    ClaimBinder,
    EMOutcome,
    LexicalClaimBinder,
    ResidualEMLoop,
    SchemaReProposer,
    SectionExtractor,
    residual_rate,
)
from ctrldoc.extract.schema_proposer import (
    SchemaProposal,
    TypedEdgeSpec,
    TypedNodeSpec,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _claim(subject: str, predicate: str, obj: str) -> ClaimTuple:
    return ClaimTuple(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity="affirmative",
        modality="asserted",
    )


def _bound(section_id: str, claim: ClaimTuple) -> BoundClaim:
    return BoundClaim(section_id=section_id, claim=claim)


def _empty_proposal() -> SchemaProposal:
    return SchemaProposal(nodes=[], edges=[])


def _proposal_with_widget() -> SchemaProposal:
    return SchemaProposal(
        nodes=[
            TypedNodeSpec(name="Widget", primitive="Entity", description="A widget"),
        ],
        edges=[
            TypedEdgeSpec(
                name="exposes",
                subject_type="Widget",
                object_type="Widget",
                description="exposes",
            )
        ],
    )


# ---------------------------------------------------------------------------
# residual_rate — the raw observable
# ---------------------------------------------------------------------------


def test_residual_rate_is_one_for_empty_schema() -> None:
    """An empty schema binds nothing; every claim is a residual."""
    claims = [
        _bound("s0", _claim("widget", "exposes", "api")),
        _bound("s1", _claim("api", "returns", "json")),
    ]
    binder = LexicalClaimBinder()
    assert residual_rate(bound_claims=claims, proposal=_empty_proposal(), binder=binder) == 1.0


def test_residual_rate_is_zero_when_all_claims_bind() -> None:
    """A schema covering every claim has residual 0.0."""
    claims = [_bound("s0", _claim("widget", "exposes", "widget"))]
    binder = LexicalClaimBinder()
    assert (
        residual_rate(bound_claims=claims, proposal=_proposal_with_widget(), binder=binder) == 0.0
    )


def test_residual_rate_handles_partial_binding() -> None:
    """Half the claims bind → residual 0.5."""
    claims = [
        _bound("s0", _claim("widget", "exposes", "widget")),
        _bound("s1", _claim("foo", "bars", "baz")),
    ]
    binder = LexicalClaimBinder()
    assert (
        residual_rate(bound_claims=claims, proposal=_proposal_with_widget(), binder=binder) == 0.5
    )


def test_residual_rate_zero_when_no_claims() -> None:
    """No claims to bind ⇒ residual defined as 0.0 (vacuously satisfied)."""
    binder = LexicalClaimBinder()
    assert residual_rate(bound_claims=[], proposal=_empty_proposal(), binder=binder) == 0.0


# ---------------------------------------------------------------------------
# LexicalClaimBinder — the default token-overlap binder
# ---------------------------------------------------------------------------


def test_lexical_binder_binds_when_subject_contains_node_name() -> None:
    binder = LexicalClaimBinder()
    proposal = SchemaProposal(
        nodes=[TypedNodeSpec(name="Widget", primitive="Entity", description="x")],
        edges=[],
    )
    assert binder.binds(claim=_claim("the widget", "is", "blue"), proposal=proposal)


def test_lexical_binder_binds_when_object_contains_node_name() -> None:
    binder = LexicalClaimBinder()
    proposal = SchemaProposal(
        nodes=[TypedNodeSpec(name="Widget", primitive="Entity", description="x")],
        edges=[],
    )
    assert binder.binds(claim=_claim("alice", "owns", "a widget"), proposal=proposal)


def test_lexical_binder_binds_when_predicate_matches_edge_name() -> None:
    binder = LexicalClaimBinder()
    proposal = SchemaProposal(
        nodes=[],
        edges=[
            TypedEdgeSpec(
                name="exposes",
                subject_type="X",
                object_type="Y",
                description="d",
            )
        ],
    )
    assert binder.binds(claim=_claim("alice", "exposes", "bob"), proposal=proposal)


def test_lexical_binder_does_not_bind_unrelated_claim() -> None:
    binder = LexicalClaimBinder()
    proposal = _proposal_with_widget()
    assert not binder.binds(claim=_claim("alice", "ate", "lunch"), proposal=proposal)


def test_lexical_binder_is_case_insensitive() -> None:
    binder = LexicalClaimBinder()
    proposal = SchemaProposal(
        nodes=[TypedNodeSpec(name="Widget", primitive="Entity", description="x")],
        edges=[],
    )
    assert binder.binds(claim=_claim("WIDGET", "is", "round"), proposal=proposal)


def test_lexical_binder_requires_full_token_match() -> None:
    """`widgets` must not bind to `Widget` — token overlap, not substring."""
    binder = LexicalClaimBinder()
    proposal = SchemaProposal(
        nodes=[TypedNodeSpec(name="Widget", primitive="Entity", description="x")],
        edges=[],
    )
    assert not binder.binds(claim=_claim("widgets", "are", "many"), proposal=proposal)


# ---------------------------------------------------------------------------
# ResidualEMLoop — convergence machinery
# ---------------------------------------------------------------------------


@dataclass
class _StubReProposer:
    """Returns the same scripted proposals, recording every call's evidence."""

    proposals: list[SchemaProposal]
    received_residuals: list[list[ClaimTuple]] = field(default_factory=list)
    call_count: int = 0

    def re_propose(
        self,
        *,
        previous: SchemaProposal,
        residual_claims: list[ClaimTuple],
    ) -> SchemaProposal:
        self.received_residuals.append(list(residual_claims))
        proposal = self.proposals[self.call_count]
        self.call_count += 1
        return proposal


@dataclass
class _StubExtractor:
    """Returns per-section claims from a fixed map, recording every call."""

    per_section: dict[str, list[ClaimTuple]]
    received_section_ids: list[list[str]] = field(default_factory=list)

    def extract_sections(self, *, section_ids: list[str]) -> list[BoundClaim]:
        self.received_section_ids.append(list(section_ids))
        out: list[BoundClaim] = []
        for sid in section_ids:
            for c in self.per_section.get(sid, []):
                out.append(_bound(sid, c))
        return out


def test_loop_returns_immediately_when_initial_residual_is_below_tau() -> None:
    """No re-propose call when the floor already binds enough claims."""
    initial = [
        _bound("s0", _claim("widget", "exposes", "widget")),
        _bound("s1", _claim("widget", "exposes", "widget")),
    ]
    proposer = _StubReProposer(proposals=[])
    extractor = _StubExtractor(per_section={})
    loop = ResidualEMLoop(
        re_proposer=proposer,
        extractor=extractor,
        binder=LexicalClaimBinder(),
        tau_residual=0.2,
    )

    outcome = loop.run(initial_proposal=_proposal_with_widget(), initial_claims=initial)

    assert proposer.call_count == 0
    assert extractor.received_section_ids == []
    assert outcome.iterations == 0
    assert outcome.converged is True
    assert outcome.final_residual_rate == 0.0


def test_loop_re_proposes_when_residual_exceeds_tau() -> None:
    """A 1.0 residual triggers one re-proposal, which then binds everything."""
    initial = [_bound("s0", _claim("foo", "bars", "baz"))]
    refined = _proposal_with_widget()
    proposer = _StubReProposer(proposals=[refined])
    # After re-extraction over section s0 the extractor now emits a claim
    # that the refined schema binds — convergence in one iteration.
    extractor = _StubExtractor(per_section={"s0": [_claim("widget", "exposes", "widget")]})
    loop = ResidualEMLoop(
        re_proposer=proposer,
        extractor=extractor,
        binder=LexicalClaimBinder(),
        tau_residual=0.2,
        max_iterations=3,
    )

    outcome = loop.run(initial_proposal=_empty_proposal(), initial_claims=initial)

    assert proposer.call_count == 1
    assert extractor.received_section_ids == [["s0"]]
    assert outcome.iterations == 1
    assert outcome.converged is True
    assert outcome.final_residual_rate == 0.0
    assert outcome.final_proposal == refined


def test_loop_passes_only_unbound_claims_as_evidence() -> None:
    """Re-propose evidence carries the residual subset, not every claim."""
    initial = [
        _bound("s0", _claim("widget", "exposes", "widget")),  # bound by initial
        _bound("s1", _claim("foo", "bars", "baz")),  # unbound — residual
        _bound("s2", _claim("alpha", "betas", "gamma")),  # unbound — residual
    ]
    refined = _proposal_with_widget()
    proposer = _StubReProposer(proposals=[refined])
    extractor = _StubExtractor(per_section={})  # second pass yields nothing new
    loop = ResidualEMLoop(
        re_proposer=proposer,
        extractor=extractor,
        binder=LexicalClaimBinder(),
        tau_residual=0.2,
        max_iterations=3,
    )

    loop.run(initial_proposal=_proposal_with_widget(), initial_claims=initial)

    assert proposer.call_count == 1
    sent = proposer.received_residuals[0]
    assert {c.subject for c in sent} == {"foo", "alpha"}


def test_loop_re_extracts_only_over_affected_sections() -> None:
    """§6.4 step 4: re-extract scoped to sections containing residual claims."""
    initial = [
        _bound("s0", _claim("widget", "exposes", "widget")),  # bound
        _bound("s1", _claim("foo", "bars", "baz")),  # unbound
        _bound("s2", _claim("widget", "exposes", "widget")),  # bound
        _bound("s3", _claim("delta", "echos", "fox")),  # unbound
    ]
    refined = _proposal_with_widget()
    proposer = _StubReProposer(proposals=[refined])
    extractor = _StubExtractor(per_section={})
    loop = ResidualEMLoop(
        re_proposer=proposer,
        extractor=extractor,
        binder=LexicalClaimBinder(),
        tau_residual=0.0,
        max_iterations=1,
    )

    loop.run(initial_proposal=_proposal_with_widget(), initial_claims=initial)

    assert extractor.received_section_ids == [["s1", "s3"]]


def test_loop_stops_at_max_iterations_when_residual_never_falls() -> None:
    """If the schema can't be refined to bind everything, terminate at max_iters."""
    initial = [_bound("s0", _claim("foo", "bars", "baz"))]
    # Every re-proposal is still empty → residual stays at 1.0 forever.
    proposer = _StubReProposer(proposals=[_empty_proposal(), _empty_proposal(), _empty_proposal()])
    extractor = _StubExtractor(per_section={"s0": [_claim("foo", "bars", "baz")]})
    loop = ResidualEMLoop(
        re_proposer=proposer,
        extractor=extractor,
        binder=LexicalClaimBinder(),
        tau_residual=0.2,
        max_iterations=3,
    )

    outcome = loop.run(initial_proposal=_empty_proposal(), initial_claims=initial)

    assert outcome.iterations == 3
    assert outcome.converged is False
    assert outcome.final_residual_rate == 1.0
    assert proposer.call_count == 3


def test_loop_preserves_originally_bound_claims_across_iterations() -> None:
    """A claim bound in iteration 0 must stay in the converged corpus."""
    initial = [
        _bound("s0", _claim("widget", "exposes", "widget")),  # bound, in s0
        _bound("s1", _claim("foo", "bars", "baz")),  # unbound, s1
    ]
    proposer = _StubReProposer(proposals=[_proposal_with_widget()])
    # s1 re-extraction yields a new claim that the refined schema still
    # binds (because `widget` is its subject).
    extractor = _StubExtractor(per_section={"s1": [_claim("widget", "exposes", "widget")]})
    loop = ResidualEMLoop(
        re_proposer=proposer,
        extractor=extractor,
        binder=LexicalClaimBinder(),
        tau_residual=0.2,
        max_iterations=2,
    )

    outcome = loop.run(initial_proposal=_proposal_with_widget(), initial_claims=initial)

    section_ids_final = {b.section_id for b in outcome.final_claims}
    assert section_ids_final == {"s0", "s1"}
    assert outcome.converged is True
    assert outcome.final_residual_rate == 0.0


def test_loop_default_thresholds_are_spec_defaults() -> None:
    """Defaults: tau_residual = 0.20, max_iterations = 3 per §6.4."""
    assert DEFAULT_TAU_RESIDUAL == 0.20
    assert DEFAULT_MAX_ITERATIONS == 3


def test_loop_rejects_invalid_tau() -> None:
    proposer = _StubReProposer(proposals=[])
    extractor = _StubExtractor(per_section={})
    with pytest.raises(ValueError, match="tau_residual"):
        ResidualEMLoop(
            re_proposer=proposer,
            extractor=extractor,
            binder=LexicalClaimBinder(),
            tau_residual=-0.1,
        )
    with pytest.raises(ValueError, match="tau_residual"):
        ResidualEMLoop(
            re_proposer=proposer,
            extractor=extractor,
            binder=LexicalClaimBinder(),
            tau_residual=1.5,
        )


def test_loop_rejects_invalid_max_iterations() -> None:
    proposer = _StubReProposer(proposals=[])
    extractor = _StubExtractor(per_section={})
    with pytest.raises(ValueError, match="max_iterations"):
        ResidualEMLoop(
            re_proposer=proposer,
            extractor=extractor,
            binder=LexicalClaimBinder(),
            max_iterations=0,
        )


def test_outcome_dataclass_is_frozen() -> None:
    """EMOutcome is an immutable record so callers can pass it around safely."""
    outcome = EMOutcome(
        final_proposal=_empty_proposal(),
        final_claims=[],
        final_residual_rate=0.0,
        iterations=0,
        converged=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.iterations = 99  # type: ignore[misc]


def test_protocols_are_runtime_checkable() -> None:
    """Defensive: the test stubs satisfy the Protocols structurally."""

    class _Bin:
        def binds(self, *, claim: ClaimTuple, proposal: SchemaProposal) -> bool:
            return True

    assert isinstance(_Bin(), ClaimBinder)

    class _RP:
        def re_propose(
            self,
            *,
            previous: SchemaProposal,
            residual_claims: list[ClaimTuple],
        ) -> SchemaProposal:
            return _empty_proposal()

    assert isinstance(_RP(), SchemaReProposer)

    class _SE:
        def extract_sections(self, *, section_ids: list[str]) -> list[BoundClaim]:
            return []

    assert isinstance(_SE(), SectionExtractor)
