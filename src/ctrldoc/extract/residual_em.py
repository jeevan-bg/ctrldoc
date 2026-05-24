"""L0 residual EM loop — step 4 of the §6.4 schema co-induction algorithm.

`SchemaProposer` (S-132) emits one initial schema from a max-entropy
sample. That alone is rarely enough: documents have long-tail concepts
the sample misses, and the proposer's coverage degrades on heterogeneous
prose. The EM loop closes the gap.

Given the universal `ClaimTuple` floor produced by the Tier-1/Tier-2
extractors, the loop:

1. Measures `unmatched_claim_rate` — the fraction of universal tuples a
   `ClaimBinder` cannot bind to any typed slot in the current schema.
2. If the rate exceeds `tau_residual` (default 0.20 from §6.4), hands
   the residual claims to a `SchemaReProposer`, which performs one
   batched LLM call to refine the schema.
3. Re-runs the extractor scoped to the sections whose claims failed to
   bind — never the whole doc, in line with §6.4's "re-extract only
   over affected sections" rule.
4. Iterates until either the residual rate drops to / below
   `tau_residual` (converged), or `max_iterations` is exhausted (the
   universal tuple floor is the safety net per §13 non-negotiable 6).

The loop is intentionally agnostic to *how* binding is computed: the
default `LexicalClaimBinder` does token-overlap matching between a
claim's subject / predicate / object surface forms and the typed-node
names + typed-edge names in the schema. Production code can plug in an
NLI-backed binder or any other `ClaimBinder` implementation; the loop
contract — measure residual, scope re-extraction, terminate
deterministically — does not change.

SPEC-REF: §6.4 (schema co-induction — residual EM)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.extract.schema_proposer import SchemaProposal

DEFAULT_TAU_RESIDUAL: float = 0.20
"""§6.4 step 4 default — residual rate above which the loop re-proposes."""

DEFAULT_MAX_ITERATIONS: int = 3
"""§6.4 step 5 implied bound — converge in at most 3 EM passes."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundClaim:
    """A universal claim tuple tagged with the section it was extracted from.

    The `section_id` lets the EM loop scope re-extraction to the affected
    sections in step 4 of §6.4. The `claim` is the universal tuple from
    §6.2; downstream binders read only its surface fields, so any
    `ClaimTuple` shape with subject / predicate / object works.
    """

    section_id: str
    claim: ClaimTuple


@dataclass(frozen=True)
class EMOutcome:
    """Outcome of one `ResidualEMLoop.run` call.

    `converged` is true iff the residual rate dropped to / below
    `tau_residual` before `max_iterations` ran out. `final_proposal` is
    always the most recent schema the loop produced — even on
    non-convergence, the caller should persist it (the universal tuple
    floor is the safety net for binding failures).
    """

    final_proposal: SchemaProposal
    final_claims: list[BoundClaim]
    final_residual_rate: float
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ClaimBinder(Protocol):
    """Decides whether a single universal claim binds to the current schema.

    Implementations are pure functions of the claim and the schema —
    binding is stateless across calls so the loop's residual sweep can
    parallelise without coordination.
    """

    def binds(self, *, claim: ClaimTuple, proposal: SchemaProposal) -> bool: ...


@runtime_checkable
class SchemaReProposer(Protocol):
    """Refines an existing schema using the latest batch of residual claims.

    Implementations make one LLM call per `re_propose` invocation; the
    loop never calls this more than `max_iterations` times. The contract
    deliberately does not constrain the relationship between `previous`
    and the returned proposal — the LLM may keep, drop, or extend types
    freely; the loop's only invariant is that residual rate is
    re-measured against whatever schema is returned.
    """

    def re_propose(
        self,
        *,
        previous: SchemaProposal,
        residual_claims: list[ClaimTuple],
    ) -> SchemaProposal: ...


@runtime_checkable
class SectionExtractor(Protocol):
    """Re-runs the Tier-1+Tier-2 extractor over a specified subset of sections.

    `section_ids` is the dedup'd ordered set of sections that contained
    residual claims in the previous iteration. The return value carries
    the freshly extracted `BoundClaim` rows for those sections; the loop
    replaces — never appends to — the prior rows for those sections.
    """

    def extract_sections(self, *, section_ids: list[str]) -> list[BoundClaim]: ...


# ---------------------------------------------------------------------------
# Default binder
# ---------------------------------------------------------------------------


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
"""A maximal alphanumeric run, optionally hyphen-joined to peers."""


def _tokenize_lower(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN.finditer(text)}


class LexicalClaimBinder:
    """Token-overlap binder — the cheapest defensible default.

    A claim binds when either:

    * a typed-node name token appears in the claim's subject or object
      surface form (after lower-casing), OR
    * a typed-edge name token appears in the claim's predicate surface
      form (after lower-casing).

    Token equality is whole-token only (we deliberately reject
    substring matches: `widgets` must NOT bind to `Widget` because the
    plural form usually denotes a different concept under §6.3 Galois
    subsumption). The binder is stateless and safe to share across
    threads.
    """

    def binds(self, *, claim: ClaimTuple, proposal: SchemaProposal) -> bool:
        subject_tokens = _tokenize_lower(claim.subject)
        object_tokens = _tokenize_lower(claim.object)
        predicate_tokens = _tokenize_lower(claim.predicate)
        for node in proposal.nodes:
            node_tokens = _tokenize_lower(node.name)
            if node_tokens and (node_tokens & subject_tokens or node_tokens & object_tokens):
                return True
        for edge in proposal.edges:
            edge_tokens = _tokenize_lower(edge.name)
            if edge_tokens and edge_tokens & predicate_tokens:
                return True
        return False


# ---------------------------------------------------------------------------
# Residual rate
# ---------------------------------------------------------------------------


def residual_rate(
    *,
    bound_claims: Sequence[BoundClaim],
    proposal: SchemaProposal,
    binder: ClaimBinder,
) -> float:
    """Fraction of `bound_claims` that did not bind to `proposal`.

    An empty input is vacuously satisfied (returns 0.0) — the binder
    has nothing to score, so the loop must not interpret it as a
    failure case. This is the only place the loop converts the binder's
    per-claim verdict into a scalar; all callers should use this helper
    instead of re-implementing the count.
    """
    if not bound_claims:
        return 0.0
    unbound = sum(1 for bc in bound_claims if not binder.binds(claim=bc.claim, proposal=proposal))
    return unbound / len(bound_claims)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class ResidualEMLoop:
    """EM-style convergence wrapper around the schema proposer + re-extractor.

    Construction validates the two scalar knobs (`tau_residual` ∈ [0,1]
    and `max_iterations` ≥ 1) so a malformed configuration fails fast
    instead of producing a non-deterministic run.

    The loop is single-shot per `run`: one `run` call corresponds to one
    document's schema co-induction. The instance carries no per-doc
    state, so the same loop object can drive many documents serially.
    """

    def __init__(
        self,
        *,
        re_proposer: SchemaReProposer,
        extractor: SectionExtractor,
        binder: ClaimBinder,
        tau_residual: float = DEFAULT_TAU_RESIDUAL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if not 0.0 <= tau_residual <= 1.0:
            raise ValueError(f"tau_residual must be in [0.0, 1.0]; got {tau_residual}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1; got {max_iterations}")
        self._re_proposer = re_proposer
        self._extractor = extractor
        self._binder = binder
        self._tau_residual = tau_residual
        self._max_iterations = max_iterations

    def run(
        self,
        *,
        initial_proposal: SchemaProposal,
        initial_claims: Sequence[BoundClaim],
    ) -> EMOutcome:
        """Iterate the EM loop until convergence or `max_iterations`.

        The bound-claims list is mutated copy-on-replace: each iteration
        rebuilds the per-section view from the previous iteration's
        claims and the fresh re-extraction over the affected sections.
        This guarantees that a claim bound in iteration 0 is preserved
        through the loop even if its section is not re-extracted later.
        """
        proposal = initial_proposal
        claims: list[BoundClaim] = list(initial_claims)
        rate = residual_rate(bound_claims=claims, proposal=proposal, binder=self._binder)
        if rate <= self._tau_residual:
            return EMOutcome(
                final_proposal=proposal,
                final_claims=claims,
                final_residual_rate=rate,
                iterations=0,
                converged=True,
            )

        iterations = 0
        while iterations < self._max_iterations:
            residual_sections, residual_claim_tuples = _split_residuals(
                claims=claims, proposal=proposal, binder=self._binder
            )
            proposal = self._re_proposer.re_propose(
                previous=proposal,
                residual_claims=residual_claim_tuples,
            )
            fresh = self._extractor.extract_sections(section_ids=residual_sections)
            claims = _merge_re_extraction(
                prior=claims, fresh=fresh, replaced_sections=set(residual_sections)
            )
            iterations += 1
            rate = residual_rate(bound_claims=claims, proposal=proposal, binder=self._binder)
            if rate <= self._tau_residual:
                return EMOutcome(
                    final_proposal=proposal,
                    final_claims=claims,
                    final_residual_rate=rate,
                    iterations=iterations,
                    converged=True,
                )

        return EMOutcome(
            final_proposal=proposal,
            final_claims=claims,
            final_residual_rate=rate,
            iterations=iterations,
            converged=False,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_residuals(
    *,
    claims: Sequence[BoundClaim],
    proposal: SchemaProposal,
    binder: ClaimBinder,
) -> tuple[list[str], list[ClaimTuple]]:
    """Pull the unbound claims and the dedup'd ordered set of their sections."""
    residual_section_ids: list[str] = []
    seen_sections: set[str] = set()
    residual_tuples: list[ClaimTuple] = []
    for bc in claims:
        if binder.binds(claim=bc.claim, proposal=proposal):
            continue
        residual_tuples.append(bc.claim)
        if bc.section_id not in seen_sections:
            seen_sections.add(bc.section_id)
            residual_section_ids.append(bc.section_id)
    return residual_section_ids, residual_tuples


def _merge_re_extraction(
    *,
    prior: Sequence[BoundClaim],
    fresh: Sequence[BoundClaim],
    replaced_sections: set[str],
) -> list[BoundClaim]:
    """Keep prior claims for untouched sections; replace claims for re-extracted ones."""
    out: list[BoundClaim] = [bc for bc in prior if bc.section_id not in replaced_sections]
    out.extend(fresh)
    return out


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TAU_RESIDUAL",
    "BoundClaim",
    "ClaimBinder",
    "EMOutcome",
    "LexicalClaimBinder",
    "ResidualEMLoop",
    "SchemaReProposer",
    "SectionExtractor",
    "residual_rate",
]
