"""L5 `merge` operation via union-find + Galois join (§6.6).

§6.6 frames `merge({A, B, C})` as:

> *Centroid distribution over the union; per-cluster strongest claim
> (Galois join) emitted; topological order via `depends_on` +
> `part_of`. **Loss invariant: every input claim ID maps to exactly
> one output cluster.***

The hard `loss_invariant_satisfied` gate is the §6.6 release
contract. This module ships the §6.6 reduction:

* Build the upper-triangular pairwise judgement matrix over input
  claims. For each ordered pair `(i < j)`:
    1. Galois floor first via `claim_subsumption(i, j)`. If
       `equivalent`, `subsumes`, or `subsumed_by` → record as
       *mergeable* (the two claims live on the same lattice
       component) without any NLI call.
    2. Otherwise consult the NLI scorer in both directions. If both
       `entailment(i → j)` and `entailment(j → i)` clear the
       `equivalence_threshold` → the pair is a paraphrase-style
       semantic equivalence → record as mergeable.
* Union-find collapses every mergeable pair into one cluster.
  Singletons (no merge partner) emit as their own cluster — that's
  what guarantees the §6.6 loss invariant by construction: every
  input id has a unique root, and roots are partitioned.
* Per cluster, pick the *strongest* representative by reducing
  `claim_meet` (Galois greatest lower bound — the strongest claim
  implying every member) across members in input order. On any
  incomparable step, fall back to the running accumulator (the
  meet is undefined for cross-axis modalities); the first
  successful reduction wins, ties tiebreak on input order.

Cost contract: NLI is asked exactly twice per `(i, j)` pair the
Galois floor returned `incomparable` for. Galois-resolvable pairs
cost zero NLI calls — modality / qualifier chains short-circuit
the fallback entirely. Worst-case upper bound is `n * (n - 1)` calls
when every pair survives the floor, but in practice the floor
absorbs most ordering cases.

SPEC-REF: §6.6 (merge = partition + Galois join with loss invariant)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.merge import (
    MERGE_PARTITION_THRESHOLD,
    InputClaim,
    MergedCluster,
    MergeOutput,
)
from ctrldoc.extract.galois import claim_meet, claim_subsumption

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


MERGE_EQUIVALENCE_THRESHOLD: float = 0.70
"""Default minimum bidirectional entailment for the NLI fallback to declare
a pair semantically equivalent (paraphrase-mergeable).

Picked at the same level as the cross-doc edge inferer's hard
threshold (§6.7) so the two §6 surfaces stay consistent — a pair
strong enough to land an `entails_across` edge is strong enough to
collapse into one merge cluster when seen from both directions.
"""


MERGE_PARTITION_ACCURACY_THRESHOLD: float = MERGE_PARTITION_THRESHOLD
"""Soft pairwise-accuracy release gate (§6.6); re-exports
`ctrldoc.eval.merge.MERGE_PARTITION_THRESHOLD = 0.85` so callers do
not need to depend on the eval substrate for the constant alone."""


# ---------------------------------------------------------------------------
# Scorer protocol — reused from coverage / compare
# ---------------------------------------------------------------------------


@runtime_checkable
class NLIScorer(Protocol):
    """3-way NLI backend. Same shape as `coverage.NLIScorer` / `compare.NLIScorer`."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeConfig:
    """Tunable knobs for the union-find+Galois-join merge operation."""

    equivalence_threshold: float = MERGE_EQUIVALENCE_THRESHOLD
    """Minimum bidirectional NLI entailment for two Galois-incomparable
    claims to be deemed mergeable (paraphrase-equivalent).

    Must lie strictly inside the unit interval `(0, 1)`. At `0.0` every
    pair would merge (no discrimination); at `1.0` no real-world NLI
    score is ever enough — both boundaries are rejected at
    construction time.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.equivalence_threshold < 1.0:
            raise ValueError(
                "equivalence_threshold must be in the open interval (0, 1) "
                f"(got {self.equivalence_threshold})"
            )


class MergeResult(BaseModel):
    """Aggregate output of one `merge` call.

    `output.clusters` aligns by cluster index with §6.6's partition
    contract; `scorer_calls` is the bookkeeping count the §6.6 cost
    contract is asserted against in tests.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output: MergeOutput
    scorer_calls: int


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def merge(
    *,
    input_claims: Sequence[InputClaim],
    scorer: NLIScorer,
    config: MergeConfig | None = None,
) -> MergeResult:
    """Partition + Galois-join merge per §6.6.

    Returns a `MergeOutput` whose clusters satisfy the §6.6 loss
    invariant (`loss_invariant_satisfied(input_ids, clusters) == True`
    by construction). Empty input short-circuits to an empty output;
    single-input short-circuits to one singleton cluster — both
    paths cost zero NLI calls.
    """
    cfg = config or MergeConfig()
    inputs = list(input_claims)

    if not inputs:
        return MergeResult(output=MergeOutput.model_construct(clusters=[]), scorer_calls=0)

    n = len(inputs)
    parent = list(range(n))

    def _find(x: int) -> int:
        # Path-compression union-find.
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        # Tiebreak union direction on input index so the per-cluster
        # member order is deterministic — smaller root absorbs.
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Pre-render claim text once per input — used by the NLI fallback.
    rendered = [_render_claim(ic.claim) for ic in inputs]

    scorer_calls = 0
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = inputs[i].claim, inputs[j].claim

            # Galois floor — handles same-SVO modality / qualifier chains
            # at zero NLI cost.
            verdict = claim_subsumption(ci, cj)
            if verdict in ("equivalent", "subsumes", "subsumed_by"):
                _union(i, j)
                continue

            # NLI fallback — semantic equivalence requires both directions.
            e_ij = scorer.score(premise=rendered[i], hypothesis=rendered[j]).entailment
            e_ji = scorer.score(premise=rendered[j], hypothesis=rendered[i]).entailment
            scorer_calls += 2
            if e_ij >= cfg.equivalence_threshold and e_ji >= cfg.equivalence_threshold:
                _union(i, j)

    # Roll up parent map into clusters by root, preserving input order.
    by_root: dict[int, list[int]] = {}
    for idx in range(n):
        root = _find(idx)
        by_root.setdefault(root, []).append(idx)

    # Emit clusters sorted by their smallest member index → deterministic
    # cluster order across runs and across pytest sessions.
    clusters: list[MergedCluster] = []
    for cluster_idx, root in enumerate(sorted(by_root.keys())):
        members = by_root[root]
        member_ids = [inputs[m].id for m in members]
        strongest = _galois_join_representative([inputs[m].claim for m in members])
        clusters.append(
            MergedCluster(
                id=f"cluster-{cluster_idx + 1}",
                member_claim_ids=member_ids,
                strongest_claim=strongest,
            )
        )

    return MergeResult(output=MergeOutput(clusters=clusters), scorer_calls=scorer_calls)


class TransportMerger:
    """`Merger`-shaped consumer of the §6.6 reduction.

    The §14 eval substrate's `MergeEvalRunner` consumes any object
    satisfying the `Merger` protocol; this class adapts the
    functional `merge` surface onto that protocol so the same
    reduction is graded directly by the existing eval fixture
    without writing a second adapter.
    """

    def __init__(
        self,
        *,
        scorer: NLIScorer,
        config: MergeConfig | None = None,
    ) -> None:
        self._scorer = scorer
        self._config = config or MergeConfig()

    def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput:
        result = merge(input_claims=input_claims, scorer=self._scorer, config=self._config)
        return result.output


# ---------------------------------------------------------------------------
# Internal — Galois-join representative selection
# ---------------------------------------------------------------------------


def _galois_join_representative(members: list[ClaimTuple]) -> ClaimTuple:
    """Reduce `claim_meet` across cluster members; pick the strongest.

    The Galois meet is the greatest lower bound — the strongest claim
    that implies both operands. For modality-driven clusters
    (e.g. `MUST`, `SHOULD`, `MAY`) the meet surfaces the most-binding
    rung. For cross-axis or paraphrase clusters the floor returns
    `None`; we fall back to the running accumulator, preserving the
    first member's claim as the deterministic default — input order
    is the tiebreak the eval substrate already keys on.

    Pre: `members` is non-empty (guaranteed by the union-find caller).
    """
    accumulator = members[0]
    for next_member in members[1:]:
        meet = claim_meet(accumulator, next_member)
        if meet is not None:
            accumulator = meet
    return accumulator


# ---------------------------------------------------------------------------
# Claim rendering — kept consistent with ops.coverage / ops.compare
# ---------------------------------------------------------------------------


def _render_claim(claim: ClaimTuple) -> str:
    """Render a `ClaimTuple` as the natural-language surface for the NLI scorer.

    Mirrors `ctrldoc.ops.coverage._render_claim` /
    `ctrldoc.ops.compare._render_claim` so the same NLI backend sees
    the same surface regardless of which op surfaces it.
    """
    subject = claim.subject.strip()
    predicate = claim.predicate.strip()
    obj = claim.object.strip()
    qualifier = claim.qualifier.strip()

    if claim.polarity == "negative":
        predicate = f"does not {predicate}"

    parts = [p for p in (subject, predicate, obj, qualifier) if p]
    return " ".join(parts)


__all__ = [
    "MERGE_EQUIVALENCE_THRESHOLD",
    "MERGE_PARTITION_ACCURACY_THRESHOLD",
    "MergeConfig",
    "MergeResult",
    "NLIScorer",
    "TransportMerger",
    "merge",
]
