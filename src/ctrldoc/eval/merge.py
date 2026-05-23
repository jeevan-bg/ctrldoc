"""merge_eval — partition + Galois-join verdict scoring.

The eval set grades a `Merger` on its ability to merge claims from
N input docs into a partition over the shared concept lattice. From
SPEC §6.6:

> `merge({A, B, C})` | Centroid distribution over the union;
> per-cluster strongest claim (Galois join) emitted; topological
> order via `depends_on` + `part_of`. **Loss invariant: every input
> claim ID maps to exactly one output cluster.**

The substrate ships three orthogonal metrics:

1. `loss_invariant_satisfied` — a hard binary gate from §6.6: the
   predicted clusters must cover every input claim id exactly once,
   with no extras. A merger that fails this is malformed.
2. `pairwise_accuracy` — the soft partition metric: across every
   pair of input claim ids, do predicted and gold agree on the
   "same cluster or not" question? This is symmetric, alignment-free,
   and naturally handles many-to-one merges.
3. `representative_match_rate` — for each predicted cluster, the
   best-overlap gold cluster's strongest_claim is compared via the
   §6.2 universal-tuple core match. This pins the Galois-join
   contract — a merger that picks the wrong representative shows up
   even when its partition is perfect.

The runner gates pass/fail on the loss invariant AND pairwise
accuracy ≥ 0.85. Representative match rate is reported but not
gated — the substrate locks the partition shape contract; the
representative choice is graded once the transport engine lands.

SPEC-REF: §6.6 (merge = partition + Galois join with loss invariant), §14
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ctrldoc.eval.claim_extraction import ClaimTuple, DocTypeLiteral, claim_tuple_matches
from ctrldoc.eval.harness import EvalResult

MERGE_PARTITION_THRESHOLD = 0.85


class InputClaim(BaseModel):
    """One claim from one input doc.

    `id` is unique across the case (not just within its doc) — the
    merge contract is keyed on global claim ids.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    doc_id: str
    doc_type: DocTypeLiteral
    claim: ClaimTuple


class MergedCluster(BaseModel):
    """One output cluster: members + the Galois-join strongest claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    member_claim_ids: list[str] = Field(min_length=1)
    strongest_claim: ClaimTuple

    @field_validator("member_claim_ids")
    @classmethod
    def _members_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("member_claim_ids must be unique within a cluster")
        return v


class MergeOutput(BaseModel):
    """A merger's full output for one case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clusters: list[MergedCluster] = Field(min_length=1)

    @field_validator("clusters")
    @classmethod
    def _cluster_ids_unique(cls, v: list[MergedCluster]) -> list[MergedCluster]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster ids must be unique within a MergeOutput")
        return v


class MergeEvalCase(BaseModel):
    """One merge case: N input docs' claims + gold clustering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    input_claims: list[InputClaim] = Field(min_length=1)
    gold_clusters: list[MergedCluster] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("input_claims")
    @classmethod
    def _input_ids_unique(cls, v: list[InputClaim]) -> list[InputClaim]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("input_claims must have unique ids within a case")
        return v

    @field_validator("gold_clusters")
    @classmethod
    def _gold_cluster_ids_unique(cls, v: list[MergedCluster]) -> list[MergedCluster]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("gold_clusters must have unique cluster ids")
        return v

    @model_validator(mode="after")
    def _gold_satisfies_loss_invariant(self) -> MergeEvalCase:
        input_ids = [c.id for c in self.input_claims]
        result = loss_invariant_satisfied(input_ids=input_ids, output_clusters=self.gold_clusters)
        if not result:
            raise ValueError(
                f"case {self.id!r}: gold_clusters violate the §6.6 loss invariant "
                "(every input claim id must appear in exactly one output cluster)"
            )
        return self


def loss_invariant_satisfied(*, input_ids: list[str], output_clusters: list[MergedCluster]) -> bool:
    """§6.6: every input claim id appears in exactly one output cluster.

    Returns False on any of: duplicate assignment (same input id in
    two clusters), missing assignment (input id absent from output),
    or extra assignment (output names an id not in input).
    """
    assigned: dict[str, int] = {}
    for ci, cluster in enumerate(output_clusters):
        for cid in cluster.member_claim_ids:
            if cid in assigned:
                return False
            assigned[cid] = ci
    return set(assigned.keys()) == set(input_ids)


def pairwise_partition_accuracy(
    *,
    predicted: list[MergedCluster],
    gold: list[MergedCluster],
    input_ids: list[str],
) -> float:
    """Across every unordered pair of input ids, fraction where
    predicted and gold agree on "same cluster or not".

    Uses the (input_id → cluster_index) map induced by each
    clustering. Pairs where an input id is missing from either map
    are counted as disagreement — this propagates partition-shape
    bugs into the metric rather than silently dropping them.

    With fewer than two input ids there are no pairs; returns 1.0
    (degenerate but well-defined).
    """
    if len(input_ids) < 2:
        return 1.0

    def _assignment(clusters: list[MergedCluster]) -> dict[str, int]:
        m: dict[str, int] = {}
        for ci, cluster in enumerate(clusters):
            for cid in cluster.member_claim_ids:
                m[cid] = ci
        return m

    pred_map = _assignment(predicted)
    gold_map = _assignment(gold)

    total = 0
    agree = 0
    n = len(input_ids)
    for i in range(n):
        a = input_ids[i]
        for j in range(i + 1, n):
            b = input_ids[j]
            total += 1
            if a not in pred_map or b not in pred_map or a not in gold_map or b not in gold_map:
                continue
            pred_same = pred_map[a] == pred_map[b]
            gold_same = gold_map[a] == gold_map[b]
            if pred_same == gold_same:
                agree += 1
    return agree / total if total else 1.0


def representative_match_rate(
    *,
    predicted: list[MergedCluster],
    gold: list[MergedCluster],
) -> float:
    """Per predicted cluster, find the gold cluster with maximum
    member overlap and check that the representative claims match
    via §6.2 core match.

    The match is many-to-one: two predicted clusters may best-overlap
    the same gold cluster (e.g. when the merger over-splits). Each
    predicted cluster is scored independently — a predicted cluster
    with no overlap to any gold cluster scores 0 for that slot.

    Returns the fraction of predicted clusters whose representative
    matches; 1.0 when there are no predicted clusters (degenerate).
    """
    if not predicted:
        return 1.0
    correct = 0
    for p in predicted:
        best_overlap = 0
        best_gold: MergedCluster | None = None
        p_members = set(p.member_claim_ids)
        for g in gold:
            overlap = len(p_members & set(g.member_claim_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_gold = g
        if best_gold is None:
            continue
        if claim_tuple_matches(extracted=p.strongest_claim, gold=best_gold.strongest_claim):
            correct += 1
    return correct / len(predicted)


@runtime_checkable
class Merger(Protocol):
    """Merger under evaluation."""

    def merge(self, *, input_claims: list[InputClaim]) -> MergeOutput: ...


class MergeEvalRunner:
    """Adapt a `Merger` into the harness `CaseRunner` shape.

    The runner gates pass/fail on the loss invariant AND pairwise
    accuracy ≥ 0.85. A merger that violates the loss invariant
    cannot pass regardless of partition shape — the §6.6 contract
    is binary on that axis.
    """

    def __init__(self, *, merger: Merger) -> None:
        self._merger = merger

    def run_case(self, case: MergeEvalCase) -> EvalResult:
        output = self._merger.merge(input_claims=list(case.input_claims))
        input_ids = [c.id for c in case.input_claims]
        invariant = loss_invariant_satisfied(input_ids=input_ids, output_clusters=output.clusters)
        pairwise = pairwise_partition_accuracy(
            predicted=output.clusters,
            gold=list(case.gold_clusters),
            input_ids=input_ids,
        )
        rep_rate = representative_match_rate(
            predicted=output.clusters, gold=list(case.gold_clusters)
        )
        passed = invariant and pairwise >= MERGE_PARTITION_THRESHOLD
        return EvalResult(
            case_id=case.id,
            passed=passed,
            score=pairwise if invariant else 0.0,
            metrics={
                "loss_invariant_satisfied": 1.0 if invariant else 0.0,
                "pairwise_accuracy": pairwise,
                "representative_match_rate": rep_rate,
            },
            notes=(
                f"inputs={len(input_ids)}, gold_clusters={len(case.gold_clusters)}, "
                f"predicted_clusters={len(output.clusters)}, "
                f"invariant={invariant}, pairwise={pairwise:.3f}, rep={rep_rate:.3f}"
            ),
        )


__all__ = [
    "MERGE_PARTITION_THRESHOLD",
    "InputClaim",
    "MergeEvalCase",
    "MergeEvalRunner",
    "MergeOutput",
    "MergedCluster",
    "Merger",
    "loss_invariant_satisfied",
    "pairwise_partition_accuracy",
    "representative_match_rate",
]
