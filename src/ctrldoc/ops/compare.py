"""L5 `compare` operation via the Galois floor + asymmetric NLI transport (§6.6).

§6.6 frames `compare(A, B)` as:

> *Asymmetric transport in both directions; per-concept-cluster cost
> summary = strengths/weaknesses.*

The eval substrate (`ctrldoc.eval.compare`) pre-clusters claims into
`{a_claim?, b_claim?}` cells and grades the per-cluster verdict over
the 3-label space `{StrengthA, StrengthB, Gap}`. This module ships
the §6.6 reduction:

* Exactly one side present → `Gap`. The other side has nothing to
  compare against; §6.6 reserves `Gap` for the structural "only one
  doc speaks to this concept" case.
* Both sides present → consult `claim_subsumption` first (§6.3
  Galois floor). The floor handles the deterministic modality /
  qualifier cases (`MUST ⊐ SHOULD ⊐ MAY`, empty qualifier ⊐ scoped)
  at zero NLI cost.
  - `subsumes` (A strictly stronger) → `StrengthA`.
  - `subsumed_by` (A strictly weaker) → `StrengthB`.
  - `equivalent` (Galois ties) → deterministic tiebreak `StrengthA`.
  - `incomparable` → fall back to asymmetric NLI ("transport in
    both directions"). Compute `e_AB = NLI_entail(A → B)` and
    `e_BA = NLI_entail(B → A)`. The side that's entailed *more
    strongly by the other* is the consequence — the other is the
    more general (stronger) claim. `StrengthA` when `e_BA > e_AB`;
    `StrengthB` when `e_AB > e_BA`. Ties below `nli_tie_gap` (the
    symmetric-paraphrase case where neither side is structurally
    stronger) tiebreak deterministically to `StrengthA`.

Cost contract: zero NLI calls for `Gap` or Galois-resolved clusters;
exactly two NLI calls per cluster that escalates to the fallback —
asymmetric means each direction is scored once. Total upper bound is
`2 * |fallback-clusters|`, never more.

SPEC-REF: §6.6 (compare = per-concept-cluster strengths/weaknesses
via asymmetric transport in both directions)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.compare import (
    COMPARE_VERDICT_THRESHOLD,
    CompareVerdictLiteral,
    ConceptComparisonInput,
)
from ctrldoc.extract.galois import claim_subsumption

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


COMPARE_NLI_TIE_GAP: float = 0.05
"""Default minimum absolute gap between `e_BA` and `e_AB` for an NLI
verdict to count as decisive.

The §6.6 framing has no asymmetry tighter than this without losing
discrimination from real NLI backends — typical 3-way softmax noise
sits comfortably under five percentage points. Pairs whose
directional entailments fall within the gap are deemed symmetric and
tiebreak deterministically to `StrengthA`.
"""


COMPARE_VERDICT_ACCURACY_THRESHOLD: float = COMPARE_VERDICT_THRESHOLD
"""Per-cluster accuracy release gate (§6.6); re-exports
`ctrldoc.eval.compare.COMPARE_VERDICT_THRESHOLD = 0.85` so callers do
not need to depend on the eval substrate for the constant alone."""


# ---------------------------------------------------------------------------
# Scorer protocol — reused from coverage / cross_doc_edges
# ---------------------------------------------------------------------------


@runtime_checkable
class NLIScorer(Protocol):
    """3-way NLI backend. Same shape as `coverage.NLIScorer`."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompareConfig:
    """Tunable knobs for the transport+Galois-based compare operation."""

    nli_tie_gap: float = COMPARE_NLI_TIE_GAP
    """Minimum absolute difference `|e_BA - e_AB|` for an NLI fallback
    verdict to count as decisive. Gaps below this threshold tiebreak
    to `StrengthA` (alphabetical, deterministic).

    Must lie strictly inside the unit interval `(0, 1)`. At `0.0` the
    tiebreak never fires (every fallback is decisive); at `1.0` no
    real-world NLI gap is ever decisive — both boundaries are
    rejected at construction time.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.nli_tie_gap < 1.0:
            raise ValueError(
                f"nli_tie_gap must be in the open interval (0, 1) (got {self.nli_tie_gap})"
            )


class CompareResult(BaseModel):
    """Aggregate output of one `compare` call.

    `verdicts` aligns by position with the cluster list the caller
    passed. `scorer_calls` is the bookkeeping count the §6.6 cost
    contract is asserted against in tests — equals `2 * |fallback
    clusters|`, never more.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdicts: list[CompareVerdictLiteral]
    scorer_calls: int


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compare(
    *,
    clusters: Sequence[ConceptComparisonInput],
    scorer: NLIScorer,
    config: CompareConfig | None = None,
) -> CompareResult:
    """Per-cluster compare verdicts via the §6.6 Galois+transport reduction.

    Returns one verdict per cluster in input order. Empty cluster list
    short-circuits to an empty result with zero scorer calls.
    """
    cfg = config or CompareConfig()
    cluster_list = list(clusters)

    if not cluster_list:
        return CompareResult(verdicts=[], scorer_calls=0)

    verdicts: list[CompareVerdictLiteral] = []
    scorer_calls = 0
    for cluster in cluster_list:
        verdict, calls = _verdict_for_cluster(cluster, scorer, cfg)
        verdicts.append(verdict)
        scorer_calls += calls

    return CompareResult(verdicts=verdicts, scorer_calls=scorer_calls)


class TransportCompareVerifier:
    """`CompareVerifier`-shaped consumer of the §6.6 reduction.

    The §14 eval substrate's `CompareEvalRunner` consumes any object
    satisfying the `CompareVerifier` protocol; this class adapts the
    functional `compare` surface onto that protocol so the same
    reduction is graded directly by the existing eval fixture
    without writing a second adapter.
    """

    def __init__(
        self,
        *,
        scorer: NLIScorer,
        config: CompareConfig | None = None,
    ) -> None:
        self._scorer = scorer
        self._config = config or CompareConfig()

    def verdicts(
        self,
        *,
        clusters: list[ConceptComparisonInput],
    ) -> list[CompareVerdictLiteral]:
        result = compare(clusters=clusters, scorer=self._scorer, config=self._config)
        return result.verdicts


# ---------------------------------------------------------------------------
# Internal — per-cluster verdict
# ---------------------------------------------------------------------------


def _verdict_for_cluster(
    cluster: ConceptComparisonInput,
    scorer: NLIScorer,
    config: CompareConfig,
) -> tuple[CompareVerdictLiteral, int]:
    """Decide the verdict for one cluster; return (verdict, scorer_calls)."""
    # Structural Gap — exactly one side present.
    if cluster.a_claim is None or cluster.b_claim is None:
        return ("Gap", 0)

    a = cluster.a_claim
    b = cluster.b_claim

    # Galois floor first — handles modality / qualifier ordering at zero NLI cost.
    verdict = claim_subsumption(a, b)
    if verdict == "subsumes":
        return ("StrengthA", 0)
    if verdict == "subsumed_by":
        return ("StrengthB", 0)
    if verdict == "equivalent":
        # Deterministic tiebreak — gold never assigns `equivalent`, and
        # alphabetical pins the choice without surprise.
        return ("StrengthA", 0)

    # Galois `incomparable` → asymmetric NLI fallback. Two calls per cluster.
    a_text = _render_claim(a)
    b_text = _render_claim(b)
    e_ab = scorer.score(premise=a_text, hypothesis=b_text).entailment
    e_ba = scorer.score(premise=b_text, hypothesis=a_text).entailment

    # The side that *entails* the other is the stronger / more general
    # claim — entailment is the §6.3 subsumption relation. `e_AB > e_BA`
    # ⇒ A entails B strongly while B does not entail A ⇒ A is the
    # more general claim ⇒ `StrengthA`. Symmetric on the other side.
    if e_ab - e_ba > config.nli_tie_gap:
        return ("StrengthA", 2)
    if e_ba - e_ab > config.nli_tie_gap:
        return ("StrengthB", 2)
    # Symmetric (paraphrase or undecided) — deterministic tiebreak.
    return ("StrengthA", 2)


# ---------------------------------------------------------------------------
# Claim rendering — kept consistent with ops.coverage._render_claim
# ---------------------------------------------------------------------------


def _render_claim(claim: ClaimTuple) -> str:
    """Render a `ClaimTuple` as the natural-language surface for the NLI scorer.

    Mirrors `ctrldoc.ops.coverage._render_claim` so the same NLI
    backend sees the same surface regardless of which op surfaces it.
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
    "COMPARE_NLI_TIE_GAP",
    "COMPARE_VERDICT_ACCURACY_THRESHOLD",
    "CompareConfig",
    "CompareResult",
    "NLIScorer",
    "TransportCompareVerifier",
    "compare",
]
