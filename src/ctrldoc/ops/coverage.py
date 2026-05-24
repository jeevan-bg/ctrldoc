"""L5 `coverage` + `list_check` via the optimal-transport core (§6.6).

§6.6 collapses both operations to a single primitive:

> *coverage(A → B): min-cost transport of B's claim-mass onto A's
> claim-mass, cost = `1 - NLI_entail(A, B)`. Unmoved mass = uncovered.*
>
> *list_check(items, D): list parsed as a tiny doc; coverage(items → D).*

The reduction is mechanical. For each target claim `t`, we ask an
`NLIScorer` for the entailment confidence `e_ij = NLI_entail(s_i → t_j)`
of every source claim `s_i`. The transport problem is built with:

* A real source row per source claim, weight `1.0`.
* A **slack source** appended at the end, weight `|targets|`. The slack
  source represents "no real source supports this target"; mass routed
  to the slack source = `Missing`.
* A target column per target claim, weight `1.0`.
* Real-cost cells: `cost[i][j] = 1 - e_ij`. Slack-cost cells:
  `slack_cost = 1 - entailment_threshold` (default `0.5`, so any real
  source with entailment confidence > 0.5 strictly beats slack).

The total source mass `|sources| + |targets|` equals the target mass
`|targets|` only when `|sources| = 0`; otherwise we pad the target side
with an "absorption" column of weight `|sources|` and zero cost to keep
the problem balanced. (The min-cost engine requires balanced marginals;
the absorption column is an implementation detail callers never see —
verdict reading slices it off.)

After solving via `min_cost_transport`, each target claim's verdict is
`Covered` iff the majority of its incoming mass comes from a real
source; `Missing` iff the majority comes from the slack source.
Contradiction-dominated NLI scores naturally cluster on the
`Missing` side because their entailment confidence is necessarily low
(the three-way softmax sums to 1, so high contradiction → low
entailment → high cost → routed to slack).

Cost contract: `|sources| * |targets|` scorer calls per `coverage`
call. Linear in the product, never worse — the transport reduction
itself is dwarfed by the NLI step.

`list_check(items, D)` is `coverage(items, D)` with the items list
treated as a tiny target. The data-flow direction is identical to
`coverage(D → items)` because §6.6 frames `list_check` as
`coverage(items → D)`, where the items are *targets* graded against
the doc *as source*. The function signature uses the same names
(`items` and `doc`) the §9 CLI exposes so the surface is readable
end-to-end.

SPEC-REF: §6.6 (optimal-transport core — one algorithm, five queries)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.calibration import NLIScore
from ctrldoc.eval.claim_extraction import ClaimTuple
from ctrldoc.eval.cross_doc_coverage import (
    CROSS_DOC_COVERAGE_THRESHOLD,
    CoverageVerdictLiteral,
)
from ctrldoc.ops.transport import TransportProblem, min_cost_transport

# ---------------------------------------------------------------------------
# Public thresholds + defaults
# ---------------------------------------------------------------------------


COVERAGE_ENTAILMENT_THRESHOLD: float = 0.5
"""Minimum entailment confidence for a real source to beat the slack source.

A target whose best real-source entailment confidence is below this
threshold gets routed to the slack source by the transport engine and
emits `Missing`. The §6.6 framing has the threshold encoded as a cost
on the slack column: `slack_cost = 1 - threshold`.
"""


COVERAGE_VERDICT_ACCURACY_THRESHOLD: float = CROSS_DOC_COVERAGE_THRESHOLD
"""Per-claim accuracy release gate (§6.6); equal to the eval substrate's.

Re-exported here so callers do not need to depend on the eval substrate
for the release-gate constant alone. Equal by construction to
`ctrldoc.eval.cross_doc_coverage.CROSS_DOC_COVERAGE_THRESHOLD = 0.85`.
"""


# ---------------------------------------------------------------------------
# Scorer protocol — reused from the calibration substrate
# ---------------------------------------------------------------------------


@runtime_checkable
class NLIScorer(Protocol):
    """3-way NLI backend. Same shape as `CalibrationScorer` and `tier2_nli.NLIScorer`."""

    def score(self, *, premise: str, hypothesis: str) -> NLIScore: ...


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageConfig:
    """Tunable knobs for the transport-based coverage operation."""

    entailment_threshold: float = COVERAGE_ENTAILMENT_THRESHOLD
    """Minimum entailment confidence for a source to beat the slack column.

    Must lie strictly inside the unit interval `(0, 1)`. A threshold at
    the boundary makes the slack column either always-win (`1.0`) or
    always-lose (`0.0`); both degenerate cases are rejected at
    construction time.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.entailment_threshold < 1.0:
            raise ValueError(
                "entailment_threshold must be in the open interval (0, 1) "
                f"(got {self.entailment_threshold})"
            )


class CoverageResult(BaseModel):
    """Aggregate output of one `coverage` (or `list_check`) call.

    `verdicts` aligns by position with the target-claim list the caller
    passed. `scorer_calls` is the bookkeeping count the §6.6 cost
    contract is asserted against in tests; it equals
    `|sources| * |targets|` for a non-empty input.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdicts: list[CoverageVerdictLiteral]
    scorer_calls: int


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def coverage(
    *,
    source: Sequence[ClaimTuple],
    target: Sequence[ClaimTuple],
    scorer: NLIScorer,
    config: CoverageConfig | None = None,
) -> CoverageResult:
    """Per-target-claim coverage verdicts via the §6.6 transport reduction.

    Returns one `Covered` / `Missing` verdict per target claim, in input
    order. An empty target list short-circuits to an empty result with
    zero scorer calls. An empty source list emits `Missing` for every
    target — no real source can support any target, the transport engine
    routes all target mass to the slack column.
    """
    cfg = config or CoverageConfig()
    target_list = list(target)
    source_list = list(source)

    if not target_list:
        return CoverageResult(verdicts=[], scorer_calls=0)

    if not source_list:
        return CoverageResult(
            verdicts=["Missing"] * len(target_list),
            scorer_calls=0,
        )

    return _solve_via_transport(
        sources=source_list,
        targets=target_list,
        scorer=scorer,
        config=cfg,
    )


def list_check(
    *,
    items: Sequence[ClaimTuple],
    doc: Sequence[ClaimTuple],
    scorer: NLIScorer,
    config: CoverageConfig | None = None,
) -> CoverageResult:
    """Per-item verdicts of a list against a doc — §6.6 framing of `list_check`.

    Per the spec, `list_check(items, D) == coverage(items → D)`. The
    items are the *targets* being graded; the doc claims act as
    *sources*. The function exists as a separate surface so the §9 CLI
    can keep its naming aligned with user intent without diluting the
    `coverage` symbol.
    """
    return coverage(source=doc, target=items, scorer=scorer, config=config)


class TransportCoverageVerifier:
    """`CrossDocCoverageVerifier`-shaped consumer of the transport reduction.

    The §14 eval substrate's `CrossDocCoverageEvalRunner` consumes any
    object satisfying the `CrossDocCoverageVerifier` protocol; this
    class adapts the functional `coverage` surface onto that protocol
    so the same transport reduction is graded directly by the existing
    eval fixture without writing a second adapter.
    """

    def __init__(
        self,
        *,
        scorer: NLIScorer,
        config: CoverageConfig | None = None,
    ) -> None:
        self._scorer = scorer
        self._config = config or CoverageConfig()

    def verdicts(
        self,
        *,
        source: list[ClaimTuple],
        target: list[ClaimTuple],
    ) -> list[CoverageVerdictLiteral]:
        result = coverage(
            source=source,
            target=target,
            scorer=self._scorer,
            config=self._config,
        )
        return result.verdicts


# ---------------------------------------------------------------------------
# Internal — transport reduction
# ---------------------------------------------------------------------------


def _solve_via_transport(
    *,
    sources: list[ClaimTuple],
    targets: list[ClaimTuple],
    scorer: NLIScorer,
    config: CoverageConfig,
) -> CoverageResult:
    """Build the §6.6 cost matrix, solve, read off per-target verdicts."""
    n_src = len(sources)
    n_tgt = len(targets)

    # 1. Render claims into the natural-language surface the NLI backend
    # consumes. Determinism: identical input claims → identical text.
    source_text = [_render_claim(s) for s in sources]
    target_text = [_render_claim(t) for t in targets]

    # 2. Ask the scorer for entailment on every (source, target) pair.
    # |sources| * |targets| calls — the §6.6 cost contract.
    entail_matrix: list[list[float]] = [[0.0] * n_tgt for _ in range(n_src)]
    for i in range(n_src):
        for j in range(n_tgt):
            score = scorer.score(premise=source_text[i], hypothesis=target_text[j])
            entail_matrix[i][j] = score.entailment
    scorer_calls = n_src * n_tgt

    # 3. Assemble the balanced transport problem.
    #
    # Source rows: n_src real rows + 1 slack row (the "no real source
    # covers this target" sink). Source weights: 1.0 per real source,
    # n_tgt for the slack source (large enough to absorb every target).
    #
    # Target columns: n_tgt real columns + 1 absorption column. The
    # absorption column zero-costs all real-source mass that does not
    # need to land on a target (the source-side surplus); without it
    # the problem is unbalanced because total source mass exceeds
    # total target mass.
    #
    # Cost matrix:
    #   * real-row x real-col cell = `1 - entailment(s_i, t_j)`
    #   * slack-row x real-col cell = `1 - entailment_threshold`
    #   * real-row x absorption-col cell = 0 (free disposal)
    #   * slack-row x absorption-col cell = 0
    real_total = float(n_src)
    target_total = float(n_tgt)
    slack_total = target_total  # slack supplies enough to absorb every target
    absorption_demand = real_total  # absorption absorbs all real source mass

    source_weights = [1.0] * n_src + [slack_total]
    target_weights = [1.0] * n_tgt + [absorption_demand]

    slack_cost = 1.0 - config.entailment_threshold
    cost_matrix: list[list[float]] = []
    for i in range(n_src):
        row = [1.0 - entail_matrix[i][j] for j in range(n_tgt)]
        row.append(0.0)  # absorption column
        cost_matrix.append(row)
    # Slack row: slack_cost to every real target, 0 to absorption.
    slack_row = [slack_cost] * n_tgt + [0.0]
    cost_matrix.append(slack_row)

    # Sanity: total source mass equals total target mass within tolerance.
    # real_total + slack_total = n_src + n_tgt
    # target_total + absorption_demand = n_tgt + n_src
    # They are equal by construction; the TransportProblem validator is
    # the suspenders to this belt.

    problem = TransportProblem(
        source_weights=source_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
    )
    plan = min_cost_transport(problem)

    # 4. Read per-target verdict from the plan's flow matrix.
    # A target is `Missing` iff the slack row carries a strict majority
    # of its incoming mass (sum of all flows into target column j must
    # equal 1.0, so we compare slack vs the rest). Tiebreak: slack ties
    # go to `Missing` to keep the verifier conservative — a target with
    # no clearly dominant source is closer to uncovered than covered.
    slack_row_idx = n_src
    verdicts: list[CoverageVerdictLiteral] = []
    for j in range(n_tgt):
        slack_flow = plan.flow[slack_row_idx][j]
        real_flow = sum(plan.flow[i][j] for i in range(n_src))
        if slack_flow >= real_flow:
            verdicts.append("Missing")
        else:
            verdicts.append("Covered")

    return CoverageResult(verdicts=verdicts, scorer_calls=scorer_calls)


# ---------------------------------------------------------------------------
# Claim rendering — single source of truth for premise / hypothesis text
# ---------------------------------------------------------------------------


def _render_claim(claim: ClaimTuple) -> str:
    """Render a `ClaimTuple` as the natural-language surface for the NLI scorer.

    Polarity flips by splicing `does not` before the predicate for plain
    verbs — matches the `tier2_nli.render_claim_text` style so the same
    NLI backend sees the same surface regardless of where the claim
    originates. The qualifier slot trails the SVO trunk so a qualified
    claim reads naturally (e.g. *"the proxy drops idle connections after
    30 seconds"*).
    """
    subject = claim.subject.strip()
    predicate = claim.predicate.strip()
    obj = claim.object.strip()
    qualifier = claim.qualifier.strip()

    if claim.polarity == "negative":
        predicate = f"does not {predicate}"

    parts = [p for p in (subject, predicate, obj, qualifier) if p]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Convenience: bulk-iterable rendering for callers that want it
# ---------------------------------------------------------------------------


def _render_claims(claims: Iterable[ClaimTuple]) -> list[str]:
    """Vectorised wrapper around `_render_claim`."""
    return [_render_claim(c) for c in claims]


__all__ = [
    "COVERAGE_ENTAILMENT_THRESHOLD",
    "COVERAGE_VERDICT_ACCURACY_THRESHOLD",
    "CoverageConfig",
    "CoverageResult",
    "NLIScorer",
    "TransportCoverageVerifier",
    "coverage",
    "list_check",
]
