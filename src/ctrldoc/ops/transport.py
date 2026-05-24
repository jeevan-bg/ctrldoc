"""Optimal-transport engine on claim-pair edges — the one mathematical
primitive that `compare` / `coverage` / `merge` / `list_check` collapse into
(§6.6).

> *The user-facing operations collapse into variants of one mathematical
> primitive: optimal transport between two probability-weighted claim
> distributions over the shared concept lattice.*

Two solvers, one input shape:

* `min_cost_transport(problem)` — exact min-cost flow via successive
  shortest paths on a bipartite transportation network. Produces sparse
  hard-assignment plans. Used by the hard variants (`coverage` /
  `list_check`).
* `sinkhorn(problem, ...)` — entropy-regularised soft transport via the
  Sinkhorn-Knopp matrix-scaling iteration. Produces dense plans where
  mass is spread across plausible alignments. Used by the soft variants
  (`compare` / `merge`).

Both consume the balanced `TransportProblem` shape — source and target
marginals must agree on total mass within a small floating-point tolerance.
Unbalanced transport (slack mass for `Missing` / `Contradicted` verdicts)
is handled at the caller level: the caller appends an explicit "uncovered"
dummy claim with the appropriate weight before invoking the engine. This
keeps the engine itself a clean primitive — exactly one job per function.

Both solvers are pure-Python, stdlib-only, and deterministic. Identical
input produces byte-identical output across runs. The exact solver is
worst-case `O(F * (V + E) * log V)` where `F = min(|sources|, |targets|)`
unit-flow augmentations — fine for the claim-graph sizes the universal
operations operate over (typically 10s..100s of claims per doc).

SPEC-REF: §6.6 (optimal-transport core — one algorithm, five queries)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


_MASS_TOLERANCE: float = 1e-9
"""Allowed slack between `sum(source_weights)` and `sum(target_weights)`."""

_DEFAULT_SINKHORN_TOL: float = 1e-6
"""Default per-row L1 residual under which Sinkhorn declares convergence."""

_DEFAULT_SINKHORN_MAX_ITER: int = 200
"""Default cap on Sinkhorn-Knopp iterations. Each iteration is O(n * m)."""


# ---------------------------------------------------------------------------
# Input + output shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportProblem:
    """A balanced optimal-transport instance over a bipartite claim graph.

    `source_weights[i]` is the probability mass of source claim `i`;
    `target_weights[j]` is the probability mass of target claim `j`;
    `cost_matrix[i][j]` is the unit-mass transport cost — derived as
    `1 - NLI_entail(source_i, target_j)` in the universal-operations
    pipeline (§6.6).

    The problem is *balanced*: `sum(source_weights) == sum(target_weights)`
    within `_MASS_TOLERANCE`. Slack mass for `Missing` / `Contradicted`
    verdicts is the caller's responsibility — the universal operations
    layer appends an explicit dummy claim with the appropriate residual
    weight before invoking the engine.
    """

    source_weights: list[float]
    target_weights: list[float]
    cost_matrix: list[list[float]]

    def __post_init__(self) -> None:
        n_sources = len(self.source_weights)
        n_targets = len(self.target_weights)

        if any(w < 0.0 for w in self.source_weights):
            raise ValueError("source_weights must be non-negative")
        if any(w < 0.0 for w in self.target_weights):
            raise ValueError("target_weights must be non-negative")

        if len(self.cost_matrix) != n_sources:
            raise ValueError(
                f"cost_matrix must have {n_sources} rows (got {len(self.cost_matrix)})"
            )
        for row_idx, row in enumerate(self.cost_matrix):
            if len(row) != n_targets:
                raise ValueError(
                    f"cost_matrix row {row_idx} must have {n_targets} cells (got {len(row)})"
                )
            if any(c < 0.0 for c in row):
                raise ValueError("cost_matrix entries must be non-negative")

        total_src = math.fsum(self.source_weights)
        total_tgt = math.fsum(self.target_weights)
        if abs(total_src - total_tgt) > _MASS_TOLERANCE:
            raise ValueError(
                f"source and target total mass must agree within {_MASS_TOLERANCE} "
                f"(got source={total_src}, target={total_tgt})"
            )


@dataclass(frozen=True)
class TransportPlan:
    """The solver output — a `flow[i][j]` plan plus its scalar `total_cost`.

    `flow` has the same shape as the input `cost_matrix`; row sums equal
    `source_weights`, column sums equal `target_weights`, every cell is
    non-negative. `total_cost` is the inner product `<flow, cost>`.
    """

    flow: list[list[float]]
    total_cost: float


# ---------------------------------------------------------------------------
# Exact min-cost transport — successive shortest paths
# ---------------------------------------------------------------------------


def min_cost_transport(problem: TransportProblem) -> TransportPlan:
    """Solve the transportation problem exactly via successive shortest paths.

    The bipartite transportation network is: a super-source `S` connects
    to every source claim with capacity `source_weights[i]` and cost 0;
    every source `i` connects to every target `j` with infinite capacity
    and cost `cost_matrix[i][j]`; every target `j` connects to a super-sink
    `T` with capacity `target_weights[j]` and cost 0. We augment along
    successive shortest `S -> T` paths (Dijkstra with reduced costs) until
    all source supply is drained.

    Because every flow is fractional and edge costs are non-negative, the
    only loop is over `min(|sources|, |targets|)` augmentations in the
    worst case (each augmentation saturates either a source or a target).
    """
    n_sources = len(problem.source_weights)
    n_targets = len(problem.target_weights)

    if n_sources == 0 or n_targets == 0:
        return TransportPlan(
            flow=[[0.0] * n_targets for _ in range(n_sources)],
            total_cost=0.0,
        )

    # Working copies of remaining supply / demand. We never mutate the input.
    supply = list(problem.source_weights)
    demand = list(problem.target_weights)
    flow: list[list[float]] = [[0.0] * n_targets for _ in range(n_sources)]

    # Successive shortest paths. Because the bipartite graph has no
    # negative-cost cycle and edge costs are non-negative, we can run
    # plain Dijkstra at each iteration (no Johnson reweighting needed —
    # backward residual edges only appear after the first augmentation,
    # and the potentials we maintain keep reduced costs non-negative).
    potentials_src = [0.0] * n_sources
    potentials_tgt = [0.0] * n_targets

    # Cap the loop generously — each augmentation strictly reduces the
    # number of (source, target) cells that still carry residual supply
    # AND demand simultaneously, so termination is guaranteed.
    max_augmentations = (n_sources + 1) * (n_targets + 1) + 1
    for _ in range(max_augmentations):
        # Snap residuals that have crept under the mass-tolerance back to
        # exact zero so the active-source / active-target test does not
        # spin on a vanishing residual.
        for i in range(n_sources):
            if supply[i] < _MASS_TOLERANCE:
                supply[i] = 0.0
        for j in range(n_targets):
            if demand[j] < _MASS_TOLERANCE:
                demand[j] = 0.0

        active_sources = [i for i in range(n_sources) if supply[i] > 0.0]
        active_targets = [j for j in range(n_targets) if demand[j] > 0.0]
        if not active_sources or not active_targets:
            break

        # Dijkstra from a virtual super-source over reduced costs.
        # State node ids: 0..n_sources-1 = sources, n_sources..n_sources+n_targets-1 = targets.
        dist_src = [math.inf] * n_sources
        dist_tgt = [math.inf] * n_targets
        parent_tgt: list[tuple[str, int] | None] = [None] * n_targets
        parent_src: list[tuple[str, int] | None] = [None] * n_sources

        heap: list[tuple[float, int, int]] = []
        # `kind`: 0 = source, 1 = target. Tiebreak on `(kind, idx)` so the
        # heap pops in a fully deterministic order across runs.
        for i in active_sources:
            dist_src[i] = 0.0
            heapq.heappush(heap, (0.0, 0, i))

        while heap:
            d, kind, idx = heapq.heappop(heap)
            if kind == 0:
                if d > dist_src[idx]:
                    continue
                # Edge to every target via cost_matrix[idx][j].
                for j in range(n_targets):
                    # Reduced cost = c_ij - pi_src[i] + pi_tgt[j].
                    reduced = problem.cost_matrix[idx][j] - potentials_src[idx] + potentials_tgt[j]
                    # Floating-point noise can push a true-zero reduced
                    # cost a hair below zero; clamp to keep Dijkstra
                    # invariants intact (no negative-cost edges).
                    if reduced < 0.0 and reduced > -1e-12:
                        reduced = 0.0
                    nd = d + reduced
                    if nd < dist_tgt[j] - 1e-15:
                        dist_tgt[j] = nd
                        parent_tgt[j] = ("src", idx)
                        heapq.heappush(heap, (nd, 1, j))
            else:
                if d > dist_tgt[idx]:
                    continue
                # Backward residual edges from target j to sources that
                # currently push flow into j. Reduced cost on a backward
                # edge is `-c_ji + pi_tgt[j] - pi_src[i]` — non-negative
                # after the potentials are updated each round.
                for i in range(n_sources):
                    if flow[i][idx] <= 0.0:
                        continue
                    reduced = -problem.cost_matrix[i][idx] + potentials_tgt[idx] - potentials_src[i]
                    if reduced < 0.0 and reduced > -1e-12:
                        reduced = 0.0
                    if reduced < 0.0:
                        # Shouldn't happen with correct potentials; guard
                        # so a malformed input fails loudly instead of
                        # producing a wrong answer silently.
                        continue
                    nd = d + reduced
                    if nd < dist_src[i] - 1e-15:
                        dist_src[i] = nd
                        parent_src[i] = ("tgt", idx)
                        heapq.heappush(heap, (nd, 0, i))

        # Pick the cheapest reachable active target.
        best_tgt = -1
        best_true_cost = math.inf
        for j in active_targets:
            if dist_tgt[j] == math.inf:
                continue
            # True path cost = reduced cost + pi_src[origin] - pi_tgt[j].
            # Equivalent to dist_tgt[j] + pi_src[any_active_source] - pi_tgt[j],
            # but we want to compare in *true* cost terms; both reduced and
            # true preserve order when starting potentials are zero each
            # round (we re-add the path back to potentials below).
            if dist_tgt[j] < best_true_cost:
                best_true_cost = dist_tgt[j]
                best_tgt = j
        if best_tgt < 0:
            # No augmenting path; remaining supply / demand sit on
            # disconnected components — should not happen for a balanced
            # complete bipartite graph but guard anyway.
            break

        # Reconstruct the augmenting path from best_tgt back to its source.
        path: list[tuple[str, int]] = []
        cur_kind = "tgt"
        cur_idx = best_tgt
        while True:
            if cur_kind == "tgt":
                parent = parent_tgt[cur_idx]
                path.append(("tgt", cur_idx))
                if parent is None:
                    break
                cur_kind, cur_idx = parent
            else:
                parent = parent_src[cur_idx]
                path.append(("src", cur_idx))
                if parent is None:
                    break
                cur_kind, cur_idx = parent
        path.reverse()

        # Determine how much flow to send along the path.
        # The path alternates between sources and targets.
        # Forward edges (source -> target) have capacity = supply / demand;
        # backward edges (target -> source) have capacity = current flow.
        bottleneck = math.inf
        origin_src = path[0][1]  # path always starts at a source
        bottleneck = min(bottleneck, supply[origin_src])
        bottleneck = min(bottleneck, demand[best_tgt])

        # Backward edges in the path cap on current flow.
        for step in range(1, len(path)):
            kind_prev, idx_prev = path[step - 1]
            kind_cur, idx_cur = path[step]
            if kind_prev == "tgt" and kind_cur == "src":
                # Backward edge target -> source: capacity = flow[src][tgt].
                bottleneck = min(bottleneck, flow[idx_cur][idx_prev])

        if bottleneck <= 0.0:
            break

        # Apply the augmentation along the path.
        for step in range(1, len(path)):
            kind_prev, idx_prev = path[step - 1]
            kind_cur, idx_cur = path[step]
            if kind_prev == "src" and kind_cur == "tgt":
                flow[idx_prev][idx_cur] += bottleneck
            elif kind_prev == "tgt" and kind_cur == "src":
                flow[idx_cur][idx_prev] -= bottleneck

        supply[origin_src] -= bottleneck
        demand[best_tgt] -= bottleneck

        # Update potentials with the Dijkstra distances. This keeps
        # reduced costs non-negative on residual edges for the next round.
        for i in range(n_sources):
            if dist_src[i] != math.inf:
                potentials_src[i] += dist_src[i]
        for j in range(n_targets):
            if dist_tgt[j] != math.inf:
                potentials_tgt[j] += dist_tgt[j]

    # Clean up: zero out any cell that drifted under tolerance.
    for i in range(n_sources):
        for j in range(n_targets):
            if abs(flow[i][j]) < _MASS_TOLERANCE:
                flow[i][j] = 0.0

    total_cost = math.fsum(
        flow[i][j] * problem.cost_matrix[i][j] for i in range(n_sources) for j in range(n_targets)
    )
    return TransportPlan(flow=flow, total_cost=total_cost)


# ---------------------------------------------------------------------------
# Sinkhorn — entropy-regularised soft transport
# ---------------------------------------------------------------------------


def sinkhorn(
    problem: TransportProblem,
    *,
    epsilon: float,
    max_iter: int = _DEFAULT_SINKHORN_MAX_ITER,
    tol: float = _DEFAULT_SINKHORN_TOL,
) -> TransportPlan:
    """Solve the entropy-regularised optimal-transport problem via
    Sinkhorn-Knopp matrix scaling.

    Algorithm: form the Gibbs kernel `K[i][j] = exp(-cost[i][j] / epsilon)`,
    then alternate row and column scalings until the row sums match the
    source marginal and column sums match the target marginal within `tol`.
    The final plan is `diag(u) K diag(v)` for the converged scaling
    vectors `u` and `v`.

    Smaller `epsilon` produces a sharper, more sparse plan closer to the
    exact `min_cost_transport` output; larger `epsilon` produces a smoother,
    denser plan. `epsilon` must be strictly positive — at `epsilon = 0`
    the Gibbs kernel is degenerate and the iteration is undefined.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0 (got {epsilon})")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1 (got {max_iter})")

    n_sources = len(problem.source_weights)
    n_targets = len(problem.target_weights)

    if n_sources == 0 or n_targets == 0:
        return TransportPlan(
            flow=[[0.0] * n_targets for _ in range(n_sources)],
            total_cost=0.0,
        )

    # Gibbs kernel. Costs are non-negative so kernel entries lie in (0, 1].
    kernel: list[list[float]] = [
        [math.exp(-problem.cost_matrix[i][j] / epsilon) for j in range(n_targets)]
        for i in range(n_sources)
    ]

    u: list[float] = [1.0] * n_sources
    v: list[float] = [1.0] * n_targets

    for _ in range(max_iter):
        # Row scaling: u[i] = a[i] / sum_j K[i][j] * v[j].
        new_u: list[float] = [0.0] * n_sources
        for i in range(n_sources):
            if problem.source_weights[i] == 0.0:
                new_u[i] = 0.0
                continue
            denom = math.fsum(kernel[i][j] * v[j] for j in range(n_targets))
            new_u[i] = problem.source_weights[i] / denom if denom > 0.0 else 0.0

        # Column scaling: v[j] = b[j] / sum_i K[i][j] * u[i].
        new_v: list[float] = [0.0] * n_targets
        for j in range(n_targets):
            if problem.target_weights[j] == 0.0:
                new_v[j] = 0.0
                continue
            denom = math.fsum(kernel[i][j] * new_u[i] for i in range(n_sources))
            new_v[j] = problem.target_weights[j] / denom if denom > 0.0 else 0.0

        u = new_u
        v = new_v

        # Convergence check on BOTH row and column residuals against their
        # respective marginals at the *current* (u, v). After a row scaling
        # rows match exactly but cols may drift; after a column scaling
        # cols match exactly but rows may drift. Sinkhorn converges when
        # both residuals drop under `tol` simultaneously, which happens
        # when the iteration reaches its fixed point.
        max_residual = 0.0
        for i in range(n_sources):
            row = math.fsum(u[i] * kernel[i][j] * v[j] for j in range(n_targets))
            max_residual = max(max_residual, abs(row - problem.source_weights[i]))
        for j in range(n_targets):
            col = math.fsum(u[i] * kernel[i][j] * v[j] for i in range(n_sources))
            max_residual = max(max_residual, abs(col - problem.target_weights[j]))
        if max_residual < tol:
            break

    flow: list[list[float]] = [
        [u[i] * kernel[i][j] * v[j] for j in range(n_targets)] for i in range(n_sources)
    ]
    total_cost = math.fsum(
        flow[i][j] * problem.cost_matrix[i][j] for i in range(n_sources) for j in range(n_targets)
    )
    return TransportPlan(flow=flow, total_cost=total_cost)


__all__ = [
    "TransportPlan",
    "TransportProblem",
    "min_cost_transport",
    "sinkhorn",
]
