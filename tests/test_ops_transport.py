"""Optimal-transport engine on claim-pair edges — `min_cost_transport` (exact
min-cost flow) and `sinkhorn` (entropy-regularised soft transport).

Per §6.6 the universal operations `compare`, `coverage`, `merge`, and
`list_check` collapse into variants of one mathematical primitive: optimal
transport between two probability-weighted claim distributions over the
shared concept lattice, with edge costs derived from `1 - NLI_entail`. This
module is that primitive — pure-Python, deterministic, no scipy.

The engine ships two solvers:

* `min_cost_transport(problem)` — exact min-cost flow via successive
  shortest paths on a bipartite transportation network. Hard assignments
  with possibly many-to-one transport (one target claim jointly covered by
  several source claims). Used by `coverage` / `list_check` where the
  semantic is "what did the source actually cover?".
* `sinkhorn(problem, ...)` — entropy-regularised soft transport via the
  Sinkhorn-Knopp matrix-scaling iteration. Produces dense plans where the
  mass is spread across plausible alignments, never concentrated on a
  single edge. Used by `compare` / `merge` where the semantic is "how do
  these two distributions overlap?".

Both solvers operate on the same `TransportProblem` shape (`source_weights`,
`target_weights`, `cost_matrix`) and emit the same `TransportPlan` shape
(`flow`, `total_cost`). Determinism is a hard requirement: identical input
must produce byte-identical output across runs, on any platform, regardless
of dict-iteration order.

SPEC-REF: §6.6 (optimal-transport core — one algorithm, five queries)
"""

from __future__ import annotations

import math

import pytest

from ctrldoc.ops.transport import (
    TransportPlan,
    TransportProblem,
    min_cost_transport,
    sinkhorn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_equal(a: float, b: float, *, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _row_sums(flow: list[list[float]]) -> list[float]:
    return [sum(row) for row in flow]


def _col_sums(flow: list[list[float]]) -> list[float]:
    if not flow:
        return []
    n_cols = len(flow[0])
    return [sum(row[j] for row in flow) for j in range(n_cols)]


# ---------------------------------------------------------------------------
# TransportProblem validation
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_problem_rejects_mismatched_cost_matrix_rows() -> None:
    """`cost_matrix` must have exactly `len(source_weights)` rows."""
    with pytest.raises(ValueError, match="cost_matrix"):
        TransportProblem(
            source_weights=[0.5, 0.5],
            target_weights=[1.0],
            cost_matrix=[[0.1]],  # only 1 row; expected 2
        )


@pytest.mark.family_determinism
def test_problem_rejects_mismatched_cost_matrix_cols() -> None:
    """Every row of `cost_matrix` must have exactly `len(target_weights)` cells."""
    with pytest.raises(ValueError, match="cost_matrix"):
        TransportProblem(
            source_weights=[1.0],
            target_weights=[0.5, 0.5],
            cost_matrix=[[0.1]],  # 1 col; expected 2
        )


@pytest.mark.family_determinism
def test_problem_rejects_negative_weight() -> None:
    """Source / target weights must be non-negative — mass cannot be negative."""
    with pytest.raises(ValueError, match="non-negative"):
        TransportProblem(
            source_weights=[1.0, -0.1],
            target_weights=[0.9],
            cost_matrix=[[0.0], [0.0]],
        )


@pytest.mark.family_determinism
def test_problem_rejects_negative_cost() -> None:
    """`1 - NLI_entail` is bounded in [0, 1]; negative costs are a contract bug."""
    with pytest.raises(ValueError, match="non-negative"):
        TransportProblem(
            source_weights=[1.0],
            target_weights=[1.0],
            cost_matrix=[[-0.5]],
        )


@pytest.mark.family_determinism
def test_problem_rejects_unbalanced_total_mass() -> None:
    """Source and target marginals must agree within tolerance — the
    transportation problem is *balanced* in this engine (slack handled by
    explicit dummy claims at the caller level, not silently here)."""
    with pytest.raises(ValueError, match="total mass"):
        TransportProblem(
            source_weights=[1.0, 1.0],
            target_weights=[1.0],
            cost_matrix=[[0.0], [0.0]],
        )


# ---------------------------------------------------------------------------
# min_cost_transport — exact
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_min_cost_one_to_one_assignment() -> None:
    """Two sources, two targets, identity-cheap pairing wins.

    Source i should align to target i because the diagonal costs are cheaper
    than the cross-diagonal ones.
    """
    problem = TransportProblem(
        source_weights=[0.5, 0.5],
        target_weights=[0.5, 0.5],
        cost_matrix=[
            [0.1, 0.9],
            [0.9, 0.1],
        ],
    )
    plan = min_cost_transport(problem)

    assert isinstance(plan, TransportPlan)
    assert _approx_equal(plan.flow[0][0], 0.5)
    assert _approx_equal(plan.flow[0][1], 0.0)
    assert _approx_equal(plan.flow[1][0], 0.0)
    assert _approx_equal(plan.flow[1][1], 0.5)
    assert _approx_equal(plan.total_cost, 0.1 * 0.5 + 0.1 * 0.5)


@pytest.mark.family_determinism
def test_min_cost_many_to_one_transport() -> None:
    """Two sources jointly cover one target (§6.6 many-to-one is supported)."""
    problem = TransportProblem(
        source_weights=[0.4, 0.6],
        target_weights=[1.0],
        cost_matrix=[
            [0.2],
            [0.3],
        ],
    )
    plan = min_cost_transport(problem)

    # All source mass flows to the single target — the only feasible plan.
    assert _approx_equal(plan.flow[0][0], 0.4)
    assert _approx_equal(plan.flow[1][0], 0.6)
    assert _approx_equal(plan.total_cost, 0.4 * 0.2 + 0.6 * 0.3)


@pytest.mark.family_determinism
def test_min_cost_marginals_preserved() -> None:
    """Row sums equal source weights; column sums equal target weights."""
    problem = TransportProblem(
        source_weights=[0.3, 0.3, 0.4],
        target_weights=[0.5, 0.5],
        cost_matrix=[
            [0.1, 0.4],
            [0.5, 0.2],
            [0.3, 0.3],
        ],
    )
    plan = min_cost_transport(problem)

    for i, expected in enumerate(problem.source_weights):
        assert _approx_equal(sum(plan.flow[i]), expected)
    for j, expected in enumerate(problem.target_weights):
        assert _approx_equal(sum(plan.flow[i][j] for i in range(3)), expected)


@pytest.mark.family_determinism
def test_min_cost_handles_zero_weight_source() -> None:
    """A zero-mass source produces zero flow on its row but doesn't break the solve."""
    problem = TransportProblem(
        source_weights=[0.0, 1.0],
        target_weights=[1.0],
        cost_matrix=[
            [0.0],
            [0.5],
        ],
    )
    plan = min_cost_transport(problem)

    assert _approx_equal(plan.flow[0][0], 0.0)
    assert _approx_equal(plan.flow[1][0], 1.0)
    assert _approx_equal(plan.total_cost, 0.5)


@pytest.mark.family_determinism
def test_min_cost_handles_zero_weight_target() -> None:
    """A zero-mass target absorbs no flow — its column sum is zero."""
    problem = TransportProblem(
        source_weights=[1.0],
        target_weights=[0.0, 1.0],
        cost_matrix=[[0.0, 0.5]],
    )
    plan = min_cost_transport(problem)

    assert _approx_equal(plan.flow[0][0], 0.0)
    assert _approx_equal(plan.flow[0][1], 1.0)
    assert _approx_equal(plan.total_cost, 0.5)


@pytest.mark.family_determinism
def test_min_cost_zero_cost_diagonal_is_free() -> None:
    """When every diagonal edge has cost 0, total cost is exactly 0."""
    problem = TransportProblem(
        source_weights=[0.25, 0.25, 0.25, 0.25],
        target_weights=[0.25, 0.25, 0.25, 0.25],
        cost_matrix=[
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
    )
    plan = min_cost_transport(problem)
    assert _approx_equal(plan.total_cost, 0.0, tol=1e-9)
    for i in range(4):
        assert _approx_equal(plan.flow[i][i], 0.25)


@pytest.mark.family_determinism
def test_min_cost_single_cell_problem() -> None:
    """1 source, 1 target — trivially full flow on the single edge."""
    problem = TransportProblem(
        source_weights=[1.0],
        target_weights=[1.0],
        cost_matrix=[[0.42]],
    )
    plan = min_cost_transport(problem)
    assert _approx_equal(plan.flow[0][0], 1.0)
    assert _approx_equal(plan.total_cost, 0.42)


@pytest.mark.family_determinism
def test_min_cost_is_deterministic_across_repeat_runs() -> None:
    """Identical input must produce byte-identical output."""
    problem = TransportProblem(
        source_weights=[0.1, 0.2, 0.3, 0.4],
        target_weights=[0.4, 0.3, 0.2, 0.1],
        cost_matrix=[
            [0.5, 0.7, 0.2, 0.9],
            [0.3, 0.4, 0.6, 0.8],
            [0.7, 0.1, 0.5, 0.4],
            [0.8, 0.6, 0.3, 0.2],
        ],
    )
    first = min_cost_transport(problem)
    second = min_cost_transport(problem)
    assert first.flow == second.flow
    assert first.total_cost == second.total_cost


@pytest.mark.family_determinism
def test_min_cost_optimal_beats_naive_diagonal() -> None:
    """When the cheap pairing is off-diagonal, the solver finds it."""
    # Cheapest pairing: src0 -> tgt1, src1 -> tgt0 (cross-diagonal).
    problem = TransportProblem(
        source_weights=[0.5, 0.5],
        target_weights=[0.5, 0.5],
        cost_matrix=[
            [0.9, 0.1],
            [0.1, 0.9],
        ],
    )
    plan = min_cost_transport(problem)

    assert _approx_equal(plan.flow[0][1], 0.5)
    assert _approx_equal(plan.flow[1][0], 0.5)
    assert _approx_equal(plan.total_cost, 0.1 * 0.5 + 0.1 * 0.5)


@pytest.mark.family_determinism
def test_min_cost_chained_split_plan() -> None:
    """One source splits across two targets when target capacities force it.

    src0 carries 1.0; tgt0 / tgt1 each demand 0.5 — the cheaper cell wins
    until its target is saturated, then the second target absorbs the rest.
    """
    problem = TransportProblem(
        source_weights=[1.0],
        target_weights=[0.5, 0.5],
        cost_matrix=[[0.1, 0.4]],
    )
    plan = min_cost_transport(problem)

    assert _approx_equal(plan.flow[0][0], 0.5)
    assert _approx_equal(plan.flow[0][1], 0.5)
    assert _approx_equal(plan.total_cost, 0.5 * 0.1 + 0.5 * 0.4)


@pytest.mark.family_determinism
def test_min_cost_three_by_three_optimal() -> None:
    """3x3 asymmetric assignment with a known-optimal pairing.

    The cheapest perfect assignment in this matrix pairs (0->2, 1->0, 2->1)
    for total cost 0.1 + 0.1 + 0.1 = 0.3. Any other assignment costs more.
    """
    problem = TransportProblem(
        source_weights=[1.0, 1.0, 1.0],
        target_weights=[1.0, 1.0, 1.0],
        cost_matrix=[
            [0.9, 0.9, 0.1],
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
        ],
    )
    plan = min_cost_transport(problem)

    assert _approx_equal(plan.flow[0][2], 1.0)
    assert _approx_equal(plan.flow[1][0], 1.0)
    assert _approx_equal(plan.flow[2][1], 1.0)
    assert _approx_equal(plan.total_cost, 0.3)


@pytest.mark.family_determinism
def test_min_cost_handles_empty_problem() -> None:
    """Zero sources and zero targets is a degenerate but legal problem."""
    problem = TransportProblem(
        source_weights=[],
        target_weights=[],
        cost_matrix=[],
    )
    plan = min_cost_transport(problem)
    assert plan.flow == []
    assert _approx_equal(plan.total_cost, 0.0)


@pytest.mark.family_determinism
def test_min_cost_total_cost_matches_flow_inner_product() -> None:
    """`total_cost == sum(flow[i][j] * cost[i][j])` — engine bookkeeping."""
    problem = TransportProblem(
        source_weights=[0.5, 0.5],
        target_weights=[0.4, 0.6],
        cost_matrix=[
            [0.3, 0.7],
            [0.8, 0.2],
        ],
    )
    plan = min_cost_transport(problem)

    manual = sum(
        plan.flow[i][j] * problem.cost_matrix[i][j]
        for i in range(len(problem.source_weights))
        for j in range(len(problem.target_weights))
    )
    assert _approx_equal(plan.total_cost, manual)


# ---------------------------------------------------------------------------
# sinkhorn — soft entropy-regularised transport
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_sinkhorn_marginals_preserved() -> None:
    """Sinkhorn output row / column sums match input marginals within tol."""
    problem = TransportProblem(
        source_weights=[0.4, 0.6],
        target_weights=[0.3, 0.7],
        cost_matrix=[
            [0.1, 0.5],
            [0.6, 0.2],
        ],
    )
    plan = sinkhorn(problem, epsilon=0.1)

    row = _row_sums(plan.flow)
    col = _col_sums(plan.flow)
    for got, want in zip(row, problem.source_weights, strict=True):
        assert _approx_equal(got, want, tol=1e-4)
    for got, want in zip(col, problem.target_weights, strict=True):
        assert _approx_equal(got, want, tol=1e-4)


@pytest.mark.family_determinism
def test_sinkhorn_approaches_min_cost_as_epsilon_shrinks() -> None:
    """At small epsilon Sinkhorn concentrates mass on the cheapest edges,
    so its total cost converges toward the exact min-cost transport.
    """
    problem = TransportProblem(
        source_weights=[0.5, 0.5],
        target_weights=[0.5, 0.5],
        cost_matrix=[
            [0.1, 0.9],
            [0.9, 0.1],
        ],
    )
    exact = min_cost_transport(problem)
    high_entropy = sinkhorn(problem, epsilon=1.0)
    low_entropy = sinkhorn(problem, epsilon=0.02)

    assert low_entropy.total_cost <= high_entropy.total_cost + 1e-9
    assert _approx_equal(low_entropy.total_cost, exact.total_cost, tol=1e-2)


@pytest.mark.family_determinism
def test_sinkhorn_is_deterministic_across_repeat_runs() -> None:
    """Same input + same epsilon + same max_iter → byte-identical flow."""
    problem = TransportProblem(
        source_weights=[0.25, 0.25, 0.25, 0.25],
        target_weights=[0.25, 0.25, 0.25, 0.25],
        cost_matrix=[
            [0.1, 0.5, 0.3, 0.4],
            [0.4, 0.2, 0.6, 0.3],
            [0.5, 0.3, 0.1, 0.5],
            [0.3, 0.4, 0.5, 0.2],
        ],
    )
    first = sinkhorn(problem, epsilon=0.1, max_iter=200)
    second = sinkhorn(problem, epsilon=0.1, max_iter=200)
    assert first.flow == second.flow
    assert first.total_cost == second.total_cost


@pytest.mark.family_determinism
def test_sinkhorn_rejects_non_positive_epsilon() -> None:
    """epsilon = 0 collapses to division-by-zero; must be rejected."""
    problem = TransportProblem(
        source_weights=[1.0],
        target_weights=[1.0],
        cost_matrix=[[0.1]],
    )
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn(problem, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn(problem, epsilon=-0.1)


@pytest.mark.family_determinism
def test_sinkhorn_flow_is_strictly_positive_at_high_epsilon() -> None:
    """High entropy → no edge is exactly zero (every cell carries some mass).

    This is the distinguishing feature vs `min_cost_transport`, which
    yields sparse plans.
    """
    problem = TransportProblem(
        source_weights=[0.5, 0.5],
        target_weights=[0.5, 0.5],
        cost_matrix=[
            [0.1, 0.9],
            [0.9, 0.1],
        ],
    )
    plan = sinkhorn(problem, epsilon=2.0)

    for row in plan.flow:
        for cell in row:
            assert cell > 0.0


@pytest.mark.family_determinism
def test_sinkhorn_total_cost_matches_flow_inner_product() -> None:
    """Same bookkeeping invariant as the exact solver."""
    problem = TransportProblem(
        source_weights=[0.3, 0.7],
        target_weights=[0.4, 0.6],
        cost_matrix=[
            [0.2, 0.7],
            [0.8, 0.3],
        ],
    )
    plan = sinkhorn(problem, epsilon=0.1)
    manual = sum(plan.flow[i][j] * problem.cost_matrix[i][j] for i in range(2) for j in range(2))
    assert _approx_equal(plan.total_cost, manual, tol=1e-9)


@pytest.mark.family_determinism
def test_sinkhorn_converges_within_tolerance() -> None:
    """At default max_iter, marginal residual must be below tolerance."""
    problem = TransportProblem(
        source_weights=[0.1, 0.2, 0.3, 0.4],
        target_weights=[0.4, 0.3, 0.2, 0.1],
        cost_matrix=[
            [0.5, 0.7, 0.2, 0.9],
            [0.3, 0.4, 0.6, 0.8],
            [0.7, 0.1, 0.5, 0.4],
            [0.8, 0.6, 0.3, 0.2],
        ],
    )
    plan = sinkhorn(problem, epsilon=0.1, max_iter=500, tol=1e-6)

    # Per-row residuals tight.
    for got, want in zip(_row_sums(plan.flow), problem.source_weights, strict=True):
        assert math.isclose(got, want, abs_tol=1e-5)
    for got, want in zip(_col_sums(plan.flow), problem.target_weights, strict=True):
        assert math.isclose(got, want, abs_tol=1e-5)
