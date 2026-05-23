"""Continuous-canary primitives.

One production-size doc is run through every playbook on every
commit. Each playbook's stable signature (a deterministic reduction
of its output) is pinned in `tests/canary/baselines/`. CI fails when
the live signature drifts more than 10% from the baseline. The
substrate here is the comparison primitive, not the runner — the
runner is a thin wrapper a future CI step composes over the
playbook fan-out.

SPEC-REF: §8.6 (cross-cutting)
"""

from __future__ import annotations

from ctrldoc.canary.canary import (
    DEFAULT_DRIFT_THRESHOLD,
    CanaryBaseline,
    CanaryReport,
    SignatureDrift,
    check_canary,
    compute_drift,
    load_baseline,
    save_baseline,
    signature_hash_of,
)

__all__ = [
    "DEFAULT_DRIFT_THRESHOLD",
    "CanaryBaseline",
    "CanaryReport",
    "SignatureDrift",
    "check_canary",
    "compute_drift",
    "load_baseline",
    "save_baseline",
    "signature_hash_of",
]
