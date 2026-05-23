"""Continuous-canary signature comparison.

A `CanaryBaseline` records, per `(doc_id, playbook)`, a deterministic
reduction of the playbook's output ("signature") plus a sha256 hash
of that reduction's canonical form. CI compares the live signature
against the pinned baseline and fails when the per-key drift exceeds
`DEFAULT_DRIFT_THRESHOLD` (10% per §8.6).

A signature is `dict[str, list[str]]`: each key is a stable invariant
the playbook emits (e.g. `chunk_ids`, `cited_chunk_ids`, `verdicts`),
each value the sorted list of items for that invariant. The
list-of-strings shape keeps the file format JSON-portable and
diff-friendly; richer types would couple the canary to the playbook's
internal models.

`signature_hash_of` produces a stable hash from canonical-JSON of the
signature so a fast-path equality check is one string comparison.

SPEC-REF: §8.6 (continuous canary)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DRIFT_THRESHOLD = 0.10


def signature_hash_of(signature: dict[str, list[str]]) -> str:
    """Stable sha256 over canonical-JSON of the signature.

    Each key's list is sorted before hashing so the hash is invariant
    under input-order rearrangement; the dict itself is also
    `sort_keys=True` serialised. Two equal signatures always produce
    the same hash; the converse holds modulo sha256.
    """
    normalised = {key: sorted(values) for key, values in signature.items()}
    payload = json.dumps(normalised, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CanaryBaseline(BaseModel):
    """One pinned `(doc_id, playbook)` signature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    playbook: str
    signature: dict[str, list[str]]
    signature_hash: str = Field(
        ...,
        description="sha256 of canonical-JSON of the signature; redundant with"
        " `signature` but enables fast equality checks against drifted runs.",
    )

    @classmethod
    def from_signature(
        cls,
        *,
        doc_id: str,
        playbook: str,
        signature: dict[str, list[str]],
    ) -> CanaryBaseline:
        return cls(
            doc_id=doc_id,
            playbook=playbook,
            signature={key: sorted(values) for key, values in signature.items()},
            signature_hash=signature_hash_of(signature),
        )


@dataclass(frozen=True)
class SignatureDrift:
    """Per-key drift fraction between baseline and current signature."""

    key: str
    baseline_size: int
    current_size: int
    symmetric_difference: int
    fraction: float

    @property
    def healthy(self) -> bool:
        return self.fraction <= DEFAULT_DRIFT_THRESHOLD


def compute_drift(
    baseline: dict[str, list[str]],
    current: dict[str, list[str]],
) -> list[SignatureDrift]:
    """Per-key drift fractions across the union of baseline + current keys.

    For each key, the drift fraction is `|symmetric_difference| /
    |union|` (Jaccard distance). A key present in only one side has a
    drift of 1.0.
    """
    all_keys = set(baseline) | set(current)
    drifts: list[SignatureDrift] = []
    for key in sorted(all_keys):
        baseline_set = set(baseline.get(key, []))
        current_set = set(current.get(key, []))
        union = baseline_set | current_set
        sym_diff = baseline_set ^ current_set
        fraction = 0.0 if not union else len(sym_diff) / len(union)
        drifts.append(
            SignatureDrift(
                key=key,
                baseline_size=len(baseline_set),
                current_size=len(current_set),
                symmetric_difference=len(sym_diff),
                fraction=fraction,
            )
        )
    return drifts


@dataclass(frozen=True)
class CanaryReport:
    """Outcome of comparing a live signature against the pinned baseline."""

    doc_id: str
    playbook: str
    drifts: tuple[SignatureDrift, ...]
    hash_match: bool
    threshold: float

    @property
    def max_drift(self) -> float:
        if not self.drifts:
            return 0.0
        return max(drift.fraction for drift in self.drifts)

    @property
    def flagged_keys(self) -> list[str]:
        return [drift.key for drift in self.drifts if drift.fraction > self.threshold]

    @property
    def passed(self) -> bool:
        return not self.flagged_keys


def check_canary(
    baseline: CanaryBaseline,
    current_signature: dict[str, list[str]],
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> CanaryReport:
    """Compute drift between `baseline` and `current_signature`.

    The returned report carries:
      - `hash_match` — fast-path equality (true when the canonical
        hashes agree).
      - `drifts` — per-key drift fractions across the union of keys.
      - `flagged_keys` — keys whose drift exceeds `threshold`.
      - `passed` — true iff every key is within threshold.

    A `hash_match=True` report always has `passed=True`; the inverse
    is not necessarily true since two distinct signatures can share
    a near-perfect signature (e.g. one extra unrelated id with many
    overlapping ones) while still passing the per-key threshold.
    """
    current_hash = signature_hash_of(current_signature)
    drifts = compute_drift(baseline.signature, current_signature)
    return CanaryReport(
        doc_id=baseline.doc_id,
        playbook=baseline.playbook,
        drifts=tuple(drifts),
        hash_match=(current_hash == baseline.signature_hash),
        threshold=threshold,
    )


def save_baseline(path: Path, baseline: CanaryBaseline) -> None:
    """Write a baseline to disk as pretty-printed canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        baseline.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def load_baseline(path: Path) -> CanaryBaseline:
    """Read a baseline from disk and re-validate against the schema."""
    return CanaryBaseline.model_validate_json(path.read_text(encoding="utf-8"))


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
