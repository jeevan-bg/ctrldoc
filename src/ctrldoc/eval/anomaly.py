"""anomaly_eval — precision on the triage queue.

Each case carries inline chunks + sections (enough to populate a
test `InMemoryStore`) and a list of `SeededAnomaly` entries
describing the anomalies the detector battery should flag. The
runner builds the store, runs `AnomalyScanPlaybook`, and scores
precision over the resulting queue. Per §8.2 the threshold is
`triage_precision ≥ 0.60`.

The seeded anomaly's `detector` field names the source detector
(matched against `Finding.ctrldoc`) and `claim_pattern` is a
substring expected in the finding's claim text — same shape as
analytical_eval's `SeededWeakness`.

SPEC-REF: §8.1 (anomaly_eval), §8.2 (anomaly_scan metrics)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ctrldoc.eval.harness import EvalResult
from ctrldoc.models import Chunk, Finding, Section
from ctrldoc.ops.scan import AnomalyScanPlaybook, Detector
from ctrldoc.store.memory import InMemoryStore

TRIAGE_PRECISION_THRESHOLD = 0.60


class SeededAnomaly(BaseModel):
    """One known anomaly the detector battery should surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    detector: str
    claim_pattern: str


class ChunkSeed(BaseModel):
    """Minimal payload the runner expands into a full `Chunk`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    section_id: str
    text: str


class SectionSeed(BaseModel):
    """Minimal payload the runner expands into a full `Section`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str
    title: str
    summary: str = ""


class AnomalyEvalCase(BaseModel):
    """One row in anomaly_eval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tags: list[str] = []
    chunks: list[ChunkSeed] = []
    sections: list[SectionSeed] = []
    seeded_anomalies: list[SeededAnomaly]


def is_true_positive(finding: Finding, seeded: list[SeededAnomaly]) -> bool:
    """True when at least one seeded anomaly matches the finding."""
    return any(
        finding.ctrldoc == entry.detector and entry.claim_pattern.lower() in finding.claim.lower()
        for entry in seeded
    )


def triage_precision(findings: list[Finding], seeded: list[SeededAnomaly]) -> float:
    """Fraction of queue entries that match a seeded anomaly.

    Returns 0.0 when the queue is empty — the playbook has no triage
    work to evaluate, which doesn't satisfy the spec's "produces a
    useful triage queue" contract.
    """
    if not findings:
        return 0.0
    tp = sum(1 for finding in findings if is_true_positive(finding, seeded))
    return tp / len(findings)


def _build_store(case: AnomalyEvalCase) -> InMemoryStore:
    store = InMemoryStore()
    if case.sections:
        store.add_sections(
            [
                Section(
                    id=seed.section_id,
                    parent_id=None,
                    title=seed.title,
                    summary=seed.summary,
                    chunk_ids=[],
                )
                for seed in case.sections
            ]
        )
    if case.chunks:
        store.add_chunks(
            [
                Chunk(
                    id=seed.chunk_id,
                    section_id=seed.section_id,
                    text=seed.text,
                    token_count=max(1, len(seed.text) // 4),
                    char_start=0,
                    char_end=len(seed.text),
                    embedding_id=f"emb-{seed.chunk_id}",
                )
                for seed in case.chunks
            ]
        )
    return store


class AnomalyEvalRunner:
    """Drive `AnomalyScanPlaybook` per case, score triage precision."""

    def __init__(self, *, detectors: list[Detector]) -> None:
        self._detectors = list(detectors)

    def run_case(self, case: AnomalyEvalCase) -> EvalResult:
        store = _build_store(case)
        queue = AnomalyScanPlaybook(detectors=self._detectors).run(store=store)
        precision = triage_precision(queue.findings, case.seeded_anomalies)
        return EvalResult(
            case_id=case.id,
            passed=precision >= TRIAGE_PRECISION_THRESHOLD,
            score=precision,
            metrics={"triage_precision": precision},
            notes=(
                f"queue={len(queue.findings)}, seeded={len(case.seeded_anomalies)}, "
                f"precision={precision:.3f}"
            ),
        )


__all__ = [
    "TRIAGE_PRECISION_THRESHOLD",
    "AnomalyEvalCase",
    "AnomalyEvalRunner",
    "ChunkSeed",
    "SectionSeed",
    "SeededAnomaly",
    "is_true_positive",
    "triage_precision",
]
