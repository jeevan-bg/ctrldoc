"""UC1 — citation-grounded QA playbook.

Pipeline:

  1. Retrieve an evidence pack for the query.
  2. Generate an answer via the stateless task runner — the model
     sees only the cacheable prefix and the rendered evidence text.
  3. Decompose the answer into atomic claims.
  4. Verify each claim independently against the index.
  5. Return an `AnswerReport` carrying the answer and per-claim
     verification verdicts.

The playbook is dependency-injected: any `QARetriever`,
`ClaimDecomposer`, and `ClaimVerifier` work, so swapping in a
production retriever or moving from heuristic to LLM-backed
verification is a constructor change.

SPEC-REF: §5.1 (UC1 qa playbook)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import Claim, EvidencePack
from ctrldoc.orch.task import StatelessTaskRunner, TaskInput
from ctrldoc.verify.claim_decomposer import ClaimDecomposer
from ctrldoc.verify.claim_verifier import ClaimVerifier


@runtime_checkable
class QARetriever(Protocol):
    """Anything that turns a query into an evidence pack."""

    def retrieve(self, query: str) -> EvidencePack: ...


class _GeneratedAnswer(BaseModel):
    """Raw generation output before claim verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str


class AnswerReport(BaseModel):
    """Result of one QA run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    answer: str
    claims: list[Claim]


class QAPlaybook:
    """Compose retrieval, generation, decomposition, verification."""

    def __init__(
        self,
        *,
        prefix: CacheablePrefix,
        retriever: QARetriever,
        task_runner: StatelessTaskRunner,
        decomposer: ClaimDecomposer,
        verifier: ClaimVerifier,
    ) -> None:
        self._prefix = prefix
        self._retriever = retriever
        self._task_runner = task_runner
        self._decomposer = decomposer
        self._verifier = verifier

    def run(self, query: str) -> AnswerReport:
        if not query.strip():
            return AnswerReport(query=query, answer="", claims=[])

        pack = self._retriever.retrieve(query)
        evidence_text = _render_evidence(pack)

        task = TaskInput(
            prefix=self._prefix,
            evidence_pack=evidence_text,
            task_input=query,
        )
        generated = self._task_runner.run(task, output_model=_GeneratedAnswer)

        claim_texts = self._decomposer.decompose(generated.answer)
        claims = [self._verifier.verify(text) for text in claim_texts]

        return AnswerReport(query=query, answer=generated.answer, claims=claims)


def _render_evidence(pack: EvidencePack) -> str:
    """Render an evidence pack as labelled spans for the generation prompt.

    Each span is prefixed with `[chunk_id]` so the model has a stable
    citation handle to reference. Order matches `pack.spans` so two
    identical retrievals produce identical prompts (cache stability).
    """
    if not pack.spans:
        return ""
    lines = [f"[{span.chunk_id}] {span.text}" for span in pack.spans]
    return "\n\n".join(lines)


__all__ = [
    "AnswerReport",
    "QAPlaybook",
    "QARetriever",
]
