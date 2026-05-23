"""UC1 trustworthy QA — end-to-end with deterministic stubs.

`QAPlaybook` composes retrieve → generate → decompose → verify.
This example wires every dependency with a stub that doesn't touch
the network, so you can run it from a fresh checkout and inspect
the pipeline shape. The production wiring swaps the stubs for the
Anthropic backends.

Run:

    python examples/01_qa.py

SPEC-REF: §5.1
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import Claim, EvidencePack, Span
from ctrldoc.orch.task import StatelessTaskRunner
from ctrldoc.playbooks.qa import QAPlaybook


@dataclass
class _StubRetriever:
    pack: EvidencePack

    def retrieve(self, query: str) -> EvidencePack:
        return self.pack


@dataclass
class _StubClient:
    response: str

    def call(self, *, system: str, user: str) -> str:
        return self.response


@dataclass
class _StubDecomposer:
    claims_by_answer: dict[str, list[str]]

    def decompose(self, text: str) -> list[str]:
        return list(self.claims_by_answer.get(text, []))


@dataclass
class _StubVerifier:
    verdicts: dict[str, Claim]

    def verify(self, claim_text: str) -> Claim:
        return self.verdicts[claim_text]


def main() -> None:
    prefix = CacheablePrefix(
        system_prompt="You are a careful QA writer.",
        doc_skeleton="# §1 Aurora\n\nIntroduces consistent hashing.",
        entity_glossary="- **aurora** [system]",
    )
    pack = EvidencePack(
        query="anything",
        spans=[
            Span(
                chunk_id="c1",
                char_start=0,
                char_end=44,
                text="Aurora uses consistent hashing across nodes.",
            )
        ],
        token_count=20,
        retrieval_plan=["search(query, view=dense, k=8)"],
    )

    answer = "Aurora uses consistent hashing."
    playbook = QAPlaybook(
        prefix=prefix,
        retriever=_StubRetriever(pack=pack),
        task_runner=StatelessTaskRunner(
            client=_StubClient(response=json.dumps({"answer": answer})),
        ),
        decomposer=_StubDecomposer(claims_by_answer={answer: [answer]}),
        verifier=_StubVerifier(
            verdicts={
                answer: Claim(
                    text=answer,
                    citations=[
                        Span(
                            chunk_id="c1",
                            char_start=0,
                            char_end=44,
                            text="Aurora uses consistent hashing across nodes.",
                        )
                    ],
                    verified=True,
                    confidence=0.9,
                    nli_score=0.95,
                    judge_score=0.9,
                ),
            }
        ),
    )

    report = playbook.run("Does Aurora use consistent hashing?")
    print(
        json.dumps(
            {
                "query": report.query,
                "answer": report.answer,
                "claims": [
                    {
                        "text": c.text,
                        "verified": c.verified,
                        "confidence": c.confidence,
                        "citations": [span.chunk_id for span in c.citations],
                    }
                    for c in report.claims
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
