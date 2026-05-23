"""UC4 analytical review — lens fan-out + synthesis.

`HeuristicLensGenerator` emits the canonical five §5.4 lenses
(assumptions / boundary_cases / consistency / ambiguity /
scope_gaps). Per-lens sweeps yield `Finding` records; the
`SynthesisRunner` reduces over the structured findings JSON via the
S-067 one-shot reduce primitive — never seeing the raw doc.

Run:

    python examples/04_analytical_review.py

SPEC-REF: §5.4
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ctrldoc.assembler import CacheablePrefix
from ctrldoc.models import Finding, Span
from ctrldoc.orch.synthesis import SynthesisRunner
from ctrldoc.playbooks.review import (
    AnalyticalReviewPlaybook,
    HeuristicLensGenerator,
    Lens,
)


@dataclass
class _ScriptedSweeper:
    """Returns one finding per lens, name-tagged."""

    def sweep(self, lens: Lens) -> list[Finding]:
        # One synthetic finding per lens so the synthesis call sees
        # a representative aggregate.
        return [
            Finding(
                ctrldoc=lens.name,
                location=Span(chunk_id=f"c-{lens.name}", char_start=0, char_end=8, text="evidence"),
                claim=f"Surface a {lens.name} concern in the source doc.",
                severity="warn",
            ),
        ]


@dataclass
class _SynthesisClient:
    response: str

    def call(self, *, system: str, user: str) -> str:
        return self.response


def main() -> None:
    narrative_payload = json.dumps(
        {
            "headline": "Five lenses, five concerns",
            "sections": ["Assumptions", "Boundary cases", "Consistency", "Ambiguity", "Scope gaps"],
            "summary": "Synthesised review covering each lens-surfaced finding.",
        }
    )
    playbook = AnalyticalReviewPlaybook(
        prefix=CacheablePrefix(
            system_prompt="You are an analytical reviewer.",
            doc_skeleton="# §1",
            entity_glossary="",
        ),
        lens_generator=HeuristicLensGenerator(),
        sweeper=_ScriptedSweeper(),
        synthesis_runner=SynthesisRunner(client=_SynthesisClient(response=narrative_payload)),
    )

    report = playbook.run("Aurora L0 kernel spec")
    print(
        json.dumps(
            {
                "doc_type": report.doc_type,
                "findings": [
                    {"lens": f.ctrldoc, "claim": f.claim, "severity": f.severity}
                    for f in report.findings
                ],
                "narrative": {
                    "headline": report.narrative.headline,
                    "sections": list(report.narrative.sections),
                    "summary": report.narrative.summary,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
