"""UC5 anomaly scan — deterministic detector battery.

The two reference detectors are fully deterministic — no LLM, no
stubs. They flag hedge words in chunk text and sections whose
summaries are empty. The four §5.5 LLM-backed detectors plug into
the same `Detector` protocol when their backends land.

Run:

    python examples/05_anomaly_scan.py

SPEC-REF: §5.5
"""

from __future__ import annotations

import json

from ctrldoc.models import Chunk, Section
from ctrldoc.ops.scan import (
    AnomalyScanPlaybook,
    EmptySummaryDetector,
    HedgeWordDetector,
)
from ctrldoc.store.memory import InMemoryStore


def main() -> None:
    store = InMemoryStore()
    store.add_sections(
        [
            Section(
                id="sec/perf",
                parent_id=None,
                title="Performance",
                summary="Aurora's perf targets.",
                chunk_ids=[],
            ),
            Section(
                id="sec/sec",
                parent_id=None,
                title="Security",
                summary="",  # blank summary → flagged by EmptySummaryDetector
                chunk_ids=[],
            ),
        ]
    )
    store.add_chunks(
        [
            Chunk(
                id="c-1",
                section_id="sec/perf",
                text="Operations should retry on transient failure.",
                token_count=8,
                char_start=0,
                char_end=44,
                embedding_id="emb-c-1",
            ),
            Chunk(
                id="c-2",
                section_id="sec/perf",
                text="A replica may diverge during partition events.",
                token_count=8,
                char_start=0,
                char_end=46,
                embedding_id="emb-c-2",
            ),
            Chunk(
                id="c-3",
                section_id="sec/perf",
                text="Writes are committed atomically within a partition.",
                token_count=8,
                char_start=0,
                char_end=51,
                embedding_id="emb-c-3",
            ),
        ]
    )

    queue = AnomalyScanPlaybook(
        detectors=[HedgeWordDetector(), EmptySummaryDetector()],
    ).run(store=store)

    print(
        json.dumps(
            {
                "findings": [
                    {
                        "detector": f.ctrldoc,
                        "severity": f.severity,
                        "claim": f.claim,
                        "chunk_id": f.location.chunk_id,
                        "text": f.location.text,
                    }
                    for f in queue.findings
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
