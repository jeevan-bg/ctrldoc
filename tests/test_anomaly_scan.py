"""UC5 `anomaly_scan` playbook — detectors fan-in, ranked queue.

Per §5.5 the playbook runs a set of `Detector`s against the index
and aggregates their findings into one ranked `AnomalyQueue`.
Detectors are deterministic or small-LLM passes; the module ships
the protocol, two reference detectors (hedge-word regex,
empty-section-summary), and the composing playbook with severity
ranking. Each detector advertises its `name`, which is what lands
on `Finding.ctrldoc` so a triage UI can group by source.

SPEC-REF: §5.5 (UC5 anomaly_scan)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from ctrldoc.models import Chunk, Finding, Section, Span
from ctrldoc.ops.scan import (
    AnomalyQueue,
    AnomalyScanPlaybook,
    Detector,
    EmptySummaryDetector,
    HedgeWordDetector,
)
from ctrldoc.store.memory import InMemoryStore

# --- store fixture ---


def _store_with(chunks: list[Chunk], sections: list[Section] | None = None) -> InMemoryStore:
    store = InMemoryStore()
    if sections:
        store.add_sections(sections)
    if chunks:
        store.add_chunks(chunks)
    return store


def _chunk(chunk_id: str, text: str, section_id: str = "sec/1") -> Chunk:
    return Chunk(
        id=chunk_id,
        section_id=section_id,
        text=text,
        token_count=10,
        char_start=0,
        char_end=len(text),
        embedding_id=f"emb-{chunk_id}",
    )


def _section(section_id: str, title: str, summary: str = "summary") -> Section:
    return Section(id=section_id, parent_id=None, title=title, summary=summary, chunk_ids=[])


# --- HedgeWordDetector ---


def test_hedge_word_detector_flags_hedge_in_chunks() -> None:
    store = _store_with([_chunk("c1", "Operations should always succeed under load.")])
    findings = HedgeWordDetector().detect(store=store)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.ctrldoc == "hedge_word"
    assert finding.severity == "warn"
    assert finding.location.chunk_id == "c1"
    # The span text equals exactly the matched hedge word.
    assert finding.location.text.lower() == "should"


def test_hedge_word_detector_finds_multiple_in_one_chunk() -> None:
    text = "This usually works but may fail; we typically retry."
    store = _store_with([_chunk("c1", text)])
    findings = HedgeWordDetector().detect(store=store)
    matched = sorted(f.location.text.lower() for f in findings)
    assert matched == ["may", "typically", "usually"]


def test_hedge_word_detector_silent_on_clean_text() -> None:
    store = _store_with([_chunk("c1", "The system commits writes atomically.")])
    assert HedgeWordDetector().detect(store=store) == []


def test_hedge_word_detector_char_offsets_match_chunk_text() -> None:
    text = "Operations should retry on failure."
    store = _store_with([_chunk("c1", text)])
    findings = HedgeWordDetector().detect(store=store)
    span = findings[0].location
    assert text[span.char_start : span.char_end].lower() == "should"


def test_hedge_word_detector_name_attribute() -> None:
    """The protocol requires a `name` so the playbook can map detector → ctrldoc."""
    assert HedgeWordDetector().name == "hedge_word"


def test_hedge_word_detector_satisfies_detector_protocol() -> None:
    assert isinstance(HedgeWordDetector(), Detector)


# --- EmptySummaryDetector ---


def test_empty_summary_detector_flags_blank_summary_sections() -> None:
    sections = [
        _section("s/full", "Full section", summary="non-empty"),
        _section("s/blank", "Blank section", summary="   "),
        _section("s/empty", "Empty section", summary=""),
    ]
    store = _store_with(chunks=[], sections=sections)
    findings = EmptySummaryDetector().detect(store=store)
    ids = sorted(f.location.chunk_id for f in findings)
    # Section id is recorded in Span.chunk_id (the closest analogue
    # we have for "where this finding points"); both blanks surface.
    assert ids == ["s/blank", "s/empty"]
    assert all(f.severity == "info" for f in findings)
    assert all(f.ctrldoc == "empty_summary" for f in findings)


def test_empty_summary_detector_silent_when_all_sections_summarised() -> None:
    sections = [
        _section("s/a", "A", summary="real"),
        _section("s/b", "B", summary="also real"),
    ]
    store = _store_with(chunks=[], sections=sections)
    assert EmptySummaryDetector().detect(store=store) == []


def test_empty_summary_detector_name() -> None:
    assert EmptySummaryDetector().name == "empty_summary"


# --- AnomalyScanPlaybook composition ---


def test_playbook_runs_every_detector_and_aggregates_findings() -> None:
    store = _store_with(
        chunks=[_chunk("c1", "Operations should retry.")],
        sections=[_section("s/1", "Sec", summary="")],
    )
    playbook = AnomalyScanPlaybook(detectors=[HedgeWordDetector(), EmptySummaryDetector()])
    queue = playbook.run(store=store)
    sources = sorted(f.ctrldoc for f in queue.findings)
    assert sources == ["empty_summary", "hedge_word"]


def test_playbook_ranks_findings_by_severity_descending() -> None:
    """critical > warn > info; within a severity, source order from the
    detector list breaks ties so the queue is fully deterministic."""

    def _finding(severity: str, ctrldoc: str) -> Finding:
        return Finding(
            ctrldoc=ctrldoc,
            location=Span(chunk_id="x", char_start=0, char_end=1, text="x"),
            claim=f"{ctrldoc} found",
            severity=severity,  # type: ignore[arg-type]
        )

    @dataclass
    class _FixedDetector:
        name: str
        out: list[Finding]
        calls: list[InMemoryStore] = field(default_factory=list)

        def detect(self, *, store: InMemoryStore) -> list[Finding]:
            self.calls.append(store)
            return list(self.out)

    detectors = [
        _FixedDetector(name="d1", out=[_finding("info", "d1"), _finding("warn", "d1")]),
        _FixedDetector(name="d2", out=[_finding("critical", "d2"), _finding("info", "d2")]),
    ]
    playbook = AnomalyScanPlaybook(detectors=detectors)
    queue = playbook.run(store=_store_with(chunks=[]))
    severities = [f.severity for f in queue.findings]
    # critical first, then both warn, then both info.
    assert severities == ["critical", "warn", "info", "info"]


def test_playbook_empty_detector_list_returns_empty_queue() -> None:
    playbook = AnomalyScanPlaybook(detectors=[])
    queue = playbook.run(store=_store_with(chunks=[]))
    assert queue.findings == []


def test_playbook_each_detector_called_with_store_once() -> None:
    @dataclass
    class _CountingDetector:
        name: str = "counting"
        calls: int = 0

        def detect(self, *, store: InMemoryStore) -> list[Finding]:
            self.calls += 1
            return []

    detectors = [_CountingDetector(name="a"), _CountingDetector(name="b")]
    AnomalyScanPlaybook(detectors=detectors).run(store=_store_with(chunks=[]))
    assert detectors[0].calls == 1
    assert detectors[1].calls == 1


def test_playbook_failing_detector_propagates_exception() -> None:
    @dataclass
    class _BoomDetector:
        name: str = "boom"

        def detect(self, *, store: InMemoryStore) -> list[Finding]:
            raise RuntimeError("detector blew up")

    with pytest.raises(RuntimeError, match="detector blew up"):
        AnomalyScanPlaybook(detectors=[_BoomDetector()]).run(store=_store_with(chunks=[]))


# --- AnomalyQueue model ---


def test_anomaly_queue_is_frozen() -> None:
    queue = AnomalyQueue(findings=[])
    with pytest.raises(ValidationError):
        queue.findings = []  # type: ignore[misc]
