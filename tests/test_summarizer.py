"""Contract tests for the section summariser.

`Summarizer` is the protocol every backend satisfies. The heuristic
reference extracts the first 1 or 2 sentences with no network or model
dependency. `summarize_sections` drives a summariser across an
iterable of `Section`s and returns updated `Section` rows with
`summary` populated.

SPEC-REF: §4.1 (ingest step 7 — section summaries), §3.1 (skeleton)
"""

from __future__ import annotations

from ctrldoc.ingest.summarizer import (
    HeuristicSummarizer,
    Summarizer,
    summarize_sections,
)
from ctrldoc.models import Section


def _section(
    *,
    section_id: str = "sec/a",
    title: str = "Section",
    chunk_ids: list[str] | None = None,
) -> Section:
    return Section(
        id=section_id,
        parent_id=None,
        title=title,
        summary="",
        chunk_ids=chunk_ids or [],
    )


def test_satisfies_protocol() -> None:
    assert isinstance(HeuristicSummarizer(), Summarizer)


# --- heuristic summariser ---


def test_heuristic_empty_input_returns_empty() -> None:
    assert HeuristicSummarizer().summarize("") == ""
    assert HeuristicSummarizer().summarize("   \n\n   ") == ""


def test_heuristic_single_sentence_passes_through() -> None:
    assert HeuristicSummarizer().summarize("Just one sentence.") == "Just one sentence."


def test_heuristic_two_sentences_kept() -> None:
    text = "First sentence here. Second sentence here. Third sentence here."
    summary = HeuristicSummarizer().summarize(text)
    assert summary == "First sentence here. Second sentence here."


def test_heuristic_respects_max_sentences_argument() -> None:
    text = "One. Two. Three. Four."
    summary = HeuristicSummarizer(max_sentences=1).summarize(text)
    assert summary == "One."


def test_heuristic_strips_surrounding_whitespace() -> None:
    text = "   \nFirst.    Second.   \n"
    summary = HeuristicSummarizer().summarize(text)
    assert summary == "First. Second."


def test_heuristic_handles_text_without_terminator() -> None:
    text = "no terminator here"
    # A body without `.?!` returns the whole stripped text.
    assert HeuristicSummarizer().summarize(text) == "no terminator here"


def test_heuristic_is_deterministic() -> None:
    text = "Alpha. Beta. Gamma."
    s = HeuristicSummarizer()
    assert s.summarize(text) == s.summarize(text)


# --- summarize_sections driver ---


def test_summarize_sections_uses_resolver_output() -> None:
    class UpperSummarizer:
        def summarize(self, text: str) -> str:
            return text.upper()

    sections = [_section(section_id="sec/a"), _section(section_id="sec/b")]
    bodies = {"sec/a": "alpha alpha", "sec/b": "beta beta"}
    out = summarize_sections(
        sections,
        body_for=lambda s: bodies[s.id],
        summarizer=UpperSummarizer(),
    )
    by_id = {s.id: s for s in out}
    assert by_id["sec/a"].summary == "ALPHA ALPHA"
    assert by_id["sec/b"].summary == "BETA BETA"


def test_summarize_sections_preserves_structural_fields() -> None:
    original = _section(section_id="sec/x", title="X", chunk_ids=["c1", "c2"])
    out = summarize_sections(
        [original],
        body_for=lambda _s: "body text.",
        summarizer=HeuristicSummarizer(),
    )[0]
    assert out.id == original.id
    assert out.title == original.title
    assert out.parent_id == original.parent_id
    assert out.chunk_ids == original.chunk_ids


def test_summarize_sections_empty_body_yields_empty_summary() -> None:
    out = summarize_sections(
        [_section()],
        body_for=lambda _s: "",
        summarizer=HeuristicSummarizer(),
    )[0]
    assert out.summary == ""


def test_summarize_sections_preserves_iteration_order() -> None:
    sections = [_section(section_id=f"sec/{i}") for i in range(4)]
    out = summarize_sections(
        sections,
        body_for=lambda s: f"body for {s.id}.",
        summarizer=HeuristicSummarizer(),
    )
    assert [s.id for s in out] == [s.id for s in sections]
