"""Integration tests for the GLiNER-backed NER tagger.

These tests download a small zero-shot NER model on first run
(~166MB) and skip cleanly when GLiNER is not installed. The model
is cached after the first download, so subsequent runs are fast.

SPEC-REF: §4.1 (ingest step 3 — NER)
"""

from __future__ import annotations

import pytest

pytest.importorskip("gliner", reason="gliner is optional; install ctrldoc[ingest] to run")

from ctrldoc.ingest.ner import NERTagger
from ctrldoc.ingest.ner_gliner import GLiNERTagger


@pytest.fixture(scope="module")
def tagger() -> GLiNERTagger:
    return GLiNERTagger()


@pytest.mark.slow
def test_satisfies_protocol(tagger: GLiNERTagger) -> None:
    assert isinstance(tagger, NERTagger)


@pytest.mark.slow
def test_basic_predictions(tagger: GLiNERTagger) -> None:
    text = "Claude was created by Anthropic. Sam Altman runs OpenAI."
    mentions = tagger.tag(text, labels=["person", "organization"])
    found = {(m.text, m.label) for m in mentions}
    # We don't pin the exact set (model probabilities drift across versions)
    # but the obvious entities should land.
    assert ("Sam Altman", "person") in found
    assert ("OpenAI", "organization") in found


@pytest.mark.slow
def test_empty_text_returns_no_mentions(tagger: GLiNERTagger) -> None:
    assert tagger.tag("", labels=["person"]) == []


@pytest.mark.slow
def test_empty_label_list_returns_no_mentions(tagger: GLiNERTagger) -> None:
    assert tagger.tag("Claude was created by Anthropic.", labels=[]) == []


@pytest.mark.slow
def test_mention_offsets_match_source(tagger: GLiNERTagger) -> None:
    text = "Claude was created by Anthropic."
    mentions = tagger.tag(text, labels=["person", "organization"])
    for m in mentions:
        assert text[m.start : m.end] == m.text
