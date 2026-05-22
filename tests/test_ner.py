"""Contract tests for the NER tagger and canonicalisation pass.

The NER tagger turns raw text into a list of `EntityMention`s
(surface form, label, char range). The canonicalisation pass folds
mentions into `Entity` records: one per canonical id, with the
distinct surface forms collected as aliases and the source chunk ids
attached.

SPEC-REF: §4.1 (ingest step 3 — NER + canonicalisation)
"""

from __future__ import annotations

import pytest

from ctrldoc.ingest.ner import (
    EntityMention,
    NERTagger,
    StubNERTagger,
    canonicalize,
)


def test_stub_tagger_satisfies_protocol() -> None:
    assert isinstance(StubNERTagger({}), NERTagger)


def test_stub_tagger_returns_configured_mentions() -> None:
    mentions = [EntityMention(text="Claude", label="person", start=0, end=6, score=0.9)]
    tagger = StubNERTagger({"hello": mentions})
    assert tagger.tag("hello", labels=["person"]) == mentions
    assert tagger.tag("other", labels=["person"]) == []


# --- EntityMention ---


def test_entity_mention_frozen() -> None:
    m = EntityMention(text="x", label="y", start=0, end=1, score=1.0)
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        m.text = "tampered"  # type: ignore[misc]


def test_entity_mention_score_clamped_to_unit_interval() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EntityMention(text="x", label="y", start=0, end=1, score=1.5)
    with pytest.raises(ValidationError):
        EntityMention(text="x", label="y", start=0, end=1, score=-0.1)


def test_entity_mention_end_must_not_precede_start() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EntityMention(text="x", label="y", start=10, end=5, score=0.5)


# --- canonicalize ---


def _mention(text: str, label: str = "person", *, start: int = 0, end: int = 1) -> EntityMention:
    return EntityMention(text=text, label=label, start=start, end=end, score=0.9)


def test_canonicalize_empty_input_returns_no_entities() -> None:
    assert canonicalize([], chunk_id="c1") == []


def test_canonicalize_single_mention_produces_one_entity() -> None:
    entities = canonicalize([_mention("Claude", "person")], chunk_id="c1")
    assert len(entities) == 1
    assert entities[0].aliases == ["Claude"]
    assert entities[0].type == "person"
    assert entities[0].mention_chunk_ids == ["c1"]


def test_canonicalize_groups_case_insensitive_surface_forms() -> None:
    mentions = [
        _mention("Claude", "person"),
        _mention("CLAUDE", "person"),
        _mention("claude", "person"),
    ]
    entities = canonicalize(mentions, chunk_id="c1")
    assert len(entities) == 1
    assert set(entities[0].aliases) == {"Claude", "CLAUDE", "claude"}


def test_canonicalize_separates_different_labels() -> None:
    mentions = [
        _mention("Claude", "person"),
        _mention("Claude", "system"),
    ]
    entities = canonicalize(mentions, chunk_id="c1")
    assert len(entities) == 2
    labels = {e.type for e in entities}
    assert labels == {"person", "system"}


def test_canonicalize_separates_different_surface_forms() -> None:
    mentions = [
        _mention("Claude", "person"),
        _mention("Sam Altman", "person"),
    ]
    entities = canonicalize(mentions, chunk_id="c1")
    ids = {e.id for e in entities}
    assert len(ids) == 2


def test_canonicalize_deterministic_id_format() -> None:
    entities = canonicalize([_mention("Sam Altman", "person")], chunk_id="c1")
    assert entities[0].id == "ent/person/sam-altman"


def test_canonicalize_emits_entities_sorted_by_id() -> None:
    mentions = [
        _mention("Zeta", "person"),
        _mention("Alpha", "person"),
        _mention("Mu", "person"),
    ]
    entities = canonicalize(mentions, chunk_id="c1")
    assert [e.id for e in entities] == sorted(e.id for e in entities)


def test_canonicalize_collects_mention_chunk_ids() -> None:
    # Same entity surfaces in two chunks; mention_chunk_ids reflects both.
    by_chunk = {
        "c1": [_mention("Claude", "person")],
        "c2": [_mention("claude", "person")],
    }
    seen: dict[str, list[str]] = {}
    for chunk_id, mentions in by_chunk.items():
        for ent in canonicalize(mentions, chunk_id=chunk_id):
            seen.setdefault(ent.id, []).extend(ent.mention_chunk_ids)
    # Build a merged view as the caller would.
    assert set(next(iter(seen.values()))) == {"c1", "c2"}
