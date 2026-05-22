"""Contract tests for the doc-skeleton and entity-glossary assembler.

The skeleton and glossary together form the cacheable prefix that
rides on every sub-task per SPEC §3.1 pillar 2. Output must be
deterministic — byte-identical given identical inputs — or the
Anthropic prompt cache stops hitting.

SPEC-REF: §4.2 (skeleton + glossary), §3.1 (cacheable prefix), §4.1
"""

from __future__ import annotations

from ctrldoc.assembler import (
    CacheablePrefix,
    assemble_cacheable_prefix,
    assemble_glossary,
    assemble_skeleton,
)
from ctrldoc.models import Entity, Section
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.tokenizer import count_tokens


def _section(
    section_id: str,
    *,
    parent_id: str | None = None,
    title: str = "",
    summary: str = "",
    chunk_ids: list[str] | None = None,
) -> Section:
    return Section(
        id=section_id,
        parent_id=parent_id,
        title=title or f"Section {section_id}",
        summary=summary or f"Summary for {section_id}.",
        chunk_ids=chunk_ids or [],
    )


def _store_with_tree() -> InMemoryStore:
    store = InMemoryStore()
    store.add_sections(
        [
            _section("s1", title="Introduction", summary="Intro paragraph."),
            _section("s1.1", parent_id="s1", title="Background", summary="Bg sentence."),
            _section("s1.2", parent_id="s1", title="Motivation", summary="Motiv sentence."),
            _section("s2", title="Architecture", summary="Arch overview."),
            _section("s2.1", parent_id="s2", title="Layer L0", summary="L0 summary."),
        ]
    )
    return store


# --- skeleton ---


def test_skeleton_empty_store_is_empty() -> None:
    assert assemble_skeleton(InMemoryStore()) == ""


def test_skeleton_renders_tree_in_dfs_order() -> None:
    out = assemble_skeleton(_store_with_tree())
    # The five titles must appear in DFS-pre-order:
    expected_order = [
        "Introduction",
        "Background",
        "Motivation",
        "Architecture",
        "Layer L0",
    ]
    positions = [out.find(title) for title in expected_order]
    assert all(p >= 0 for p in positions), positions
    assert positions == sorted(positions)


def test_skeleton_uses_heading_depth_for_nesting() -> None:
    out = assemble_skeleton(_store_with_tree())
    # Roots get one `#`, level-1 children get `##`.
    assert "# Introduction" in out
    assert "## Background" in out
    assert "## Motivation" in out
    assert "# Architecture" in out
    assert "## Layer L0" in out


def test_skeleton_includes_summary_text() -> None:
    out = assemble_skeleton(_store_with_tree())
    for summary in ("Intro paragraph.", "Bg sentence.", "L0 summary."):
        assert summary in out


def test_skeleton_is_deterministic() -> None:
    store = _store_with_tree()
    assert assemble_skeleton(store) == assemble_skeleton(store)


def test_skeleton_ignores_orphan_with_unknown_parent() -> None:
    store = InMemoryStore()
    store.add_sections(
        [
            _section("s1", title="Root"),
            _section("orphan", parent_id="nonexistent", title="Orphan"),
        ]
    )
    out = assemble_skeleton(store)
    assert "Root" in out
    assert "Orphan" not in out


# --- glossary ---


def test_glossary_empty_store_is_empty() -> None:
    assert assemble_glossary(InMemoryStore()) == ""


def test_glossary_sorted_by_id() -> None:
    store = InMemoryStore()
    store.add_entities(
        [
            Entity(id="ent-z", aliases=["Z"], type="concept", mention_chunk_ids=[]),
            Entity(id="ent-a", aliases=["A"], type="person", mention_chunk_ids=[]),
            Entity(id="ent-m", aliases=["M"], type="system", mention_chunk_ids=[]),
        ]
    )
    out = assemble_glossary(store)
    assert out.index("ent-a") < out.index("ent-m") < out.index("ent-z")


def test_glossary_renders_id_type_and_aliases() -> None:
    store = InMemoryStore()
    store.add_entities(
        [
            Entity(
                id="ent-claude",
                aliases=["Claude", "the assistant"],
                type="system",
                mention_chunk_ids=["c1"],
            ),
        ]
    )
    out = assemble_glossary(store)
    assert "ent-claude" in out
    assert "system" in out
    assert "Claude" in out
    assert "the assistant" in out


def test_glossary_handles_entity_without_aliases() -> None:
    store = InMemoryStore()
    store.add_entities([Entity(id="ent-1", aliases=[], type="concept", mention_chunk_ids=[])])
    out = assemble_glossary(store)
    assert "ent-1" in out
    assert "concept" in out


def test_glossary_is_deterministic() -> None:
    store = InMemoryStore()
    store.add_entities(
        [
            Entity(id="ent-b", aliases=["B"], type="concept", mention_chunk_ids=[]),
            Entity(id="ent-a", aliases=["A"], type="concept", mention_chunk_ids=[]),
        ]
    )
    assert assemble_glossary(store) == assemble_glossary(store)


# --- cacheable prefix ---


def test_cacheable_prefix_packs_three_parts() -> None:
    store = _store_with_tree()
    prefix = assemble_cacheable_prefix(store, system_prompt="You are a careful judge.")
    assert isinstance(prefix, CacheablePrefix)
    assert prefix.system_prompt == "You are a careful judge."
    assert "Introduction" in prefix.doc_skeleton
    assert prefix.entity_glossary == ""  # tree-only store has no entities


def test_cacheable_prefix_render_concatenates_deterministically() -> None:
    store = _store_with_tree()
    p1 = assemble_cacheable_prefix(store, system_prompt="sys")
    p2 = assemble_cacheable_prefix(store, system_prompt="sys")
    assert p1.render() == p2.render()


def test_cacheable_prefix_token_count_under_spec_budget() -> None:
    """SPEC §4.2: skeleton + glossary should be a few thousand tokens for real docs."""
    store = _store_with_tree()
    prefix = assemble_cacheable_prefix(store, system_prompt="x")
    # Synthetic tree is tiny; comfortably under the loose 5k budget.
    assert count_tokens(prefix.render()) < 5000


def test_cacheable_prefix_is_frozen() -> None:
    import pytest
    from pydantic import ValidationError

    prefix = assemble_cacheable_prefix(InMemoryStore(), system_prompt="x")
    with pytest.raises(ValidationError):
        prefix.system_prompt = "tampered"  # type: ignore[misc]
