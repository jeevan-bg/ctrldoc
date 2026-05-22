"""Contract tests for the semantic chunker.

The chunker turns parsed sections into `Chunk` records that the
embedder, vector index, BM25 index, and storage layer consume. It
must obey the SPEC §4.1 invariants:
  - leaf chunks ≤ 512 tokens
  - never split mid-sentence (or mid-function for code)
  - chunk ids are stable across re-ingests

SPEC-REF: §4.1 (ingest step 4)
"""

from __future__ import annotations

from ctrldoc.ingest.chunker import (
    DEFAULT_MAX_TOKENS,
    chunk_section,
    chunk_sections,
)
from ctrldoc.ingest.parser import ParsedSection
from ctrldoc.tokenizer import count_tokens


def _section(
    *,
    section_id: str = "sec/intro",
    text: str = "",
    char_start: int = 0,
    char_end: int | None = None,
    parent_id: str | None = None,
    title: str = "Intro",
) -> ParsedSection:
    return ParsedSection(
        id=section_id,
        parent_id=parent_id,
        title=title,
        text=text,
        char_start=char_start,
        char_end=char_end if char_end is not None else len(text),
    )


# --- defaults ---


def test_default_max_tokens_is_512() -> None:
    assert DEFAULT_MAX_TOKENS == 512


# --- single chunk path ---


def test_short_section_becomes_single_chunk() -> None:
    section = _section(text="Just one short paragraph. Two sentences only.")
    chunks = chunk_section(section)
    assert len(chunks) == 1
    assert "Just one short paragraph" in chunks[0].text
    assert chunks[0].section_id == "sec/intro"
    assert chunks[0].embedding_id == ""  # filled in by S-036


def test_empty_section_produces_no_chunks() -> None:
    assert chunk_section(_section(text="")) == []
    assert chunk_section(_section(text="   \n\n  ")) == []


# --- splitting ---


def test_long_section_splits_below_max_tokens() -> None:
    # Each sentence is ~10 tokens; build ~300 sentences so total >> 512.
    sentence = "This is one sentence in the body of the section. "
    text = sentence * 300
    section = _section(text=text, char_end=len(text))
    chunks = chunk_section(section, max_tokens=128)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 128
        assert chunk.text.strip() != ""


def test_split_never_breaks_mid_sentence() -> None:
    # Sentences are intentionally a uniform 8 tokens each; max=20 → exactly 2 sentences/chunk.
    sentence = "A sentence packed with a few words here. "
    text = sentence * 12
    section = _section(text=text, char_end=len(text))
    chunks = chunk_section(section, max_tokens=20)
    for chunk in chunks:
        stripped = chunk.text.strip()
        # Each non-final chunk must end at a sentence terminator.
        assert stripped.endswith((".", "?", "!")) or chunk is chunks[-1]


def test_oversized_single_sentence_emits_one_chunk_over_budget() -> None:
    # When even one sentence already exceeds the budget, the chunker must
    # still emit it (data preservation) and the chunk count_tokens may
    # exceed the cap. This matches the SPEC ("never split mid-sentence").
    sentence = "word " * 600  # ≈ 600 tokens, no terminator
    section = _section(text=sentence, char_end=len(sentence))
    chunks = chunk_section(section, max_tokens=128)
    assert len(chunks) == 1
    assert chunks[0].token_count > 128


# --- chunk ids ---


def test_chunk_ids_are_deterministic() -> None:
    section = _section(text="One. Two. Three.")
    a = [c.id for c in chunk_section(section)]
    b = [c.id for c in chunk_section(section)]
    assert a == b


def test_chunk_ids_differ_when_text_differs() -> None:
    a = chunk_section(_section(text="hello world."))[0].id
    b = chunk_section(_section(text="goodbye world."))[0].id
    assert a != b


def test_chunk_ids_unique_within_section() -> None:
    sentence = "Unique sentence number {i}. "
    text = "".join(sentence.format(i=i) for i in range(20))
    chunks = chunk_section(_section(text=text), max_tokens=20)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# --- char ranges ---


def test_chunk_char_ranges_lie_within_section() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 20
    section = _section(text=text, char_start=100, char_end=100 + len(text))
    chunks = chunk_section(section, max_tokens=40)
    for chunk in chunks:
        assert 100 <= chunk.char_start <= chunk.char_end <= 100 + len(text)


def test_chunk_token_count_matches_tokenizer() -> None:
    section = _section(text="A few tokens here.")
    chunk = chunk_section(section)[0]
    assert chunk.token_count == count_tokens(chunk.text)


# --- multi-section dispatcher ---


def test_chunk_sections_returns_chunks_and_updated_sections() -> None:
    sections = [
        _section(section_id="sec/a", text="First section body. Two sentences."),
        _section(section_id="sec/b", text="Second section body."),
    ]
    chunks, updated_sections = chunk_sections(sections)
    assert {c.section_id for c in chunks} == {"sec/a", "sec/b"}
    by_id = {s.id: s for s in updated_sections}
    chunk_ids_for_a = [c.id for c in chunks if c.section_id == "sec/a"]
    assert list(by_id["sec/a"].chunk_ids) == chunk_ids_for_a


def test_chunk_sections_preserves_input_order_in_chunks() -> None:
    sections = [
        _section(section_id=f"sec/{name}", text=f"{name} body. extra sentence.")
        for name in ("a", "b", "c", "d")
    ]
    chunks, _ = chunk_sections(sections)
    seen: list[str] = []
    for c in chunks:
        if c.section_id not in seen:
            seen.append(c.section_id)
    assert seen == ["sec/a", "sec/b", "sec/c", "sec/d"]
