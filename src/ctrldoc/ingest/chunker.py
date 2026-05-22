"""Semantic chunker.

Turns parsed sections into leaf `Chunk` records that the rest of the
pipeline indexes. Each chunk is ≤ `max_tokens` measured by the
project tokenizer, never splits mid-sentence, and gets a stable
content-hash id so a re-ingest produces byte-identical chunk rows.

When a single sentence is already larger than `max_tokens`, the
chunker emits it anyway and lets the chunk exceed the budget — the
alternative (truncating) would lose information silently. Callers
that care can read `Chunk.token_count` and decide.

SPEC-REF: §4.1 (ingest step 4 — semantic chunking)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from ctrldoc.ingest.parser import ParsedSection
from ctrldoc.models import Chunk, Section
from ctrldoc.tokenizer import count_tokens
from ctrldoc.versioning import hash_chunk

DEFAULT_MAX_TOKENS: Final[int] = 512

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_section(
    section: ParsedSection,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    """Split a single parsed section into one or more chunks."""
    text = section.text.strip()
    if not text:
        return []

    atoms = _split_into_atoms(text)
    if not atoms:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        chunk_text = " ".join(buffer).strip()
        if chunk_text:
            chunks.append(_make_chunk(section, chunk_text, len(chunks)))
        buffer = []
        buffer_tokens = 0

    for atom in atoms:
        atom_tokens = count_tokens(atom)
        if atom_tokens > max_tokens:
            # Oversized atom: flush whatever's buffered, then emit it alone.
            flush()
            chunks.append(_make_chunk(section, atom, len(chunks)))
            continue
        if buffer and buffer_tokens + atom_tokens > max_tokens:
            flush()
        buffer.append(atom)
        buffer_tokens += atom_tokens

    flush()
    return chunks


def chunk_sections(
    sections: Iterable[ParsedSection],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[list[Chunk], list[Section]]:
    """Chunk every parsed section and return chunks plus the updated `Section`s.

    Each returned `Section` has its `chunk_ids` populated with the leaf ids in
    chunk order. The structural fields (`parent_id`, `title`, etc.) come
    straight from the parsed input; `summary` starts empty and will be filled
    in by the summariser slice (S-037).
    """
    all_chunks: list[Chunk] = []
    updated_sections: list[Section] = []
    for parsed in sections:
        section_chunks = chunk_section(parsed, max_tokens=max_tokens)
        all_chunks.extend(section_chunks)
        updated_sections.append(
            Section(
                id=parsed.id,
                parent_id=parsed.parent_id,
                title=parsed.title,
                summary="",
                chunk_ids=[c.id for c in section_chunks],
            )
        )
    return all_chunks, updated_sections


def _split_into_atoms(text: str) -> list[str]:
    """Split `text` into sentence-sized units.

    Paragraph breaks (`\\n\\n`) become hard boundaries; within a paragraph,
    we split on terminator-then-whitespace. Whitespace-only fragments are
    discarded.
    """
    atoms: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                atoms.append(sentence)
    return atoms


def _make_chunk(section: ParsedSection, text: str, ordinal: int) -> Chunk:
    char_start, char_end = _locate(section, text)
    chunk = Chunk(
        id="pending",  # replaced below; needed because hash_chunk needs a fully-formed instance
        section_id=section.id,
        text=text,
        token_count=count_tokens(text),
        char_start=char_start,
        char_end=char_end,
        embedding_id="",
        metadata={"ordinal": ordinal},
    )
    chunk_id = hash_chunk(chunk)
    return chunk.model_copy(update={"id": chunk_id})


def _locate(section: ParsedSection, text: str) -> tuple[int, int]:
    """Best-effort lookup of `text` inside the section body.

    Falls back to the full section range if not found verbatim (e.g. after
    whitespace collapse).
    """
    idx = section.text.find(text)
    if idx == -1:
        return section.char_start, section.char_end
    return section.char_start + idx, section.char_start + idx + len(text)


__all__ = ["DEFAULT_MAX_TOKENS", "chunk_section", "chunk_sections"]
