"""Tier-1 deterministic claim-graph extractor — the §6.4 floor.

Four pattern families produce typed edges between concept clusters with
no LLM involved:

1. **Hearst patterns** — `X such as Y`, `X including Y`, `X is a Y` and
   their variants emit `example_of` and `is_a` edges from surface-form
   lexical cues (Hearst 1992).
2. **Heading tree** — a child `Section`'s title is `part_of` its parent's
   title; the structural containment is a free signal already encoded in
   the parsed document tree.
3. **Sliding-window PMI** — within a configurable token window, tokens
   that co-occur above a positive pointwise-mutual-information threshold
   are linked with `related_to`. This is the lightest-weight
   distributional signal we can compute over a single doc with no
   training corpus.
4. **Coref identity** — two mentions sharing a normalized surface form
   collapse into one `Tier1Concept`; the identity is recorded as a
   self-edge of type `equivalent_to` so the audit trail surfaces it
   explicitly. Real morphological coref is the Tier-2 backend's job
   (S-128 territory); the Tier-1 floor uses lexical identity only.

All edges carry the fixed heuristic prior `HEURISTIC_CONFIDENCE = 0.9`
from §6.5 and the `source="heuristic"` provenance tag. The extractor is
fully deterministic: identical input bytes produce identical output
bytes across runs.

SPEC-REF: §6.4 (schema co-induction floor), §6.5 (heuristic edge prior)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from ctrldoc.models import Chunk, Section, Span
from ctrldoc.versioning import content_hash

HEURISTIC_CONFIDENCE: float = 0.9
"""The §6.5 fixed prior every Tier-1 heuristic emits."""

PatternLiteral: TypeAlias = Literal[
    "hearst_such_as",
    "hearst_including",
    "hearst_is_a",
    "heading_tree",
    "pmi_window",
    "coref_identity",
]

EdgeTypeLiteral: TypeAlias = Literal[
    "is_a",
    "part_of",
    "example_of",
    "related_to",
    "equivalent_to",
]


# --- public dataclasses ------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Tier1Mention(_Strict):
    """A surface mention discovered by one of the four pattern families."""

    id: str
    text: str
    span: Span


class Tier1Concept(_Strict):
    """A canonical cluster of mentions sharing the same normalized surface form."""

    id: str
    canonical_name: str
    primitive_type: Literal["Entity"] = "Entity"
    mention_ids: list[str]


class Tier1Edge(_Strict):
    """A heuristic typed edge between concept clusters."""

    src_id: str
    dst_id: str
    type: EdgeTypeLiteral
    pattern: PatternLiteral
    confidence: float = Field(gt=0.0, le=1.0)
    source: Literal["heuristic"] = "heuristic"
    citations: list[Span]


class Tier1Extraction(_Strict):
    """Aggregate output of one extractor invocation."""

    mentions: list[Tier1Mention]
    concepts: list[Tier1Concept]
    edges: list[Tier1Edge]


@dataclass(frozen=True)
class Tier1Config:
    """Tunable knobs. Defaults are reasonable for spec/runbook prose."""

    pmi_window_tokens: int = 16
    pmi_min_count: int = 2
    pmi_threshold: float = 1.0


# --- regex library ----------------------------------------------------------

# Hearst patterns operate on lower-cased text. A noun-phrase here is
# a bounded run of 1-4 alpha-tokens (separated by single spaces); the
# bound keeps the hyponym phrase from greedily swallowing the predicate.
# Hyphens are admitted inside tokens (`warm-blooded`) but not as
# separators between tokens.
_NP_TOKEN = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
_NP_CHARS = rf"{_NP_TOKEN}(?:\s{_NP_TOKEN}){{0,3}}"
_HEARST_SUCH_AS = re.compile(
    rf"(?P<hyper>{_NP_CHARS})\s+such\s+as\s+(?P<hypo>{_NP_CHARS})(?=[\s,;.]|$)",
)
_HEARST_INCLUDING = re.compile(
    rf"(?P<hyper>{_NP_CHARS})\s+including\s+(?P<hypo>{_NP_CHARS})(?=[\s,;.]|$)",
)
# `X is a Y` / `X is an Y` — hyponym first, hypernym second.
_HEARST_IS_A = re.compile(
    rf"(?P<hypo>{_NP_CHARS})\s+is\s+an?\s+(?P<hyper>{_NP_CHARS})(?=[\s,;.]|$)",
)

# A token is a maximal run of alphanumerics. Single-character tokens
# (mostly "a") are excluded from PMI because they over-dominate the
# co-occurrence count without carrying conceptual signal.
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]+")

# Common closed-class words excluded from PMI cooccurrence.
_PMI_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "for",
        "with",
        "by",
        "from",
        "as",
        "at",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
        "his",
        "her",
        "have",
        "has",
        "had",
        "not",
        "no",
        "if",
        "then",
        "than",
        "so",
        "such",
        "via",
        "into",
        "onto",
        "they",
        "we",
        "you",
        "i",
        "all",
        "any",
        "each",
        "every",
        "some",
        "many",
        "few",
        "more",
        "most",
        "less",
        "least",
        "do",
        "does",
        "did",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "elsewhere",
        "live",
        "lives",
    }
)


# --- private helpers --------------------------------------------------------


# Verb-like tokens that signal a noun-phrase boundary inside a matched
# Hearst span. When present, the NP is truncated to the prefix before
# the first verb-like token.
_NP_TERMINATORS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "and",
        "or",
        "but",
        "while",
        "because",
        "since",
        "though",
        "although",
    }
)


def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip outer punctuation."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _trim_np(np: str) -> str:
    """Truncate a noun-phrase at the first verb-like or connective token.

    Hearst regex captures up to four tokens; many sentences continue
    with `<NP> are/and/or ...` after the noun. We split on whitespace
    and stop at the first terminator token. Result is the leading
    noun-phrase substring, lower-cased.
    """
    tokens = np.split()
    out: list[str] = []
    for tok in tokens:
        if tok in _NP_TERMINATORS:
            break
        out.append(tok)
    return " ".join(out)


def _mention_id(canonical: str, span: Span) -> str:
    """Content-hashed mention id stable across runs."""
    payload = f"{canonical}|{span.chunk_id}|{span.char_start}|{span.char_end}"
    return content_hash(payload)


def _concept_id(canonical: str) -> str:
    """Content-hashed concept id keyed on canonical surface form."""
    return content_hash(f"tier1-concept|{canonical}")


def _span_for_match(chunk: Chunk, start: int, end: int) -> Span:
    """Build a `Span` for `chunk.text[start:end]`."""
    return Span(
        chunk_id=chunk.id,
        char_start=start,
        char_end=end,
        text=chunk.text[start:end],
    )


def _heading_span(section: Section) -> Span:
    """A surrogate `Span` carrying the section title.

    Heading-tree edges have no chunk-anchored citation (the heading lives
    in the parsed tree, not the chunk text), so we synthesise a `Span`
    pinned to a section-derived pseudo chunk id. The pseudo id stays
    deterministic across runs.
    """
    return Span(
        chunk_id=f"heading:{section.id}",
        char_start=0,
        char_end=len(section.title),
        text=section.title,
    )


# --- public entry point -----------------------------------------------------


def extract_tier1(
    *,
    sections: Iterable[Section],
    chunks: Iterable[Chunk],
    config: Tier1Config | None = None,
) -> Tier1Extraction:
    """Run all four Tier-1 heuristics and return the deterministic union.

    Output ordering: concepts sorted by canonical name; mentions sorted
    by (concept id, chunk id, char_start); edges sorted by (pattern,
    src_id, dst_id). This makes diffs human-readable and the run
    bit-identical given identical input.
    """
    cfg = config or Tier1Config()
    chunks_list = list(chunks)
    sections_list = list(sections)

    # Bag of (canonical_name → list[Span]) accumulated across all heuristics.
    mention_spans: dict[str, list[Span]] = defaultdict(list)

    # Edges as plain tuples while we collect, sorted + deduped before return.
    pending_edges: list[tuple[PatternLiteral, EdgeTypeLiteral, str, str, Span]] = []

    # 1. Hearst patterns over chunk text.
    hearst_rules: tuple[tuple[PatternLiteral, re.Pattern[str], EdgeTypeLiteral, str, str], ...] = (
        ("hearst_such_as", _HEARST_SUCH_AS, "example_of", "hypo", "hyper"),
        ("hearst_including", _HEARST_INCLUDING, "example_of", "hypo", "hyper"),
        ("hearst_is_a", _HEARST_IS_A, "is_a", "hypo", "hyper"),
    )
    for chunk in chunks_list:
        text = chunk.text
        lower = text.lower()
        for pattern_name, regex, edge_type, src_group, dst_group in hearst_rules:
            for m in regex.finditer(lower):
                hypo_raw = m.group("hypo")
                hyper_raw = m.group("hyper")
                hypo = _trim_np(_normalize(hypo_raw))
                hyper = _trim_np(_normalize(hyper_raw))
                if not hypo or not hyper or hypo == hyper:
                    continue
                hypo_span = _span_for_match(chunk, m.start("hypo"), m.start("hypo") + len(hypo))
                hyper_span = _span_for_match(chunk, m.start("hyper"), m.start("hyper") + len(hyper))
                mention_spans[hypo].append(hypo_span)
                mention_spans[hyper].append(hyper_span)
                citation_span = _span_for_match(chunk, m.start(), m.end())
                src = hypo if src_group == "hypo" else hyper
                dst = hyper if dst_group == "hyper" else hypo
                pending_edges.append((pattern_name, edge_type, src, dst, citation_span))

    # 2. Heading tree — child section is part_of parent section.
    by_id = {s.id: s for s in sections_list}
    for sec in sections_list:
        if sec.parent_id is None or sec.parent_id not in by_id:
            continue
        parent = by_id[sec.parent_id]
        child_name = _normalize(sec.title)
        parent_name = _normalize(parent.title)
        if not child_name or not parent_name or child_name == parent_name:
            continue
        child_span = _heading_span(sec)
        parent_span = _heading_span(parent)
        mention_spans[child_name].append(child_span)
        mention_spans[parent_name].append(parent_span)
        pending_edges.append(("heading_tree", "part_of", child_name, parent_name, child_span))

    # 3. Sliding-window PMI over the joined chunk corpus.
    chunk_tokens: list[list[tuple[str, int, int]]] = []
    unigram_counts: Counter[str] = Counter()
    cooc_counts: Counter[tuple[str, str]] = Counter()
    total_windows = 0
    for chunk in chunks_list:
        tokens: list[tuple[str, int, int]] = []
        for m in _TOKEN.finditer(chunk.text):
            tok = m.group(0).lower()
            if tok in _PMI_STOPWORDS or len(tok) < 2:
                continue
            tokens.append((tok, m.start(), m.end()))
        chunk_tokens.append(tokens)
        for tok, _, _ in tokens:
            unigram_counts[tok] += 1
        # Sliding-window co-occurrence (unordered pair, no self-pair).
        for i, (tok_i, _, _) in enumerate(tokens):
            window_end = min(i + cfg.pmi_window_tokens, len(tokens))
            for j in range(i + 1, window_end):
                tok_j = tokens[j][0]
                if tok_i == tok_j:
                    continue
                pair = (tok_i, tok_j) if tok_i < tok_j else (tok_j, tok_i)
                cooc_counts[pair] += 1
        total_windows += max(0, len(tokens) - 1)

    if total_windows > 0 and sum(unigram_counts.values()) > 0:
        total_tokens = sum(unigram_counts.values())
        for (a, b), c_ab in cooc_counts.items():
            if c_ab < cfg.pmi_min_count:
                continue
            p_ab = c_ab / total_windows
            p_a = unigram_counts[a] / total_tokens
            p_b = unigram_counts[b] / total_tokens
            denom = p_a * p_b
            if denom <= 0.0:
                continue
            pmi = math.log2(p_ab / denom)
            if pmi < cfg.pmi_threshold:
                continue
            # Citation = first chunk where the pair co-occurred.
            cite = _first_cooccurrence_span(chunk_tokens, chunks_list, a, b)
            if cite is None:
                continue
            mention_spans[a].append(cite)
            mention_spans[b].append(cite)
            pending_edges.append(("pmi_window", "related_to", a, b, cite))

    # 4. Coref identity — lexical-surface repetitions cluster.
    surface_spans: dict[str, list[Span]] = defaultdict(list)
    for chunk in chunks_list:
        for m in _TOKEN.finditer(chunk.text):
            tok = m.group(0)
            tok_lower = tok.lower()
            if tok_lower in _PMI_STOPWORDS or len(tok_lower) < 2:
                continue
            # Only multi-occurrence proper-noun-ish tokens (first-char upper).
            if not tok[0].isupper():
                continue
            surface_spans[tok_lower].append(_span_for_match(chunk, m.start(), m.end()))
    for canonical, spans in surface_spans.items():
        if len(spans) < 2:
            continue
        for sp in spans:
            mention_spans[canonical].append(sp)
        # One self-edge per identity cluster; citation = first mention.
        pending_edges.append(("coref_identity", "equivalent_to", canonical, canonical, spans[0]))

    # --- Build concept + mention rows from the accumulated mention_spans. ---
    # De-dupe identical spans per canonical name (a Hearst match registers
    # both endpoints; PMI registers the cooccurrence span on both tokens).
    concept_rows: list[Tier1Concept] = []
    mention_rows: list[Tier1Mention] = []
    canonical_to_concept_id: dict[str, str] = {}
    for canonical in sorted(mention_spans):
        seen: set[tuple[str, int, int]] = set()
        m_ids: list[str] = []
        for sp in mention_spans[canonical]:
            key = (sp.chunk_id, sp.char_start, sp.char_end)
            if key in seen:
                continue
            seen.add(key)
            mid = _mention_id(canonical, sp)
            mention_rows.append(Tier1Mention(id=mid, text=canonical, span=sp))
            m_ids.append(mid)
        cid = _concept_id(canonical)
        canonical_to_concept_id[canonical] = cid
        concept_rows.append(Tier1Concept(id=cid, canonical_name=canonical, mention_ids=m_ids))

    # --- Build edges (de-duplicated; same (pattern, src, dst) collapses). ---
    seen_edges: set[tuple[PatternLiteral, EdgeTypeLiteral, str, str]] = set()
    edge_rows: list[Tier1Edge] = []
    for pattern, edge_type, src_canon, dst_canon, citation in pending_edges:
        src_id = canonical_to_concept_id.get(src_canon)
        dst_id = canonical_to_concept_id.get(dst_canon)
        if src_id is None or dst_id is None:
            continue
        edge_key = (pattern, edge_type, src_id, dst_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_rows.append(
            Tier1Edge(
                src_id=src_id,
                dst_id=dst_id,
                type=edge_type,
                pattern=pattern,
                confidence=HEURISTIC_CONFIDENCE,
                citations=[citation],
            )
        )

    mention_rows.sort(key=lambda m: (m.text, m.span.chunk_id, m.span.char_start))
    edge_rows.sort(key=lambda e: (e.pattern, e.src_id, e.dst_id))

    return Tier1Extraction(
        mentions=mention_rows,
        concepts=concept_rows,
        edges=edge_rows,
    )


def _first_cooccurrence_span(
    chunk_tokens: list[list[tuple[str, int, int]]],
    chunks_list: list[Chunk],
    a: str,
    b: str,
) -> Span | None:
    """First chunk-anchored span covering one cooccurrence of `(a, b)`."""
    for chunk, tokens in zip(chunks_list, chunk_tokens, strict=False):
        positions_a = [i for i, (tok, _, _) in enumerate(tokens) if tok == a]
        positions_b = [i for i, (tok, _, _) in enumerate(tokens) if tok == b]
        if not positions_a or not positions_b:
            continue
        # Earliest pair sharing the same window — by construction one exists
        # in this chunk because the cooccurrence counter saw it here.
        best: tuple[int, int] | None = None
        for ai in positions_a:
            for bi in positions_b:
                lo, hi = (ai, bi) if ai < bi else (bi, ai)
                if best is None or lo < best[0] or (lo == best[0] and hi < best[1]):
                    best = (lo, hi)
        if best is None:
            continue
        lo_idx, hi_idx = best
        lo_start = tokens[lo_idx][1]
        hi_end = tokens[hi_idx][2]
        return _span_for_match(chunk, lo_start, hi_end)
    return None


__all__ = [
    "HEURISTIC_CONFIDENCE",
    "EdgeTypeLiteral",
    "PatternLiteral",
    "Tier1Concept",
    "Tier1Config",
    "Tier1Edge",
    "Tier1Extraction",
    "Tier1Mention",
    "extract_tier1",
]
