"""L0 schema co-induction — propose typed nodes/edges from a max-entropy sample.

This module is step 2 of the §6.4 EM-style schema co-induction algorithm:
given the document's parsed chunks plus their embeddings, draw a small
diverse sample, hand it to a single Tier-3 LLM call, and parse the
result into a `SchemaProposal`. The proposal is cached as YAML on disk
so sibling docs in the same workspace can reuse it without paying for
a second LLM call.

The three responsibilities are deliberately separated:

* `max_entropy_sample` — pure, deterministic farthest-point selection
  on a list of embedding vectors. No I/O, no LLM. Greedy seed = the
  vector furthest from the cloud centroid; each subsequent pick is the
  vector that maximises the minimum cosine distance to anything already
  picked. Ties break by chunk-ordinal for byte-stable reproducibility.

* `SchemaProposer.propose` — wires the sample through a single
  `TaskClient.call` whose system message pins the closed 10-primitive
  library from §6.4 and whose user message carries only the sampled
  excerpts (never the full doc; §13 non-negotiable 1 also forbids
  feeding the raw doc into the LLM). The returned JSON is parsed into
  a `SchemaProposal` and validated against the primitive alphabet.

* `dump_schema_yaml` / `load_schema_yaml` — deterministic YAML I/O on
  the proposal. The on-disk shape is the byte-stable cache the
  workspace round-trips. The serialiser writes a constrained subset
  (block-style mappings of scalars, no anchors, no aliases) so the
  output is reproducible across runs and platforms; the parser is the
  symmetric counterpart and accepts only that subset.

SPEC-REF: §6.4 (schema co-induction)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, get_args, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ctrldoc.ingest.embedder import Embedder
from ctrldoc.models import Chunk
from ctrldoc.models_v1 import PrimitiveTypeLiteral

PRIMITIVE_LIBRARY: tuple[str, ...] = get_args(PrimitiveTypeLiteral)
"""The closed 10-element atomic library, sourced from `PrimitiveTypeLiteral`."""

_DEFAULT_SAMPLE_K = 10
"""Mid-point of the 8-12 sample range specified in §6.4 step 2."""

_SYSTEM_PROMPT = (
    "You are a schema proposer for a multi-document analysis system.\n"
    "Given excerpts from one document, propose typed nodes and typed edges that "
    "best describe the document's domain.\n\n"
    "Constraints:\n"
    "- Every typed node must declare a primitive from this closed library:\n"
    "  Entity, Event, Process, Property, Quantity, Definition, Assertion, "
    "Obligation, Citation, Relation.\n"
    "- Every typed edge must reference typed node names you defined.\n"
    "- Use short PascalCase names (1-3 words).\n"
    "- Do not invent primitives outside the library.\n"
    "- Return only JSON of the form:\n"
    '  {"nodes": [{"name": str, "primitive": str, "description": str}], '
    '"edges": [{"name": str, "subject_type": str, "object_type": str, '
    '"description": str}]}\n'
)


class TypedNodeSpec(BaseModel):
    """One proposed typed node in a per-doc schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    primitive: PrimitiveTypeLiteral
    description: str = Field(min_length=1)

    @field_validator("name", "description")
    @classmethod
    def _strip_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class TypedEdgeSpec(BaseModel):
    """One proposed typed edge between typed nodes in a per-doc schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    subject_type: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    description: str = Field(min_length=1)

    @field_validator("name", "subject_type", "object_type", "description")
    @classmethod
    def _strip_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class SchemaProposal(BaseModel):
    """The single Tier-3 LLM call's structured output.

    `nodes` and `edges` are the typed-node and typed-edge surfaces the
    downstream Tier-2 schema-typed extractor binds claims against.
    Empty lists are permitted (the universal claim tuple is always the
    floor, even when induction yields nothing).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: list[TypedNodeSpec]
    edges: list[TypedEdgeSpec]


@runtime_checkable
class TaskClient(Protocol):
    """LLM-agnostic transport, mirrored from `ctrldoc.orch.task.TaskClient`."""

    def call(self, *, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Max-entropy chunk sampling
# ---------------------------------------------------------------------------


def max_entropy_sample(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    *,
    k: int,
) -> list[Chunk]:
    """Greedy farthest-point selection of k chunks on the embedding cloud.

    The seed is the chunk furthest from the cloud's centroid; subsequent
    picks maximise the minimum cosine distance to anything already
    picked. Ties break by input ordinal so the output is byte-stable
    across runs. Returns at most `min(k, len(chunks))` chunks, in pick
    order.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings must have the same length; "
            f"got {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )
    if not chunks:
        return []

    cap = min(k, len(chunks))
    vecs: list[list[float]] = [list(e) for e in embeddings]

    # Seed: the chunk furthest from the centroid.
    centroid = _centroid(vecs)
    seed_distances = [_cosine_distance(v, centroid) for v in vecs]
    seed_idx = _argmax_with_first_tie_break(seed_distances)
    picked_indices: list[int] = [seed_idx]

    # Track the minimum distance from each candidate to the picked set.
    min_dist_to_picked = [_cosine_distance(v, vecs[seed_idx]) for v in vecs]
    min_dist_to_picked[seed_idx] = -math.inf  # exclude from future picks

    while len(picked_indices) < cap:
        next_idx = _argmax_with_first_tie_break(min_dist_to_picked)
        picked_indices.append(next_idx)
        min_dist_to_picked[next_idx] = -math.inf
        for i, vec in enumerate(vecs):
            if min_dist_to_picked[i] == -math.inf:
                continue
            candidate = _cosine_distance(vec, vecs[next_idx])
            if candidate < min_dist_to_picked[i]:
                min_dist_to_picked[i] = candidate

    return [chunks[i] for i in picked_indices]


# ---------------------------------------------------------------------------
# Schema proposer
# ---------------------------------------------------------------------------


class SchemaProposer:
    """Wires sampling + a single LLM call + structured parsing.

    The proposer is stateless across `propose` calls and safe to reuse.
    `k` defaults to 10, the mid-point of the §6.4 sample range. The
    embedder is invoked exactly once per `propose` call (one
    `embed_batch` over every chunk).
    """

    def __init__(
        self,
        *,
        client: TaskClient,
        embedder: Embedder,
        k: int = _DEFAULT_SAMPLE_K,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self._client = client
        self._embedder = embedder
        self._k = k

    def propose(self, *, chunks: Sequence[Chunk], doc_id: str) -> SchemaProposal:
        if not chunks:
            raise ValueError("propose requires at least one chunk")
        embeddings = self._embedder.embed_batch([c.text for c in chunks])
        sample = max_entropy_sample(chunks, embeddings, k=self._k)
        user_message = _render_evidence(sample, doc_id=doc_id)
        raw = self._client.call(system=_SYSTEM_PROMPT, user=user_message)
        return _parse_proposal(raw)


# ---------------------------------------------------------------------------
# YAML cache (deterministic, stdlib-only)
# ---------------------------------------------------------------------------

_INDENT = "  "


def dump_schema_yaml(proposal: SchemaProposal, path: Path) -> None:
    """Write `proposal` to `path` as deterministic block-style YAML.

    The parent directory tree is created if missing — workspaces may
    write their first schema before any other file has been laid down
    in the workspace dir. The output is byte-stable across runs so the
    cache key (file hash) is reproducible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("nodes:")
    if not proposal.nodes:
        lines[-1] = "nodes: []"
    else:
        for node in proposal.nodes:
            lines.append(f"{_INDENT}- name: {_quote(node.name)}")
            lines.append(f"{_INDENT * 2}primitive: {_quote(node.primitive)}")
            lines.append(f"{_INDENT * 2}description: {_quote(node.description)}")
    lines.append("edges:")
    if not proposal.edges:
        lines[-1] = "edges: []"
    else:
        for edge in proposal.edges:
            lines.append(f"{_INDENT}- name: {_quote(edge.name)}")
            lines.append(f"{_INDENT * 2}subject_type: {_quote(edge.subject_type)}")
            lines.append(f"{_INDENT * 2}object_type: {_quote(edge.object_type)}")
            lines.append(f"{_INDENT * 2}description: {_quote(edge.description)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_schema_yaml(path: Path) -> SchemaProposal:
    """Parse a file produced by `dump_schema_yaml` back into a `SchemaProposal`.

    Accepts only the deterministic block-style subset the dumper emits
    (mapping of two top-level keys, each holding either `[]` or a list
    of single-line key/value entries with double-quoted scalars).
    Anything else raises `ValueError` — we never want to silently parse
    a hand-edited file into something the system trusts.
    """
    if not path.exists():
        raise FileNotFoundError(f"schema YAML not found: {path}")
    text = path.read_text(encoding="utf-8")
    nodes: list[TypedNodeSpec] = []
    edges: list[TypedEdgeSpec] = []
    section: str | None = None
    current: dict[str, str] = {}
    pending: list[dict[str, str]] = []

    def _flush() -> None:
        if not current:
            return
        pending.append(dict(current))
        current.clear()

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if raw_line == "nodes: []":
            _flush()
            _commit(section, pending, nodes, edges)
            section = "nodes"
            pending = []
            continue
        if raw_line == "edges: []":
            _flush()
            _commit(section, pending, nodes, edges)
            section = "edges"
            pending = []
            continue
        if raw_line == "nodes:":
            _flush()
            _commit(section, pending, nodes, edges)
            section = "nodes"
            pending = []
            continue
        if raw_line == "edges:":
            _flush()
            _commit(section, pending, nodes, edges)
            section = "edges"
            pending = []
            continue
        if raw_line.startswith(f"{_INDENT}- "):
            _flush()
            key, value = _parse_kv(raw_line[len(_INDENT) + 2 :])
            current[key] = value
            continue
        if raw_line.startswith(_INDENT * 2):
            key, value = _parse_kv(raw_line[len(_INDENT) * 2 :])
            current[key] = value
            continue
        raise ValueError(f"unexpected YAML line: {raw_line!r}")
    _flush()
    _commit(section, pending, nodes, edges)
    return SchemaProposal(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _render_evidence(sample: Sequence[Chunk], *, doc_id: str) -> str:
    lines: list[str] = [f"Document: {doc_id}", "Sampled excerpts:", ""]
    for ordinal, chunk in enumerate(sample, start=1):
        lines.append(f"[{ordinal}] (chunk {chunk.id})")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_proposal(raw: str) -> SchemaProposal:
    payload = _strip_fence(raw).strip()
    return SchemaProposal.model_validate_json(payload)


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline == -1:
            return text
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text


def _centroid(vecs: Sequence[Sequence[float]]) -> list[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    acc = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            acc[i] += x
    n = float(len(vecs))
    return [x / n for x in acc]


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / math.sqrt(na * nb)


def _argmax_with_first_tie_break(values: Sequence[float]) -> int:
    best_idx = 0
    best = values[0]
    for i in range(1, len(values)):
        if values[i] > best:
            best = values[i]
            best_idx = i
    return best_idx


def _quote(value: str) -> str:
    # Round-trip-safe quoting: escape backslashes and double quotes, then wrap.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _unquote(value: str) -> str:
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        raise ValueError(f"expected double-quoted scalar, got: {value!r}")
    body = value[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt == "\\":
                out.append("\\")
            elif nxt == '"':
                out.append('"')
            else:
                raise ValueError(f"unsupported escape sequence in YAML scalar: \\{nxt}")
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_kv(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"malformed YAML key/value: {text!r}")
    key, _, value = text.partition(":")
    return key.strip(), _unquote(value.strip())


def _commit(
    section: str | None,
    pending: list[dict[str, str]],
    nodes: list[TypedNodeSpec],
    edges: list[TypedEdgeSpec],
) -> None:
    if section is None or not pending:
        return
    if section == "nodes":
        for entry in pending:
            nodes.append(TypedNodeSpec(**entry))  # type: ignore[arg-type]
    elif section == "edges":
        for entry in pending:
            edges.append(TypedEdgeSpec(**entry))
    else:
        raise ValueError(f"unknown YAML section: {section!r}")


__all__ = [
    "PRIMITIVE_LIBRARY",
    "SchemaProposal",
    "SchemaProposer",
    "TaskClient",
    "TypedEdgeSpec",
    "TypedNodeSpec",
    "dump_schema_yaml",
    "load_schema_yaml",
    "max_entropy_sample",
]
