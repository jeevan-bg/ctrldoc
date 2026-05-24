"""Tests for the L0 schema-proposer (max-entropy sample + LLM proposal + YAML cache).

SPEC-REF: §6.4
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctrldoc.extract.schema_proposer import (
    SchemaProposal,
    SchemaProposer,
    TypedEdgeSpec,
    TypedNodeSpec,
    dump_schema_yaml,
    load_schema_yaml,
    max_entropy_sample,
)
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.models import Chunk


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        section_id="sec-0",
        text=text,
        token_count=max(1, len(text.split())),
        char_start=0,
        char_end=len(text),
        embedding_id=f"emb-{chunk_id}",
    )


# ---------------------------------------------------------------------------
# Max-entropy chunk sampling
# ---------------------------------------------------------------------------


def test_max_entropy_sample_returns_requested_count() -> None:
    """k=8 across 30 distinct chunks must yield exactly 8 unique chunk ids."""
    chunks = [_make_chunk(f"c{i}", f"unique text content number {i}") for i in range(30)]
    embedder = HashEmbedder(dimension=32, seed=1)
    embeddings = embedder.embed_batch([c.text for c in chunks])

    picked = max_entropy_sample(chunks, embeddings, k=8)

    assert len(picked) == 8
    assert len({c.id for c in picked}) == 8


def test_max_entropy_sample_caps_at_available_chunks() -> None:
    """k>len(chunks) must return every chunk, not raise."""
    chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(3)]
    embedder = HashEmbedder(dimension=16, seed=2)
    embeddings = embedder.embed_batch([c.text for c in chunks])

    picked = max_entropy_sample(chunks, embeddings, k=10)

    assert len(picked) == 3
    assert {c.id for c in picked} == {c.id for c in chunks}


def test_max_entropy_sample_rejects_mismatched_lengths() -> None:
    chunks = [_make_chunk("c0", "a"), _make_chunk("c1", "b")]
    embedder = HashEmbedder(dimension=8, seed=3)
    embeddings = embedder.embed_batch(["a"])  # too short on purpose

    with pytest.raises(ValueError, match="length"):
        max_entropy_sample(chunks, embeddings, k=2)


def test_max_entropy_sample_rejects_non_positive_k() -> None:
    chunks = [_make_chunk("c0", "x")]
    with pytest.raises(ValueError, match="k must be"):
        max_entropy_sample(chunks, [[0.0]], k=0)


def test_max_entropy_sample_handles_empty_input() -> None:
    assert max_entropy_sample([], [], k=5) == []


def test_max_entropy_sample_is_deterministic() -> None:
    """Identical input → identical output. Cache discipline depends on this."""
    chunks = [_make_chunk(f"c{i}", f"distinct phrase {i} here") for i in range(20)]
    embedder = HashEmbedder(dimension=24, seed=4)
    embeddings = embedder.embed_batch([c.text for c in chunks])

    first = [c.id for c in max_entropy_sample(chunks, embeddings, k=6)]
    second = [c.id for c in max_entropy_sample(chunks, embeddings, k=6)]

    assert first == second


def test_max_entropy_sample_prefers_diverse_points() -> None:
    """When one chunk is far from the cluster, it must be picked early."""
    # 9 near-duplicates plus 1 orthogonal outlier.
    cluster_vecs: list[list[float]] = [[1.0, 0.0, 0.0] for _ in range(9)]
    # Add tiny per-chunk jitter so seed selection (argmax norm) is unambiguous.
    for i, vec in enumerate(cluster_vecs):
        vec[1] = i * 1e-6
    outlier = [0.0, 0.0, 1.0]
    embeddings = [outlier, *cluster_vecs]
    chunks = [_make_chunk(f"c{i}", f"text-{i}") for i in range(10)]

    picked = max_entropy_sample(chunks, embeddings, k=2)
    picked_ids = {c.id for c in picked}

    # Seed is the chunk farthest from the centroid → outlier "c0".
    # Second pick is the chunk farthest from the seed → any cluster member.
    assert "c0" in picked_ids
    assert len(picked_ids) == 2


# ---------------------------------------------------------------------------
# Schema proposal Pydantic surface
# ---------------------------------------------------------------------------


def test_typed_node_spec_rejects_unknown_primitive() -> None:
    with pytest.raises(ValueError):
        TypedNodeSpec(name="Widget", primitive="Gadget", description="a thing")  # type: ignore[arg-type]


def test_typed_edge_spec_rejects_blank_subject_or_object() -> None:
    with pytest.raises(ValueError):
        TypedEdgeSpec(name="depends_on", subject_type="", object_type="Entity", description="x")
    with pytest.raises(ValueError):
        TypedEdgeSpec(name="depends_on", subject_type="Entity", object_type="", description="x")


def test_schema_proposal_round_trips_json() -> None:
    proposal = SchemaProposal(
        nodes=[
            TypedNodeSpec(name="API", primitive="Entity", description="A callable API surface"),
            TypedNodeSpec(name="Deadline", primitive="Quantity", description="A time bound"),
        ],
        edges=[
            TypedEdgeSpec(
                name="exposes",
                subject_type="API",
                object_type="API",
                description="API X re-exports API Y",
            )
        ],
    )

    rehydrated = SchemaProposal.model_validate_json(proposal.model_dump_json())
    assert rehydrated == proposal


# ---------------------------------------------------------------------------
# SchemaProposer end-to-end (single batched LLM call)
# ---------------------------------------------------------------------------


class _CountingClient:
    """TaskClient stub that records each call and returns a canned JSON body."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def call(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.payload


def test_schema_proposer_issues_exactly_one_llm_call() -> None:
    """§6.4 step 2 says 'one Tier-3 LLM call' — non-negotiable."""
    chunks = [_make_chunk(f"c{i}", f"sentence number {i} about Topic-{i}") for i in range(15)]
    embedder = HashEmbedder(dimension=32, seed=5)
    payload = json.dumps(
        {
            "nodes": [
                {"name": "Topic", "primitive": "Entity", "description": "A subject of discussion"}
            ],
            "edges": [],
        }
    )
    client = _CountingClient(payload)
    proposer = SchemaProposer(client=client, embedder=embedder, k=8)

    proposal = proposer.propose(chunks=chunks, doc_id="doc-1")

    assert len(client.calls) == 1
    assert len(proposal.nodes) == 1
    assert proposal.nodes[0].name == "Topic"


def test_schema_proposer_evidence_only_contains_sampled_chunks() -> None:
    """The user message must carry only the k sampled excerpts — never the full doc."""
    # Use chunk ids (zero-padded so c01 is not a substring of c10) — both id
    # and text appear verbatim in the evidence header, but only id avoids the
    # numeric substring trap.
    chunks = [_make_chunk(f"c{i:02d}", f"distinct content number {i:02d}") for i in range(40)]
    embedder = HashEmbedder(dimension=24, seed=6)
    payload = json.dumps({"nodes": [], "edges": []})
    client = _CountingClient(payload)
    proposer = SchemaProposer(client=client, embedder=embedder, k=10)

    proposer.propose(chunks=chunks, doc_id="doc-2")

    _system, user = client.calls[0]
    occurrences = sum(1 for c in chunks if f"(chunk {c.id})" in user)
    assert occurrences == 10


def test_schema_proposer_prompt_lists_primitive_library() -> None:
    """The system message must enumerate the 10 closed primitives."""
    chunks = [_make_chunk("c0", "anything")]
    embedder = HashEmbedder(dimension=8, seed=7)
    payload = json.dumps({"nodes": [], "edges": []})
    client = _CountingClient(payload)
    proposer = SchemaProposer(client=client, embedder=embedder, k=1)

    proposer.propose(chunks=chunks, doc_id="doc-3")

    system, _user = client.calls[0]
    for primitive in (
        "Entity",
        "Event",
        "Process",
        "Property",
        "Quantity",
        "Definition",
        "Assertion",
        "Obligation",
        "Citation",
        "Relation",
    ):
        assert primitive in system, f"primitive {primitive!r} missing from system prompt"


def test_schema_proposer_rejects_unknown_primitive_in_response() -> None:
    """Defense-in-depth: the LLM can hallucinate a primitive; we must catch it."""
    chunks = [_make_chunk("c0", "x")]
    embedder = HashEmbedder(dimension=8, seed=8)
    bad = json.dumps(
        {"nodes": [{"name": "X", "primitive": "Gadget", "description": "y"}], "edges": []}
    )
    client = _CountingClient(bad)
    proposer = SchemaProposer(client=client, embedder=embedder, k=1)

    with pytest.raises(ValueError):
        proposer.propose(chunks=chunks, doc_id="doc-4")


def test_schema_proposer_rejects_empty_chunks() -> None:
    embedder = HashEmbedder(dimension=8, seed=9)
    payload = json.dumps({"nodes": [], "edges": []})
    client = _CountingClient(payload)
    proposer = SchemaProposer(client=client, embedder=embedder, k=8)

    with pytest.raises(ValueError, match="at least one chunk"):
        proposer.propose(chunks=[], doc_id="doc-5")


# ---------------------------------------------------------------------------
# YAML cache (round-trip through the file path that the workspace stores)
# ---------------------------------------------------------------------------


def test_dump_and_load_schema_yaml_round_trips(tmp_path: Path) -> None:
    proposal = SchemaProposal(
        nodes=[
            TypedNodeSpec(name="Function", primitive="Entity", description="A code function"),
            TypedNodeSpec(name="Tag", primitive="Property", description="A label on a function"),
        ],
        edges=[
            TypedEdgeSpec(
                name="tagged_with",
                subject_type="Function",
                object_type="Tag",
                description="Function carries Tag",
            )
        ],
    )
    target = tmp_path / "doc-7" / "schema.yaml"

    dump_schema_yaml(proposal, target)
    loaded = load_schema_yaml(target)

    assert loaded == proposal


def test_dump_schema_yaml_is_deterministic(tmp_path: Path) -> None:
    """Two consecutive dumps must produce byte-identical files (cache discipline)."""
    proposal = SchemaProposal(
        nodes=[
            TypedNodeSpec(name="Beta", primitive="Entity", description="b"),
            TypedNodeSpec(name="Alpha", primitive="Entity", description="a"),
        ],
        edges=[
            TypedEdgeSpec(
                name="relates_to",
                subject_type="Alpha",
                object_type="Beta",
                description="x",
            )
        ],
    )
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    dump_schema_yaml(proposal, a)
    dump_schema_yaml(proposal, b)
    assert a.read_bytes() == b.read_bytes()


def test_load_schema_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_schema_yaml(tmp_path / "absent.yaml")


def test_dump_schema_yaml_creates_parent_directories(tmp_path: Path) -> None:
    """Workspace dirs may not exist yet — the dump must mkdir -p."""
    target = tmp_path / "deep" / "nested" / "schema.yaml"
    proposal = SchemaProposal(nodes=[], edges=[])
    dump_schema_yaml(proposal, target)
    assert target.exists()
