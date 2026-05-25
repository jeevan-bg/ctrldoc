"""`ctrldoc compare / coverage / merge / list-check / graph / schema / calibration` CLI surface.

Top-level Typer commands that wire the §6.10 tool dispatcher to the CLI.
Each command is a thin (≤ 40-line) wrapper: build the Pydantic input
from CLI args, register the per-installation handler floor on a fresh
dispatcher (storage-backed deps from `<runs_path>/indexes/`), call the
dispatcher, render Markdown + JSON via the shared `_emit_output`.

Unwired tools (whose deps aren't satisfied — typically the NLI scorer
or LLM-backed deps) surface as a structured `status: "not_implemented"`
envelope rather than crashing the CLI; the response still names the
tool and lists what's missing so an operator can decide where to wire.

`schema show / pin` are the only commands that don't dispatch through
the tool surface — they read / write the per-doc induced-schema YAML
file directly under `<runs_path>/indexes/<doc_id>.schema.yaml` (write
target for `pin` is `<runs_path>/workspaces/<workspace_id>/schema.yaml`).

SPEC-REF: §9 (CLI surface), §6.10 (tool surface)
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ctrldoc.cli import app

runner = CliRunner()


_CONFIG_TEMPLATE = """\
[models]
planner = "claude-opus-4-7"
judge_tier1 = "qwen2.5:7b-instruct-q4_K_M"
judge_tier2 = "claude-opus-4-7"
verifier_nli = "deberta-v3-large-mnli"
embedder = "bge-m3"

[budgets]
max_cost_usd = 5.0
max_tokens_per_call = 16000
max_wall_clock_min = 30

[concurrency]
anthropic_concurrent = 8
ollama_concurrent = 2

[paths]
index_path = "{index_path}"
runs_path = "{runs_path}"
traces_path = "{traces_path}"
"""


def _write_config(tmp_path: Path) -> Path:
    index_path = tmp_path / "index"
    runs_path = tmp_path / "runs"
    traces_path = tmp_path / "traces"
    for p in (index_path, runs_path, traces_path):
        p.mkdir(exist_ok=True)
    cfg = tmp_path / "ctrldoc.toml"
    cfg.write_text(
        _CONFIG_TEMPLATE.format(
            index_path=index_path.as_posix(),
            runs_path=runs_path.as_posix(),
            traces_path=traces_path.as_posix(),
        ),
        encoding="utf-8",
    )
    return cfg


def _invoke(cfg: Path, *args: str, format_flag: str = "json") -> object:
    return runner.invoke(
        app,
        ["--config", str(cfg), "--format", format_flag, *args],
    )


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_calibration_returns_empty_envelope_with_no_data(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "calibration")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "calibration"
    assert payload["status"] == "ok"
    assert payload["ece_per_backend"] == {}
    assert payload["sample_sizes"] == {}


def test_calibration_markdown_renders_header(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "calibration", format_flag="markdown")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    assert "# ctrldoc — calibration" in result.stdout  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_returns_not_implemented_without_nli_scorer(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "compare", "ws-deadbeef", "doc-a", "doc-b")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "compare"
    assert payload["status"] == "not_implemented"
    assert payload["tool_name"] == "compare"
    assert payload["inputs"]["workspace_id"] == "ws-deadbeef"
    assert payload["inputs"]["doc_ids"] == ["doc-a", "doc-b"]


def test_compare_requires_two_doc_ids(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "compare", "ws-deadbeef", "doc-a")
    assert result.exit_code == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_returns_not_implemented_without_nli_scorer(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(
        cfg,
        "coverage",
        "--workspace",
        "ws-deadbeef",
        "--target",
        "doc-b",
        "--source",
        "doc-a",
    )
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "coverage"
    assert payload["status"] == "not_implemented"
    assert payload["tool_name"] == "coverage"
    assert payload["inputs"]["workspace_id"] == "ws-deadbeef"
    assert payload["inputs"]["target_doc_id"] == "doc-b"
    assert payload["inputs"]["source_doc_id"] == "doc-a"


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_returns_not_implemented_without_nli_scorer(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    out_path = tmp_path / "merged.md"
    result = _invoke(
        cfg,
        "merge",
        "--workspace",
        "ws-deadbeef",
        "--output",
        str(out_path),
        "doc-a",
        "doc-b",
    )
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "merge"
    assert payload["status"] == "not_implemented"
    assert payload["inputs"]["doc_ids"] == ["doc-a", "doc-b"]
    assert payload["inputs"]["output_path"] == str(out_path)


# ---------------------------------------------------------------------------
# list-check
# ---------------------------------------------------------------------------


def test_list_check_returns_not_implemented_without_nli_scorer(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    list_md = tmp_path / "list.md"
    list_md.write_text("- alpha claim\n- beta claim\n", encoding="utf-8")
    result = _invoke(cfg, "list-check", str(list_md), "doc-a")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "list-check"
    assert payload["status"] == "not_implemented"
    assert payload["inputs"]["doc_id"] == "doc-a"
    items = payload["inputs"]["items"]
    assert len(items) == 2
    assert items[0]["text"] == "alpha claim"
    assert items[1]["text"] == "beta claim"


def test_list_check_rejects_missing_list_file(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "list-check", str(tmp_path / "absent.md"), "doc-a")
    assert result.exit_code == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# graph show / query / traverse
# ---------------------------------------------------------------------------


def test_graph_show_returns_not_implemented_without_per_doc_edges(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "graph", "show", "doc-a")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "graph show"
    assert payload["status"] == "not_implemented"
    assert payload["tool_name"] == "map"
    assert payload["inputs"]["doc_id"] == "doc-a"


def test_graph_query_returns_concept_id_null_when_no_match(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "graph", "query", "doc-a", "--concept", "Backpropagation")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "graph query"
    assert payload["status"] == "ok"
    assert payload["tool_name"] == "lookup_concept"
    assert payload["concept_id"] is None


def test_graph_traverse_emits_empty_node_list_on_empty_store(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(
        cfg,
        "graph",
        "traverse",
        "claim-deadbeef",
        "--edge-type",
        "entails",
        "--direction",
        "forward",
        "--hops",
        "2",
    )
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "graph traverse"
    assert payload["status"] == "ok"
    assert payload["tool_name"] == "traverse"
    assert payload["node_ids"] == []


def test_graph_traverse_rejects_unknown_edge_type(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(
        cfg,
        "graph",
        "traverse",
        "claim-deadbeef",
        "--edge-type",
        "totally-bogus",
        "--direction",
        "forward",
        "--hops",
        "2",
    )
    assert result.exit_code == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# schema show / pin
# ---------------------------------------------------------------------------


def _seed_doc_schema(runs_path: Path, doc_id: str) -> Path:
    """Write a minimal induced-schema YAML under the per-doc index dir."""
    from ctrldoc.extract.schema_proposer import (
        SchemaProposal,
        TypedEdgeSpec,
        TypedNodeSpec,
        dump_schema_yaml,
    )

    proposal = SchemaProposal(
        nodes=[TypedNodeSpec(name="Concept", primitive="Entity", description="A core concept.")],
        edges=[
            TypedEdgeSpec(
                name="related_to",
                subject_type="Concept",
                object_type="Concept",
                description="Generic relation.",
            )
        ],
    )
    indexes_dir = runs_path / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    schema_path = indexes_dir / f"{doc_id}.schema.yaml"
    dump_schema_yaml(proposal, schema_path)
    return schema_path


def test_schema_show_emits_yaml_payload_when_present(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    runs_path = tmp_path / "runs"
    seeded = _seed_doc_schema(runs_path, doc_id="doc-a")
    result = _invoke(cfg, "schema", "show", "doc-a")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "schema show"
    assert payload["status"] == "ok"
    assert payload["doc_id"] == "doc-a"
    assert payload["schema_path"] == str(seeded)
    assert "nodes:" in payload["yaml"]


def test_schema_show_exits_two_when_absent(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "schema", "show", "doc-missing")
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "no induced schema" in result.stderr  # type: ignore[attr-defined]


def test_schema_pin_copies_source_yaml_into_workspace_dir(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    runs_path = tmp_path / "runs"
    _seed_doc_schema(runs_path, doc_id="doc-a")
    result = _invoke(
        cfg,
        "schema",
        "pin",
        "--workspace",
        "ws-audit-2026",
        "--from",
        "doc-a",
    )
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    payload = json.loads(result.stdout)  # type: ignore[attr-defined]
    assert payload["command"] == "schema pin"
    assert payload["status"] == "ok"
    pinned = Path(payload["pinned_path"])
    assert pinned.exists()
    assert pinned.parent.name == "ws-audit-2026"
    # Content equality with the source.
    src = runs_path / "indexes" / "doc-a.schema.yaml"
    assert pinned.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_schema_pin_exits_two_when_source_absent(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(
        cfg,
        "schema",
        "pin",
        "--workspace",
        "ws-audit",
        "--from",
        "doc-missing",
    )
    assert result.exit_code == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# markdown rendering for not-implemented stubs
# ---------------------------------------------------------------------------


def test_compare_markdown_renders_not_implemented_block(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)
    result = _invoke(cfg, "compare", "ws-deadbeef", "doc-a", "doc-b", format_flag="markdown")
    assert result.exit_code == 0, result.stderr  # type: ignore[attr-defined]
    assert "# ctrldoc — compare" in result.stdout  # type: ignore[attr-defined]
    assert "not implemented" in result.stdout.lower()  # type: ignore[attr-defined]
