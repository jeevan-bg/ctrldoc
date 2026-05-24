# ctrldoc

**Universal claim-graph substrate for citation-grounded, calibrated, replayable multi-document analysis.**

`ctrldoc` ingests arbitrarily large documents into a typed claim graph, derives a shared concept lattice across N documents in a workspace, and serves every analytical operation — coverage, compare, merge, list-check, qa, map — through one optimal-transport engine. Every edge carries calibrated confidence with shipped ECE; every verdict is replayable from an append-only ledger; every output cites source spans in the original documents. A Model Context Protocol (MCP) server exposes the full 13-tool surface to Claude (Desktop, CLI, or any MCP-aware host).

## Why

Large documents break LLMs in predictable ways:

- **Context rot** — long prompts degrade in the middle.
- **Context dilution** — relevant signal gets buried.
- **Drift** — multi-turn reasoning veers off topic.
- **Hallucination** — claims appear without grounding.
- **Cross-doc blind spots** — even when one doc fits in context, comparing N of them does not.

`ctrldoc` removes all of these *by construction*: every sub-task is a fresh, stateless API call with only the evidence it needs; every claim is independently verified against the index; every output carries citations to exact spans; and cross-doc claims are aligned in a shared concept lattice (not a multi-doc super-prompt).

## Status

**v1.0.0 — universal substrate shipped.** The claim graph (L1.5) and workspace (L2.5) are the new primitives; `compare` / `coverage` / `merge` / `list_check` collapse into one optimal-transport engine; schema co-induction (§6.4) makes adapters emerge per document; probabilistic edges carry calibrated confidence with shipped ECE per backend; the MCP server (§11) is the Claude integration. Storage `schema_version` bumped `0.1.0`→`0.2.0` — v0.3 indexes require re-ingest (see [MIGRATION_v0.3_to_v1.0.md](MIGRATION_v0.3_to_v1.0.md)). The full v0.3 surface (`ingest`, `qa`, `scan`, `map`, `audit`, `review`) is preserved unchanged. See [CHANGELOG.md](CHANGELOG.md) for the per-release breakdown and [docs/SPEC.md](docs/SPEC.md) for the live v1 specification.

## Install

```bash
git clone https://github.com/<your-username>/ctrldoc.git
cd ctrldoc
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,index]"
```

Requirements: Python 3.11+, macOS or Linux. For local LLM backends (optional): [Ollama](https://ollama.com) with `ollama pull bge-m3 qwen2.5:7b-instruct-q4_K_M`. For Anthropic-backed operations: set `ANTHROPIC_API_KEY` (or place it in a `.env` file at the repo root — never commit it).

## Quickstart

The repo ships a synthetic gold document at `tests/fixtures/synthetic/gold_doc.md` so you can verify the install without any LLM credentials.

```bash
# 1. Ingest the synthetic doc under the heuristic profile —
#    deterministic L0 pipeline end-to-end, no LLM, no Ollama.
ctrldoc --profile heuristic ingest tests/fixtures/synthetic/gold_doc.md \
    --output-dir ./runs --doc-id aurora

# 2. Build a workspace and add the doc.
ctrldoc workspace create demo
ctrldoc workspace add demo aurora

# 3. See the full CLI surface.
ctrldoc --help
```

For end-to-end Python walkthroughs of the v1 substrate (workspace, optimal-transport coverage, merge — all hermetic, no API key needed), see [`examples/v1/`](examples/v1/):

```bash
python examples/v1/01_workspace.py
python examples/v1/02_coverage_transport.py
python examples/v1/03_merge_transport.py
```

The v0.3 per-playbook walkthroughs (still functional) live in [`examples/`](examples/).

## The v1 end state (from §16 of the spec)

```bash
ctrldoc workspace create due-diligence
ctrldoc workspace add due-diligence company-spec.pdf
ctrldoc workspace add due-diligence security-policy.pdf
ctrldoc workspace add due-diligence soc2-report.pdf
ctrldoc coverage --workspace due-diligence \
    --target soc2-report.pdf --source security-policy.pdf
```

You get a Markdown report + JSON payload where every line is `(claim, verdict, calibrated_confidence, citations_in_both_docs)`. You can run `ctrldoc ledger replay` six months later and reproduce every verdict inside the §6.5 ±0.02 determinism gate. Or you run `ctrldoc mcp serve` and ask Claude in chat: *"compare these three docs"* — Claude gets back the same structured result, traceable to source spans, with shipped ECE telling it how much to trust each number.

## CLI surface

| Command | Purpose | Layer |
|---|---|---|
| `ingest` | Parse → coref → NER → chunk → embed → index a document. | L0–L1 |
| `workspace {create,add,list,info}` | CRUD over the L2.5 multi-doc primitive; shares one concept lattice. | L2.5 |
| `coverage` | Per-target-claim verdicts via optimal transport — `(Covered, Missing, Contradicted)` with calibrated confidence and source-span citations in both docs. | L5 |
| `compare` | Per-cluster verdicts (`StrengthA, StrengthB, Gap`) across N docs. | L5 |
| `merge` | Lossless synthesis of N docs — every input claim maps to exactly one output cluster (the §13 loss invariant). | L5 |
| `list-check` | Per-item verdicts of a list against a doc. | L5 |
| `qa` | "What does the doc say about X — with citations?" | L5 |
| `map` | Render the concept graph (Mermaid). | L5 |
| `scan` | Deterministic anomaly detector battery (hedge words, empty summaries, …). | L5 |
| `audit` / `review` | v0.3 single-doc audits — preserved unchanged. | L5 |
| `graph {show,query}` | Inspect the per-doc typed claim graph. | L1.5 |
| `schema {show,pin}` | Inspect or pin the per-doc induced schema (YAML cache from §6.4). | L0 |
| `calibration` | One-shot ECE measurement for any NLI backend (release gate `ECE ≤ 0.05`). | L3 |
| `ledger {list,show,replay}` | Walk the append-only verdict ledger; `replay` is the §6.5 determinism gate. | L4 |
| `mcp serve` | Model Context Protocol server over stdio; exposes the 13-tool surface to any MCP-aware host. | meta |

`ctrldoc --help` enumerates every flag.

## Architecture (one line per layer)

- **L0 Adaptive ingest** — parse → coref → NER → chunk → embed → index; **per-doc schema co-induction** (§6.4) emits a YAML schema cached on disk.
- **L1 Multi-view index** — structural tree + dense vectors + BM25 + entity index + **claims / concepts / typed_edges** tables (v1 additions).
- **L1.5 Claim graph** — universal claim tuple as the logic floor (§6.2); span / claim / concept triplane.
- **L2 Retrieval** — planner emits a small DSL; fusion blends dense + BM25 + entity + **personalized PageRank** over typed edges (§6.9); reranker prunes.
- **L2.5 Workspace** — N docs share one Galois concept lattice (§6.3); cross-doc edges (`aligned_with`, `entails_across`, `contradicts_across`) are lazy, cached, and linear in `|A| × k`.
- **L3 Probabilistic edge inference** — heuristic + NLI + LLM-judge with paraphrase voting (§6.5) and isotonic calibration; shipped ECE per backend.
- **L4 Tool-using orchestrator** — forced tool calls only (no free-form reasoning); append-only verdict ledger; replay within ±0.02.
- **L5 Universal operations** — one optimal-transport engine drives `compare` / `coverage` / `merge` / `list_check` / `map` / `qa`.
- **L6 Trace renderer** — proof trace: spans → claims → edges → verdict + calibrated confidence.

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Design principles (the 14 non-negotiables, §13)

**From v0.3 (preserved verbatim in v1):**

1. The LLM never sees the raw full document.
2. Every claim is cited or refused.
3. Every sub-task is a fresh, stateless API call.
4. Storage is pluggable; SQLite is the default, not the contract.
5. Every output carries provenance.

**New in v1:**

6. Universal claim tuple is always extracted — never replaced by adapter-only output.
7. Residual rate is observable; the CLI surfaces `unmatched_claim_rate`.
8. Edges carry calibrated confidence — no boolean edges in shipped output.
9. ECE is shipped per backend; release blocks if `ECE > 0.05`.
10. Optimal-transport ops respect the loss invariant — `merge` maps every input claim to exactly one output cluster.
11. Every cross-doc edge has a source-span citation in both documents.
12. The tool-using orchestrator uses forced tool calls — no free-form reasoning at L4.
13. The verdict ledger is append-only and replayable within ±0.02.
14. MCP tool schemas are versioned and bumped on breaking change.

## Documentation

- [docs/SPEC.md](docs/SPEC.md) — the live v1 normative specification.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system overview.
- [docs/SPEC_TRACE.md](docs/SPEC_TRACE.md) — spec-to-code traceability matrix.
- [docs/DECISIONS.md](docs/DECISIONS.md) — architectural decision records (ADR index in `docs/decisions/INDEX.md`).
- [docs/TESTING.md](docs/TESTING.md) — test strategy and the 14 test families.
- [examples/v1/](examples/v1/) — runnable v1 walkthroughs.
- [examples/](examples/) — v0.3 per-playbook walkthroughs (still functional).
- [MIGRATION_v0.3_to_v1.0.md](MIGRATION_v0.3_to_v1.0.md) — upgrade guide.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute.
- [CHANGELOG.md](CHANGELOG.md) — release history.

## License

Apache 2.0. See [LICENSE](LICENSE).
