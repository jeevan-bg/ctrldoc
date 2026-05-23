# ctrldoc

**Feed arbitrarily large documents to an LLM and get citation-grounded, drift-free, hallucination-bounded analysis.**

`ctrldoc` is a local-first analysis substrate. It indexes a document once, then serves a fixed set of analytical playbooks — trustworthy QA, coverage audits, quality audits, analytical reviews, anomaly scans, and concept-relation mapping — by routing tiny, isolated, structured tasks to language models. The LLM never sees the raw document; only retrieved spans, structured findings, or distilled state.

## Why

Large documents break LLMs in predictable ways:

- **Context rot** — long prompts degrade in the middle.
- **Context dilution** — relevant signal gets buried.
- **Drift** — multi-turn reasoning veers off topic.
- **Hallucination** — claims appear without grounding.

`ctrldoc` removes all four *by construction*: every sub-task is a fresh, stateless API call with only the evidence it needs, every claim is independently verified against the index, and every output carries citations to exact spans in the source.

## Status

**v0.2.3 — CLI wiring in progress.** Every protocol seam already has a real production backend (see v0.2.0 below); v0.2.x is wiring the typer CLI through the production stack so `ctrldoc ingest / audit / qa / review / scan / map` drives real documents end-to-end with Markdown reports. Six playbooks, eval harness, family invariants (ingest, retrieval, verifier, adversarial, determinism, performance, canary), CLI, runnable examples. Production wirings shipped in v0.2.0: `BAAI/bge-reranker-v2-m3` (L2 reranker), `cross-encoder/nli-deberta-v3-large` (L3 NLI), `bge-m3` (L0 dense embedder, via Ollama), `qwen2.5:7b-instruct-q4_K_M` (L3 tier-1 LLM-judge, via Ollama), `sqlite-vec` (L1 persistent dense-vector index), and `fastcoref` (L0 coreference resolver). See [CHANGELOG.md](CHANGELOG.md) for the per-release breakdown, and [docs/SPEC.md](docs/SPEC.md) for the full specification.

## Install

```bash
git clone https://github.com/<your-username>/ctrldoc.git
cd ctrldoc
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,index]"
```

Requirements: Python 3.11+, macOS or Linux. For local LLM backends (optional): [Ollama](https://ollama.com) with `ollama pull bge-m3 qwen2.5:7b-instruct-q4_K_M`. For Anthropic-backed playbooks: set `ANTHROPIC_API_KEY` (or place it in a `.env` file at the repo root — never commit it).

## Quickstart

The repo ships a synthetic gold document at `tests/fixtures/synthetic/gold_doc.md` so you can verify the install without any LLM credentials.

```bash
# 1. Ingest the synthetic doc under the heuristic profile —
#    deterministic L0 pipeline end-to-end, no LLM, no Ollama.
#    Writes the Markdown report + JSON result to ./runs/<run_id>/.
ctrldoc --profile heuristic ingest tests/fixtures/synthetic/gold_doc.md \
    --output-dir ./runs --doc-id aurora

# 2. Run the deterministic anomaly-scan detectors (hedge words +
#    empty section summaries). No LLM required.
ctrldoc scan

# 3. See the full CLI surface.
ctrldoc --help
```

The `ingest` subcommand writes a Markdown report (`runs/<run_id>/report.md`) plus a structured JSON payload (`runs/<run_id>/result.json`). Pass `--format json` for just JSON, `--format both` for both. The default `--profile thrifty` upgrades the embedder to `bge-m3` via Ollama and persists per-doc SQLite + sqlite-vec at `runs/indexes/<doc_hash>.{db,vec.db}`. The QA / audit / review / map subcommands still emit a stub JSON envelope; the per-playbook wiring lands in S-113 .. S-117.

For end-to-end Python walkthroughs of every UC playbook (deterministic stubs, no API key needed), see [`examples/`](examples/):

```bash
python examples/01_qa.py
python examples/05_anomaly_scan.py
```

## Use cases (one playbook each)

| # | Playbook | Question it answers |
|---|---|---|
| UC1 | `qa` | "What does the doc say about X — with citations?" |
| UC2 | `coverage_audit` | "Does the doc address every item in this checklist?" |
| UC3 | `quality_audit` | "Is this a well-formed L0 spec / RFC / runbook?" |
| UC4 | `analytical_review` | "What are the weaknesses and gaps in this doc?" |
| UC5 | `anomaly_scan` | "Surface suspicious patterns for triage." |
| UC6 | `relation_map` | "How do these concepts relate across the doc?" |

## Architecture (one line per layer)

- **L0 Ingest** — parse → coref → NER → chunk → embed → index.
- **L1 Multi-view index** — structural tree + dense vectors + BM25 + entity index.
- **L2 Retrieval** — planner emits a small DSL; executor fuses views; reranker prunes.
- **L3 Verifier** — decomposes claims, re-retrieves independently, NLI + LLM-judge.
- **L4 Orchestrator** — stateless tasks, structured outputs, tiered model routing, prompt caching.
- **L5 Playbooks** — one per use case, ~200–500 LOC each.

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Design principles (non-negotiable)

1. The LLM never sees the raw full document.
2. Every claim is cited or refused.
3. Every sub-task is a fresh, stateless API call.
4. Storage is pluggable; SQLite is the default, not the contract.
5. Every output carries provenance.

## Documentation

- [docs/SPEC.md](docs/SPEC.md) — the full MVP specification.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system overview.
- [docs/SPEC_TRACE.md](docs/SPEC_TRACE.md) — spec-to-code traceability matrix.
- [docs/DECISIONS.md](docs/DECISIONS.md) — architectural decision records.
- [docs/TESTING.md](docs/TESTING.md) — test strategy and the 14 test families.
- [examples/](examples/) — runnable per-playbook walkthroughs.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute.
- [CHANGELOG.md](CHANGELOG.md) — release history.

## License

Apache 2.0. See [LICENSE](LICENSE).
