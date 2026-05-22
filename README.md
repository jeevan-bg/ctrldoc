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

Pre-MVP. Active development. See [docs/SPEC.md](docs/SPEC.md) for the full specification and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system overview.

## Install

```bash
git clone https://github.com/<your-username>/ctrldoc.git
cd ctrldoc
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requirements: Python 3.11+, macOS or Linux. For local models: [Ollama](https://ollama.com) and `ollama pull bge-m3 qwen2.5:7b-instruct-q4_K_M`.

## Quick start

```bash
# Index a document
ctrldoc ingest path/to/doc.pdf --out ./index.db

# Ask a question (UC1)
ctrldoc qa ./index.db "What does §4.2 imply about fault tolerance?"

# Audit coverage (UC2)
ctrldoc audit ./index.db --checklist threats.md

# Analytical review (UC4)
ctrldoc review ./index.db --lenses auto
```

See [examples/](examples/) for full walkthroughs.

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
- [docs/DECISIONS.md](docs/DECISIONS.md) — architectural decision records.
- [docs/TESTING.md](docs/TESTING.md) — test strategy and the 14 test families.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute.
- [CHANGELOG.md](CHANGELOG.md) — release history.

## License

Apache 2.0. See [LICENSE](LICENSE).
