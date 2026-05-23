# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] — 2026-05-23

Production dense-embedder landing behind the v0.1.0 protocol seam.

### Added

- **`OllamaEmbedder` (S-036b)** — `bge-m3` via a local Ollama HTTP
  service behind the existing `Embedder` protocol seam. Lazy SDK
  client, native 1024-d output L2-normalised so downstream cosine
  matches the heuristic `HashEmbedder`. Empty input maps to the
  zero vector. Seven integration tests skip cleanly when the
  `ollama` SDK is absent or no service is reachable on
  `127.0.0.1:11434`. Lives in `ctrldoc.ingest.embedder_ollama` so
  the dependency-free heuristic ref stays Ollama-free.

## [0.1.2] — 2026-05-23

Production NLI backend landing behind the v0.1.0 protocol seam.

### Added

- **`DeBERTaNLIChecker` (S-051b)** — `cross-encoder/nli-deberta-v3-large`
  behind the existing `NLIChecker` protocol seam. Lazy model load;
  softmax-normalised `(contradiction, entailment, neutral)` head
  with the model's `id2label` validated against the spec's
  three-label vocabulary at load time. Integration tests skip when
  `transformers` is absent and pull the model on first run
  (~750 MB cached afterward). Lives in `ctrldoc.verify.nli_deberta`
  so the heuristic ref in `ctrldoc.verify.nli` stays torch-free.

## [0.1.1] — 2026-05-23

Production backend landings behind v0.1.0 protocol seams. The
substrate is unchanged; the v0.1.0 heuristic references remain in
place as deterministic baselines, and production wirings opt in
per call site.

### Added

- **`BGEReranker` (S-043b)** — `BAAI/bge-reranker-v2-m3` cross-encoder
  behind the existing `Reranker` protocol seam. Lazy model load,
  joint `(query, candidate.text)` scoring, descending-score
  truncation to `k`, deterministic ties on input order. Integration
  tests skip when `transformers` is absent and pull the model
  on first run (~500 MB cached afterward). Lives in
  `ctrldoc.retrieval.reranker_bge` so the heuristic refs in
  `ctrldoc.retrieval.reranker` stay torch-free.

## [0.1.0] — 2026-05-23

First tagged release. The MVP substrate is in place; LLM-backed
backends ship behind protocol seams that production wirings plug
into without touching playbook code.

### Added

**L0 — Ingest.** Markdown / PDF / Python parsers, identity coreference,
GLiNER NER with canonicalisation, semantic chunker that never splits
mid-sentence, deterministic `HashEmbedder` reference, section
summariser (heuristic + Anthropic backend), and an `ingest_document`
pipeline with incremental re-ingest.

**L1 — Multi-view index.** `Store` protocol with in-memory and
SQLite reference implementations; entity inverted index; `Tantivy`
BM25; pure-Python cosine vector index (`sqlite-vec` queued); skeleton
+ glossary assembler producing the cacheable prefix; `PRAGMA
integrity_check` + backup-before-destructive-op safety.

**L2 — Retrieval.** Retrieval DSL (`search` / `expand` / `neighbors`)
with a discriminated-union schema, executor that fuses across views,
Reciprocal Rank Fusion with `k=60`, reranker protocol with heuristic
references, evidence-pack builder honouring the `≤6k` token cap, and
a planner with cache-controlled Anthropic backend.

**L3 — Verifier.** Claim decomposer (heuristic + Anthropic backend),
NLI checker, two-tier LLM-judge with tier-2 escalation, `ClaimVerifier`
with refusal logic and a one-pass broad-depth repair, and the §8.6
family-9 calibration suite (FP ≤ 2%, FN ≤ 5%).

**L4 — Orchestrator.** Stateless task primitive (one fresh API call
per sub-task), Anthropic prompt-cache wrapper with `cache_control`
on the prefix, tiered routing (`local` vs `opus`), batched task
runner for shared-evidence fan-out, semaphore-bounded async
concurrency, streaming progress events, resumability checkpoints,
synthesis primitive (one-shot reduce over structured findings).

**L5 — Playbooks.** All six UC playbooks (`qa`, `coverage_audit`,
`quality_audit`, `analytical_review`, `anomaly_scan`,
`relation_map`) with deterministic stubs + protocol seams for
production LLM wiring.

**Eval & hardening (§8).** Per-playbook eval-set runners (`qa_eval`,
`qa_refusal`, `coverage_eval`, `quality_eval`, `analytical_eval`,
`anomaly_eval`, `relation_eval`) with §8.2 threshold gates; family-8
adversarial detectors + invariants (homoglyphs, zero-width, bidi
override, prompt-injection); family-10 determinism (byte-identical
re-ingest, snapshot anchors); family-11 cost/latency baselines
matching the §8.4 table; §8.7 LLM-as-judge with bias controls
(rubric, A/B swap, Cohen's κ, drift detection); §8.6 cross-cutting
continuous canary with sha256 signature pinning.

**CLI & docs (§6, §12).** `ctrldoc` typer CLI with six subcommands;
`python -m ctrldoc` entry point; six runnable per-playbook examples
in `examples/`; verified Quickstart in `README.md`.

### Known limitations (queued)

- `S-022b` sqlite-vec extension wiring — blocked on a Python build
  with `--enable-loadable-sqlite-extensions`.
- `S-034b` fastcoref backend — blocked on upstream incompatibility
  with the current `transformers` release.
- `S-036b` BGE-M3 via Ollama — blocked on a local Ollama service.
- `S-043b` BGE-reranker-v2-m3 — queued (~500 MB HF cross-encoder).
- `S-051b` deberta-v3-large-mnli — queued (~750 MB HF model).
- `S-052b` Qwen2.5-7B via Ollama — queued (Ollama not running).

All six queued slices have a deterministic heuristic reference
in place; production wirings plug in behind the existing protocol
seams without touching playbook code.

## [0.0.0] — Pre-release

Project initialized.
