# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-05-23

Minor version bump rolling up the six post-v0.1.0 production
backend landings (`0.1.1` … `0.1.6`) into a single release tag.

### Summary

Every protocol seam in the v0.1.0 substrate now has a real
production backend behind it:

| Seam | Backend | Slice | Release |
|---|---|---|---|
| `Reranker` (§4.3) | `BAAI/bge-reranker-v2-m3` | S-043b | 0.1.1 |
| `NLIChecker` (§4.4) | `cross-encoder/nli-deberta-v3-large` | S-051b | 0.1.2 |
| `Embedder` (§4.1) | `bge-m3` via Ollama | S-036b | 0.1.3 |
| `LLMJudge` (§4.4) | `qwen2.5:7b-instruct-q4_K_M` via Ollama | S-052b | 0.1.4 |
| `VectorIndex` (§4.2) | `sqlite-vec` (`vec0`, cosine) | S-022b | 0.1.5 |
| `CorefResolver` (§4.1) | `fastcoref` (LingMess) | S-034b | 0.1.6 |

Heuristic / dependency-free references stay in place as
behavioural oracles — production backends opt in per call site.

## [0.1.6] — 2026-05-23

Final production-backend landing — every v0.1.0 protocol seam now
has a real wiring behind it.

### Added

- **`FastCorefResolver` (S-034b)** — `fastcoref` (LingMess) behind
  the existing `CorefResolver` protocol seam. Lazy model load on
  first `resolve()` call; the canonical mention per cluster is
  picked as the longest span (ties broken by earliest position);
  non-canonical mentions are rewritten right-to-left so char
  offsets stay valid through the rewrite. Empty / whitespace-only
  / no-anaphora inputs short-circuit to a passthrough. Eight
  integration tests skip cleanly when `fastcoref` is absent and
  cover protocol conformance, pronoun resolution, multi-cluster
  rewrites, determinism, and length-non-decreasing.

## [0.1.5] — 2026-05-23

Production persistent vector index landing behind the v0.1.0
protocol seam.

### Added

- **`SqliteVecVectorIndex` (S-022b)** — `sqlite-vec` `vec0`
  virtual table with `distance_metric=cosine`, behind the existing
  `VectorIndex` protocol seam. A sidecar `id_map` table maps
  stable string `chunk_id` ↔ integer `rowid`. Cosine distance
  from sqlite-vec is converted to cosine similarity so the
  contract is identical to `InMemoryVectorIndex`. Ties resolve
  host-side by insertion order to match the reference. Fifteen
  integration tests skip cleanly when `sqlite-vec` is absent or
  Python lacks loadable-extension support; covers protocol
  conformance, dimension pin, idempotent add, remove (incl.
  unknown id no-op), search ordering / k-truncation / edge cases,
  and behavioural parity with `InMemoryVectorIndex` on a shared
  fixture.

## [0.1.4] — 2026-05-23

Production tier-1 LLM-judge landing behind the v0.1.0 protocol seam.

### Added

- **`OllamaLLMJudge` (S-052b)** — `qwen2.5:7b-instruct-q4_K_M` via
  a local Ollama HTTP service behind the existing `LLMJudge`
  protocol seam. Lazy SDK client; temperature pinned at 0 for
  determinism; markdown code fences stripped before JSON parsing;
  confidence clamped to `[0, 1]`; missing/non-numeric keys raise
  with a short message. Six integration tests skip cleanly when
  the SDK is absent or no Ollama service is reachable. Lives in
  `ctrldoc.verify.judge_ollama` so the heuristic ref stays
  Ollama-free. `EscalatingLLMJudge` (S-053) now has a real local
  tier-1 backend to wrap.

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
