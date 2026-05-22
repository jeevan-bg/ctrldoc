# ctrldoc — MVP Spec

*A local-first substrate for analyzing arbitrarily large documents with citation-grounded, hallucination-bounded outputs.*

---

## 1. What It Is

A local-first document analysis substrate that lets an LLM analyze arbitrarily large documents (10k–100k+ words) with **citation-grounded, hallucination-bounded, drift-free** outputs. Built on one principle: **the LLM never sees the raw document.** It only sees retrieved spans, structured findings, or distilled state. All raw content is mediated through a multi-view index + verification layer + stateless per-task orchestration.

Frontier LLM (Claude Opus) does the hard reasoning. Local models (via Ollama) do the cheap, repetitive work. Prompt caching minimizes token burn.

---

## 2. Use Cases (MVP must satisfy all five)

| # | Use Case | Example |
|---|---|---|
| **UC1** | Trustworthy QA on large doc | "What does §4.2 imply about fault tolerance?" |
| **UC2** | Coverage audit (doc vs. doc) | "Does the spec satisfy every threat in the threat model?" |
| **UC3** | Quality audit (doc vs. criteria) | "Is this a good L0 kernel spec?" |
| **UC4** | Analytical review (open-ended) | "Find strengths, weaknesses, loopholes in this 10k-line doc." |
| **UC5** | Anomaly surfacing | "Surface suspicious patterns for human triage." |
| **UC6** | Concept relation mapping | "How do concepts X, Y, Z relate across this doc?" |

All six reuse the same substrate; each is a thin orchestration playbook on top.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CLI / Python API / (later) MCP server                      │
├─────────────────────────────────────────────────────────────┤
│  L5  Playbooks (one per use case)                           │
│      qa · coverage_audit · quality_audit ·                  │
│      analytical_review · anomaly_scan · relation_map        │
├─────────────────────────────────────────────────────────────┤
│  L4  Orchestrator                                           │
│      stateless per-task calls · structured outputs ·        │
│      batching · tiered LLM routing · synthesis              │
├─────────────────────────────────────────────────────────────┤
│  L3  Verifier                                               │
│      claim decomposition · NLI entailment · LLM-judge ·     │
│      refusal-on-failure                                     │
├─────────────────────────────────────────────────────────────┤
│  L2  Retrieval                                              │
│      planner (LLM) · multi-view fusion · reranker ·         │
│      evidence pack builder (≤6k tokens)                     │
├─────────────────────────────────────────────────────────────┤
│  L1  Multi-View Index                                       │
│      structural tree · dense vectors · BM25 ·               │
│      entity index · section summaries (skeleton)            │
├─────────────────────────────────────────────────────────────┤
│  L0  Ingest                                                 │
│      parse · coref · NER · semantic chunk · embed · index   │
└─────────────────────────────────────────────────────────────┘
```

**Storage:** SQLite (vectors via `sqlite-vec`, BM25 via Tantivy, metadata in tables). Single file per document collection. No server.

### 3.1 Two Architectural Pillars

The system's guarantees (no rot, no dilution, no drift, bounded cost) come from two pillars working together. Everything else is implementation.

**Pillar 1 — Stateless Tasks (the "loop").**
Every sub-task (per-item judge, per-ctrldoc sweep, per-claim verify) is an **independent API call with fresh context**. No accumulation across tasks. Each call sees only what it needs: `{system_prompt, doc_skeleton, entity_glossary, evidence_pack, task_input}`. Output is structured JSON. The orchestrator collects JSON results and feeds only the *distilled findings* to a synthesis call — never raw doc, never prior reasoning. This is what eliminates context rot and drift by construction.

**Pillar 2 — Shared Prompt Cache.**
The cacheable prefix `{system_prompt, doc_skeleton, entity_glossary}` is identical across every sub-task in a session. Anthropic's prompt cache keys on content prefix, not session ID — so N fresh sub-tasks hit the same cache N times. Effective cost of the prefix: ~10% of nominal after first call.

**Why they combine well, not badly.**
Naively, "fresh sessions" sounds expensive (no context reuse). With prefix caching, fresh sessions are nearly free on the prefix and only pay full cost on the small task-specific tail. You get isolation *and* economy. This is the unlock that makes large-doc analysis affordable.

Practical rule: design every playbook as a *map* of stateless tasks followed by a single *reduce* (synthesis) over their structured outputs. Never carry context across tasks.

---

## 4. Components

### 4.0 Data Model

Pydantic models. Every layer reads/writes these. Stable contracts between layers — change these only with a `schema_version` bump.

```python
class Chunk:
    id: str                     # content-hashed, stable
    section_id: str
    text: str
    token_count: int
    char_start: int; char_end: int
    embedding_id: str
    metadata: dict

class Section:
    id: str                     # native ID (§3.2.1, function_name)
    parent_id: str | None
    title: str
    summary: str                # 1–2 sentences (skeleton)
    chunk_ids: list[str]

class Entity:
    id: str                     # canonical
    aliases: list[str]
    type: str                   # person | concept | system | ...
    mention_chunk_ids: list[str]

class Span:
    chunk_id: str
    char_start: int; char_end: int
    text: str                   # exact extracted span

class EvidencePack:
    query: str
    spans: list[Span]
    token_count: int            # ≤ 6000
    retrieval_plan: list[str]   # DSL trace

class Claim:
    text: str
    citations: list[Span]
    verified: bool
    confidence: float
    nli_score: float
    judge_score: float

class Verdict:                  # coverage_audit / quality_audit
    item_id: str
    verdict: Literal["Covered", "Partial", "NotCovered", "Ambiguous"]
    citations: list[Span]
    confidence: float

class Finding:                  # analytical_review / anomaly_scan
    ctrldoc: str
    location: Span
    claim: str
    severity: Literal["info", "warn", "critical"]

class RelationEdge:             # relation_map
    src_concept: str; dst_concept: str
    type: Literal["depends_on", "contradicts", "refines",
                  "instantiates", "conflicts_with",
                  "prerequisite_of", "alternative_to"]
    citations: list[Span]
    confidence: float

class Provenance:
    run_id: str
    timestamp: str
    playbook: str; playbook_version: str
    schema_version: str
    index_hash: str
    models: dict[str, str]      # {"planner": "...", "judge_tier1": "...", ...}
    tokenizer: str

class PlaybookOutput:
    provenance: Provenance
    result: Any                 # playbook-specific payload
```

### 4.1 Ingest (L0)

Input: a file (PDF, MD, TXT, code).
Output: persisted multi-view index.

Pipeline:
1. **Parse** to structural tree. `pymupdf4llm` for PDF, native parser for MD, `tree-sitter` for code. Preserve native IDs (§3.2.1, function names, line ranges).
2. **Coreference resolution** with `fastcoref`. Replace pronouns with canonical IDs.
3. **NER** with GLiNER (zero-shot). Canonicalize entities with an LLM pass; build entity glossary.
4. **Semantic chunking.** Leaf chunks ≤512 tokens. Never split mid-sentence/function/table. Stable chunk IDs.
5. **Embed** chunks with `BGE-M3` (dense + sparse in one model) via Ollama.
6. **BM25 index** with Tantivy.
7. **Section summaries.** One LLM pass: 1–2 sentences per section. Concatenate → `doc_skeleton` (~1–3k tokens for 100k words).
8. **Persist** to SQLite: chunks, embeddings, BM25 index, entity glossary, skeleton.

Idempotent. Cacheable. Runs once per document.

### 4.2 Multi-View Index (L1)

Four addressable views over the same doc:

| View | Purpose | Backend |
|---|---|---|
| Structural tree | Hierarchy queries, navigation | SQLite table (parent_id, section_id) |
| Dense vectors | Semantic similarity | `sqlite-vec` |
| BM25 lexical | Exact terms, numbers, proper nouns | Tantivy |
| Entity index | Entity-centric queries | SQLite inverted index |

Skeleton + entity glossary are always cheap to load (~3–5k tokens total).

### 4.3 Retrieval (L2)

Per query:
1. **Planner LLM** sees `{skeleton, entity_glossary, query}`. Emits a plan in a small DSL:
   ```
   search(query, view=dense|lexical|entity, k=N)
   expand(section_id)
   neighbors(entity_id, hops=1)
   ```
2. **Executor** runs plan. Fuses results across views with Reciprocal Rank Fusion.
3. **Reranker** (`BGE-reranker-v2-m3`) reduces top-50 → top-8.
4. **Evidence pack builder** assembles ≤6k tokens with stable span IDs.

The planner never sees raw chunks. The downstream judge sees only the evidence pack.

### 4.4 Verifier (L3)

Per generated answer:
1. **Decompose** into atomic claims via constrained-JSON output.
2. For each claim, **independent retrieval** against the index (don't trust generator's citations).
3. **NLI check** (`deberta-v3-large-mnli`) + **LLM-as-judge** (tier-1: local Qwen2.5-7B; escalate to Opus if disagree).
4. Both must pass. Failures → one repair pass with broader retrieval; if still failing → drop claim or refuse.

Output: `{answer, claims: [{text, citations, verified: bool, confidence}]}`.

### 4.5 Orchestrator (L4)

Stateless task primitive:
```python
def task(prompt, evidence_pack, schema, llm="local"|"opus") -> structured_output
```

Each task = fresh API call. No accumulated context across tasks. Structured outputs only (JSON, validated against schema).

**Tiered routing:**
- Local 7B for: simple judging, claim decomposition, easy ctrldoc scans.
- Opus for: planning, hard judges (escalated), final synthesis.

**Batching:** items sharing evidence packs → one Opus call covers many.

**Prompt caching:** every Opus call prefixes with `{system_prompt, doc_skeleton, entity_glossary}`. Cached once per session. Subsequent calls pay ~10% input cost.

### 4.6 Playbooks (L5)

Each playbook is ~200–500 LOC orchestrating L2–L4. See Section 5.

### 4.7 Cross-Cutting Concerns

Mandatory operational primitives. All playbooks depend on these.

**Tokenizer.** `tiktoken cl100k_base` everywhere — chunking, budget accounting, cache-prefix sizing. Single source of truth.

**Versioning.** Three independent versions tracked in every index:
- `schema_version` (data model)
- `index_version` (storage format)
- `embedding_model_version` (BGE-M3 weights hash)

On mismatch at index open: fail fast with a clear "re-ingest required" error. No silent migrations.

**Configuration.** Single `ctrldoc.toml` per project:
```toml
[models]
planner = "claude-opus-4-7"
judge_tier1 = "qwen2.5:7b-instruct-q4_K_M"
judge_tier2 = "claude-opus-4-7"
verifier_nli = "deberta-v3-large-mnli"
embedder = "bge-m3"

[budgets]
max_cost_usd = 20.0
max_tokens_per_call = 16000
max_wall_clock_min = 30

[concurrency]
anthropic_concurrent = 8
ollama_concurrent = 2

[paths]
index_path = "./ctrldoc.db"
runs_path  = "./runs/"
traces_path = "./traces/"
```
Secrets via env vars only (`ANTHROPIC_API_KEY`). Never in `ctrldoc.toml`.

**Observability / tracing.** Every LLM call writes a JSONL trace record:
```
{run_id, task_id, playbook, model, prompt_hash, response_hash,
 tokens_in, tokens_out, cost_usd, latency_ms, cache_hit, error}
```
Stored at `traces/{run_id}.jsonl`. Replayable: `ctrldoc replay <run_id>` reproduces every call (cached responses) for debugging.

**Cost / budget guard.** Hard kill switch. Orchestrator tracks running cost; aborts cleanly on `max_cost_usd` overrun. Default: 2× expected per playbook (from §9). No silent overspend possible.

**Resumability.** Every playbook checkpoints structured results to `runs/{run_id}/state.json` after each completed sub-task. `ctrldoc resume <run_id>` picks up from the last successful checkpoint. Crash at item 47 of 100 → resume from 48, not 1.

**Streaming progress.** Orchestrator emits progress events to stdout (and optional callback):
```
{event: "task_started"|"completed"|"failed", task_id, progress: "N/M", eta_seconds, cost_so_far_usd}
```
Never block silently longer than 5s.

**Concurrency policy.**
- Anthropic API: semaphore = `min(tier_rpm/60, anthropic_concurrent)`.
- Ollama (M-series MacBook): 1 concurrent 7B inference; 2 concurrent embedder/reranker.
- Async parallelism across sub-tasks always; never serialize what can fan out.

**Index integrity.**
- `PRAGMA integrity_check` at every index open.
- Content hash stored per chunk; verified on read.
- Backup snapshot (`index.db.bak`) before any destructive operation (re-ingest, migration).

**Human-in-the-loop checkpoints.** Playbooks needing review (criteria approval, checklist edit, concept-list edit) emit a `review.json`:
```
{playbook, stage, items: [{id, content, action: "approve"|"edit"|"reject"|"pending"}]}
```
User edits with any text editor; `ctrldoc resume` picks up changes. No GUI required for MVP.

**Provenance on every output.** Every `PlaybookOutput` carries a `Provenance` (§4.0). Required for reproducibility and audit trail. Never returned without it.

---

## 5. The Five Playbooks

### 5.1 `qa(query)` — UC1: Trustworthy QA

```
1. retrieve(query) → evidence_pack
2. generate_answer(query, evidence_pack) → answer_with_citations
3. verify(answer) → {verified_claims, refused_claims}
4. return AnswerReport
```

### 5.2 `coverage_audit(checklist_doc, target_doc)` — UC2

```
1. extract_checklist(checklist_doc) → [item_1, ..., item_N]  (one-time, human-reviewable)
2. cluster_items_by_topic(items) → clusters                  (saves LLM calls)
3. for each cluster (parallel, fresh context per cluster):
     evidence = retrieve_for_cluster(target_doc)
     verdicts = judge_cluster_items(items_in_cluster, evidence)  # batched
     verify_each(verdicts)
4. aggregate → CoverageReport {item, verdict ∈ {Covered, Partial, NotCovered, Ambiguous}, citations, confidence}
```

Inverted loop variant (auto-selected when items >> sections): for each section, identify which items it addresses. Used when N_items > 500.

### 5.3 `quality_audit(target_doc, doc_type)` — UC3

```
1. generate_criteria(doc_type) → checklist  (LLM enumerates, human approves)
2. (delegate to) coverage_audit(checklist, target_doc)
```

Same as UC2 once the checklist exists. The only added step is criteria generation + approval.

### 5.4 `analytical_review(target_doc, ctrldoces=auto)` — UC4

```
1. generate_ctrldoces(doc_type) → [assumptions, boundary_cases, consistency,
                                 ambiguity, scope_gaps, ...]      (~20–40)
2. for each ctrldoc (parallel, fresh context):
     findings = sweep_doc_with_ctrldoc(target_doc, ctrldoc)
       → bounded windows (8k tokens, 1k overlap)
       → extract {claim, location, severity, citation}
3. claims = flatten(findings)
4. cluster_claims_by_topic(claims)
5. pairwise_check(claims_within_and_across_clusters) → contradictions, gaps
6. synthesize_report(structured_findings) → ReviewReport
```

Synthesis sees structured findings JSON (~5–20k tokens), never raw doc.

### 5.5 `anomaly_scan(target_doc)` — UC5

Six detectors, each a deterministic or small-LLM pass:

| Detector | Mechanism |
|---|---|
| Hedge-word | Regex on `usually|typically|should|may` near security/correctness terms |
| Asymmetry | Token-count by topic cluster; flag >5× variance on comparable concerns |
| Justification gap | LLM pass: claim without rationale/citation/derivation |
| Undefined terms | Term frequency analysis; flag terms used but not defined |
| Boundary silence | LLM pass per section: "Are edge cases addressed?" |
| Embedding outlier | Chunks with low cosine to neighbors |

Output: ranked `AnomalyQueue` with severity, location, reason. Human triages.

### 5.6 `relation_map(target_doc, concepts=auto)` — UC6

```
1. extract_concepts(target_doc) → [c_1, ..., c_M]
     → reuses entity index; LLM pass to add abstract concepts beyond named entities
     → human can edit/extend the list
2. for each concept pair (c_i, c_j) (parallel, fresh context):
     evidence = retrieve_co_occurrence(c_i, c_j)
     if evidence is empty: skip (no relation in doc)
     relation = classify_relation(c_i, c_j, evidence)
       → one of: depends_on, contradicts, refines, instantiates,
                 conflicts_with, prerequisite_of, alternative_to, unrelated
     verify(relation, evidence)
3. aggregate → RelationGraph {nodes: concepts, edges: [{src, dst, type, citations, confidence}]}
4. (optional) synthesize_summary(graph) → narrative description of key relationships
```

Output: structured graph (JSON) + optional narrative. Reuses entity index, retrieval, and verifier — no new substrate. Cap pair enumeration with a topic-cluster prefilter when M > 30 to avoid O(M²) blowup.

---

## 6. Tech Stack

| Layer | Tech | Notes |
|---|---|---|
| Language | Python 3.11+ | Move hot paths to Rust post-MVP |
| Parsing | `pymupdf4llm`, `tree-sitter`, native MD parser | |
| Coref | `fastcoref` | |
| NER | GLiNER | Zero-shot, no training data |
| Embeddings | `BGE-M3` via Ollama | Dense + sparse |
| Vector store | `sqlite-vec` | Single-file, no server |
| BM25 | Tantivy (Python bindings) | |
| Reranker | `BGE-reranker-v2-m3` via Ollama or HF | |
| Local LLM | Qwen2.5-7B-Instruct (Q4_K_M) via Ollama | Tier-1 judge |
| NLI | `deberta-v3-large-mnli` via MLX (Mac) | Fast on M-series |
| Cloud LLM | Claude Opus via Anthropic SDK | Hard reasoning + synthesis |
| Caching | Anthropic prompt caching | `cache_control` on skeleton + glossary |
| Schema | Pydantic + JSON Schema | Structured outputs |
| CLI | Typer | |
| Orchestration | Plain Python `asyncio` | No LangChain/LlamaIndex |

**MacBook target:** M-series, 16GB+ RAM. All local models run quantized via Ollama. Verified to run on a single laptop.

---

## 7. Build Plan (8 weeks, solo or 2-person)

| Week | Deliverable | Pass Criteria |
|---|---|---|
| **1** | **Eval harness first.** 200–500 labeled examples across UC1–UC6 on a sample doc. | Harness runs; baseline scores recorded |
| **2** | L0 ingest + L1 multi-view index | Sample doc indexed; queries hit chunks in <50ms |
| **3** | L2 retrieval + planner | Top-k recall@10 ≥ 90% on eval |
| **4** | L3 verifier + refusal logic | ≤2% unsupported-claim rate; refusal triggers correctly on out-of-doc queries |
| **5** | Playbook 5.1 (`qa`) + 5.2 (`coverage_audit`) | Both pass their eval subsets |
| **6** | Playbook 5.3 (`quality_audit`) + 5.4 (`analytical_review`) + 5.5 (`anomaly_scan`) + 5.6 (`relation_map`) | All six playbooks green on eval |
| **7** | Cost optimization: prompt caching, batching, tiered routing | Cost ≤ targets in §9; latency ≤ targets |
| **8** | Hardening: adversarial eval, regression suite, docs | All gates green; ready for real-world test |

**LOC estimate:** ~8–12k Python.

---

## 8. Testing Strategy

### 8.1 Eval Sets (build in Week 1, before any code)

| Set | Size | Purpose |
|---|---|---|
| `qa_eval` | 100 (q, gold_answer, gold_spans) | UC1 — single/multi-hop QA |
| `qa_refusal` | 30 (q where answer not in doc) | UC1 — must refuse, not fabricate |
| `coverage_eval` | 5 (checklist, doc) pairs with gold verdicts per item | UC2 |
| `quality_eval` | 3 docs with expert-rated quality reports | UC3 |
| `analytical_eval` | 3 docs with known seeded weaknesses | UC4 — recall on known issues |
| `anomaly_eval` | 3 docs with seeded anomalies | UC5 |
| `relation_eval` | 3 docs with gold concept-pair relations | UC6 |

### 8.2 Per-Playbook Metrics

| Playbook | Metric | Target |
|---|---|---|
| `qa` | Citation precision, refusal accuracy | ≥95% precision, ≥90% correct refusals |
| `coverage_audit` | Per-item verdict accuracy vs. gold | ≥90% |
| `quality_audit` | Criteria coverage vs. expert checklist | ≥85% |
| `analytical_review` | Recall on seeded weaknesses | ≥80% |
| `anomaly_scan` | Precision on triage queue (% real anomalies) | ≥60% (recall matters more) |
| `relation_map` | Relation-type accuracy on gold pairs | ≥80% |

### 8.3 Property Tests (must always pass)

1. **No context rot:** assert no LLM call exceeds 16k tokens of input (excluding cache prefix).
2. **No drift:** run same query 5× → ≥80% citation overlap.
3. **No fabrication:** every claim in output has a citation that survives independent re-verification.
4. **Refusal works:** seeded out-of-doc questions → 100% refusal rate.
5. **Determinism (seeded):** with `temperature=0` and fixed seed, outputs reproducible within tolerance.

### 8.4 Cost / Latency Benchmarks

Per playbook, on a 10k-line sample spec:

| Playbook | Cost | Wall-clock |
|---|---|---|
| `qa` | <$0.10 | <30s |
| `coverage_audit` (100 items) | <$5 | <5 min |
| `quality_audit` (auto criteria, 50 items) | <$3 | <3 min |
| `analytical_review` (30 ctrldoces) | <$10 | <10 min |
| `anomaly_scan` | <$2 | <2 min |

Track in CI; fail build if regression > 20%.

### 8.5 Adversarial Tests

- Inject contradictory chunks → verifier must surface contradiction, not pick one.
- Inject near-duplicate-but-wrong chunks → reranker + verifier must reject.
- Prompt-injection in source document → system prompt must hold; refuse to follow.
- Out-of-doc trivia questions → 100% refusal.

### 8.6 Full Test Family Taxonomy

Every test in the suite must fall into one of these 14 families. If you have a test that doesn't fit, either the test is wrong or this list needs extending.

| # | Family | Catches | Minimum tests |
|---|---|---|---|
| 1 | **Ingest completeness** | Lost chunks, broken parsing | Token round-trip ≥98%; every section → ≥1 chunk; no orphan chunks; re-parse determinism (hash match) |
| 2 | **Needle-in-haystack (NIAH)** | Retrieval blind spots | Inject sentinels at start/mid/end/random; 100% retrieval @ top-5 |
| 3 | **Synthetic ground-truth doc** | Playbook correctness | One hand-built 5–10k-token doc with seeded sections, entities, contradictions, relations, gaps; run all 6 playbooks; compare to gold |
| 4 | **Reachability invariant** | Orphaned content | ctrldoc-per-section analytical_review → every section appears in findings ≥1× |
| 5 | **Negative / refusal** | Fabrication | 30 out-of-doc questions → 100% refusal; contradictory chunks → both surfaced, not picked |
| 6 | **Referential integrity** | Hallucinated citations | Every cited span ID exists, contains cited text, and survives independent re-retrieval |
| 7 | **Robustness / edge inputs** | Parser/chunker failures | Empty doc, single-chunk doc, no-structure doc, no-entity doc, all-duplicate doc, oversized sentence (>512 tokens), malformed MD, broken PDF, mixed languages, tables/code/math blocks |
| 8 | **Adversarial / security** | Source-doc attacks | Prompt-injection in source, homoglyphs, zero-width chars, RTL overrides, adversarial paraphrase, jailbreak strings in source |
| 9 | **Verifier calibration** | Verifier misjudgment | FP rate <2%; FN rate <5%; confidence bins match accuracy; tier-1 vs tier-2 agreement tracked |
| 10 | **Determinism / reproducibility** | Silent drift | `temperature=0` + fixed seed → byte-identical output; re-ingest → byte-identical index; snapshot tests on canonical outputs |
| 11 | **Performance / cost gates** | Token/latency/RAM regressions | Per-playbook latency budget; per-playbook cost budget; per-call ≤16k input tokens; prompt-cache hit rate ≥90%; peak RAM ≤8GB |
| 12 | **Failure-mode / resilience** | Crashes, corruption, data loss | Malformed LLM JSON → recover; Ollama down → graceful; API rate-limit → backoff; mid-ingest crash → resumable; OOM/disk-full → clean failure |
| 13 | **Incremental update / freshness** | Stale or broken state | Edit one section → only affected chunks re-indexed; add/remove section → glossary + skeleton updated; old citations flagged stale |
| 14 | **Concurrency** | Race conditions | Parallel sub-tasks have no shared mutable state; SQLite read/write races; async cancellation cleans up subprocesses |

**Cross-cutting practices (apply to all 14 families):**
- **Property-based testing** via Hypothesis on chunkers, mergers, verifiers, retrievers.
- **Regression dashboard** — every metric tracked per commit; trends visible; auto-alert on >5% drop.
- **Continuous canary** — one production-size doc (~10k lines) run through all 6 playbooks on every commit; baseline outputs committed; CI fails if drift >10%.

**Coverage rule:** before any playbook ships, it must have ≥1 test from every applicable family. No exceptions.

### 8.7 LLM-as-Judge Eval (for subjective outputs)

UC4 (`analytical_review`) and UC5 (`anomaly_scan`) produce partly subjective outputs that can't be fully scored against gold. Use LLM-judge with strict bias controls:

- **Judge model.** Frontier (Claude Opus). Must differ from the model that produced the output (no self-grading).
- **Rubric.** Structured prompt with 3–5 scoring dimensions, each 1–5 scale, with exemplars.
- **Bias controls.** Blind-shuffle output order; swap A/B positions; average over 3 runs with different seeds.
- **Inter-rater check.** 10% of items also human-scored; require judge-human agreement ≥0.7 (Cohen's κ).
- **Stability check.** Same output scored twice → score variance <0.5. Flag if higher.
- **Track judge drift.** Anchor outputs scored on every commit; alert on >0.5 score shift between runs.

---

## 9. Performance / Cost Targets

- **Ingest:** ~30s–2min per 10k-line doc (one-time, local).
- **Storage:** ~50–200MB per 10k-line doc indexed.
- **Query latency (UC1):** 5–30s end-to-end.
- **Audit latency (UC2/UC3, 100 items):** 2–8 min with parallelism.
- **Cost per audit:** $3–10 (tiered + cached) vs. ~$50+ naive.
- **MacBook RAM:** peak ≤8GB during inference.

---

## 10. Scalability

| Axis | MVP capacity | Scaling path |
|---|---|---|
| Doc size | 1M words | Already chunking-bounded; no architectural change |
| Doc count | ~100 docs / SQLite file | Migrate to Qdrant or FalkorDB (storage trait abstracted day 1) |
| Concurrent users | 1 (local CLI) | Wrap orchestrator in FastAPI; deploy as service |
| Playbook count | 5 | Plugin architecture (each playbook = directory with `playbook.py`) |
| Throughput | ~1 audit/min | Parallel sub-tasks already async; horizontal scale via job queue |

Storage layer is abstracted behind a trait/protocol from day 1 — SQLite → Qdrant is a config flip, not a rewrite.

---

## 11. Out of Scope (MVP)

- Multi-document cross-reference beyond pairwise audit (UC2)
- Multilingual documents
- Image/figure/diagram understanding (text-only MVP)
- Real-time collaboration / multi-user
- Hosted SaaS deployment
- IDE integration (CLI + Python API only; MCP server post-MVP)
- Write-time enforcement (read-time verification only)

---

## 12. Build Order — Day 1 Checklist

1. `pip install pymupdf4llm tree-sitter fastcoref gliner sqlite-vec tantivy-py ollama anthropic pydantic typer pytest`
2. `ollama pull bge-m3 qwen2.5:7b-instruct-q4_K_M`
3. Build eval harness (`tests/eval/`) with the 6 eval sets — labels first, code second.
4. Implement L0 ingest end-to-end on one sample doc.
5. Verify the index round-trip: chunk → embed → retrieve → expected span.
6. Move up the stack one layer per week.
7. **Never build a playbook before its eval subset exists.**

---

## 13. The Non-Negotiables

- **Eval harness exists before any code that it scores.**
- **No LLM call ever sees the raw full document.** Only skeleton, glossary, retrieved spans, or structured findings.
- **Every claim cited or refused.** No uncited prose in outputs.
- **Every playbook is stateless per task.** Fresh context per sub-call.
- **Storage layer is abstracted.** SQLite is the MVP backend, not the architecture.

If any of these slip, the product loses its core guarantee.
