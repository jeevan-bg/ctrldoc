# Architectural Decision Records

This file collects the significant architectural choices in `ctrldoc`. Each entry follows the [ADR format](https://adr.github.io/): Context, Decision, Consequences. Records are immutable once accepted; superseded records remain in place with a pointer to their replacement.

---

## ADR-0001 — Six-layer pipeline with a stable contract between each layer

**Status:** Accepted

**Context.** Large-document analysis demands two properties that pull in opposite directions: every sub-step must be replaceable as the field moves (embedders, rerankers, judges, parsers all churn fast), and the overall system must give end-to-end guarantees (no rot, no drift, citation-grounded outputs). A monolithic pipeline gives end-to-end guarantees but locks in component choices. A free-form composition gives flexibility but makes guarantees impossible to prove.

**Decision.** The system is a six-layer stack — Ingest (L0), Multi-view index (L1), Retrieval (L2), Verifier (L3), Orchestrator (L4), Playbooks (L5) — connected by stable Pydantic contracts (`Chunk`, `Section`, `Entity`, `Span`, `EvidencePack`, `Claim`, `Verdict`, `Finding`, `RelationEdge`, `PlaybookOutput`). Layers can be reimplemented in isolation provided contracts hold. A `schema_version` bump is required to change any contract.

**Consequences.**

- Each layer can be tested independently against its contract.
- Component swaps (e.g. a new reranker) become local changes.
- The compiler-style discipline of typed boundaries replaces ad-hoc integration.
- The overhead of defining and maintaining contracts is a real cost; we accept it.

---

## ADR-0002 — Stateless per-task execution with shared prompt prefix

**Status:** Accepted

**Context.** Naive approaches keep one long-running LLM session for an analysis: cheaper on token count (context is reused) but vulnerable to context rot, dilution, and drift. The alternative — a fresh API call per sub-task — eliminates these but appears prohibitively expensive.

**Decision.** Every sub-task is a fresh, stateless API call with input `{system_prompt, doc_skeleton, entity_glossary, evidence_pack, task_input}`. The first three components form a deterministic cacheable prefix shared across all sub-tasks in a session. Anthropic prompt caching keys on content, so the prefix cost is paid once per session, regardless of sub-task count.

**Consequences.**

- Context rot, dilution, and drift are eliminated by construction.
- Sub-task economics are dominated by the task-specific tail (small).
- All cross-task synthesis happens by reducing over structured outputs, never by reading raw context twice.
- The orchestrator must enforce cache discipline (prefix never varies within a session).

---

## ADR-0003 — SQLite as the MVP storage backend, behind a swappable trait

**Status:** Accepted

**Context.** The MVP must run on a single laptop, zero-install, no server. Production scale will eventually want a horizontally scalable vector store. Picking either now is a trap: SQLite is right for the laptop case but wrong for scale; a hosted vector DB is right for scale but wrong for the laptop.

**Decision.** A `Store` protocol in `src/ctrldoc/store/` abstracts persistence. The MVP ships a SQLite implementation (`sqlite-vec` for dense vectors, `tantivy` for BM25, plain SQL for metadata). Switching to Qdrant, FalkorDB, or any other backend is a configuration change, not a rewrite.

**Consequences.**

- MVP has zero infrastructure dependencies — `pip install` and a Python interpreter.
- Storage cost is bounded: one file per indexed collection.
- The trait carries an integration surface we must keep small and stable.

---

## ADR-0004 — Verifier independence: claims are re-retrieved, not trusted

**Status:** Accepted

**Context.** Generator–verifier pipelines often share retrieval state. When they do, a generator that retrieves the wrong evidence produces a wrong-but-self-consistent answer the verifier rubber-stamps.

**Decision.** The verifier performs *independent* retrieval against the index for every claim it checks. It does not consume the generator's citations as ground truth; it consumes them only as a hypothesis to falsify. Each claim is then checked by NLI entailment and an LLM judge; both must pass.

**Consequences.**

- Generator and verifier failure modes do not correlate.
- The verifier is more expensive per claim than a citation check; the cost is justified.
- The verifier can refuse the generator's answer; refusal is preferred over fabrication.

---

## ADR-0005 — Tokenizer as a single source of truth

**Status:** Accepted

**Context.** Several layers care about token counts: the chunker (≤512 tokens per chunk), the evidence pack builder (≤6000 tokens), the cache prefix sizer, and the budget guard. If they disagree by even a few percent, drift accumulates.

**Decision.** A single tokenizer (`tiktoken` `cl100k_base`) is used everywhere in the codebase. All layers import it from one location. Other tokenizers are forbidden in production code paths.

**Consequences.**

- Token accounting is internally consistent.
- A change to the tokenizer requires re-ingestion and is gated by `schema_version`.
- We pay a small accuracy cost on models that use other tokenizers internally; we accept it.

---

## ADR-0006 — Read-time verification only; no write-time enforcement

**Status:** Accepted

**Context.** Documents could be enforced to be self-consistent at write time (a linter that refuses to commit contradictions). This would be powerful but also constrains authoring and is out of scope for a substrate library.

**Decision.** `ctrldoc` analyzes existing documents at read time and reports findings. It does not block, alter, or enforce policy on the source document. All outputs are reports; the consumer decides what to do.

**Consequences.**

- The library composes with any authoring workflow.
- We cannot guarantee a document is free of issues, only that we found the ones we report.
- Write-time enforcement remains a possible future product surface.
