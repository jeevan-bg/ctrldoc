# ctrldoc — Specification Digest

A compact summary of `docs/SPEC.md`. Optimized for fast loading and stable content. For any detail, follow the SPEC-REF back to the full document.

## The one principle (§1)

The LLM never sees the raw document. It sees only retrieved spans, structured findings, or distilled state. Every guarantee follows from this.

## Use cases (§2)

| ID | Use case |
|---|---|
| UC1 | Trustworthy QA on large doc |
| UC2 | Coverage audit (doc vs. doc) |
| UC3 | Quality audit (doc vs. criteria) |
| UC4 | Open-ended analytical review |
| UC5 | Anomaly surfacing |
| UC6 | Concept-relation mapping |

## Architecture in six layers (§3)

| Layer | Purpose |
|---|---|
| L0 Ingest | parse, coref, NER, chunk, embed |
| L1 Multi-view index | structural tree, dense vectors, BM25, entity index, skeleton |
| L2 Retrieval | planner DSL, fusion, reranker, evidence pack (≤ 6k tokens) |
| L3 Verifier | claim decomposition, independent re-retrieval, NLI, LLM judge, refusal |
| L4 Orchestrator | stateless tasks, structured outputs, tiered routing, prompt caching, synthesis |
| L5 Playbooks | one per use case |

## Two pillars (§3.1)

- **Stateless tasks.** Every sub-task is a fresh API call with input `{system_prompt, doc_skeleton, entity_glossary, evidence_pack, task_input}` → structured JSON output. No accumulation. Synthesis happens by reducing over structured outputs, never raw doc.
- **Shared prompt cache.** The cacheable prefix `{system_prompt, doc_skeleton, entity_glossary}` is identical across every sub-task in a session. Fresh sessions become nearly free on the prefix.

## Data model contracts (§4.0)

Pydantic models: `Chunk`, `Section`, `Entity`, `Span`, `EvidencePack`, `Claim`, `Verdict`, `Finding`, `RelationEdge`, `Provenance`, `PlaybookOutput`. Changes require a `schema_version` bump.

## Storage (§3, §4.2)

SQLite by default: `sqlite-vec` for dense vectors, `tantivy` for BM25, plain SQL for metadata. The `Store` protocol abstracts persistence — switching backends is a config change.

## Cross-cutting (§4.7)

- Tokenizer: `tiktoken cl100k_base` everywhere.
- Versioning: `schema_version`, `index_version`, `embedding_model_version` — fail-fast on mismatch.
- Trace: every LLM call writes a JSONL record (`run_id, task_id, prompt_hash, response_hash, tokens, cost, latency, cache_hit`).
- Budget: hard kill switch at `max_cost_usd`.
- Resumability: checkpoint after every sub-task.
- Provenance: every output carries run id, models used, schema version, index hash.

## Playbooks (§5)

Each is 200–500 LOC orchestrating L2–L4. Map then reduce:

1. **qa** — retrieve → answer → verify.
2. **coverage_audit** — extract checklist, cluster items, judge each cluster against retrieved evidence.
3. **quality_audit** — generate criteria, delegate to coverage_audit.
4. **analytical_review** — generate lenses, sweep doc per lens, cluster claims, pairwise consistency check, synthesize.
5. **anomaly_scan** — six detectors (hedge-words, asymmetry, justification gap, undefined terms, boundary silence, embedding outlier).
6. **relation_map** — extract concepts, classify each pair, verify, aggregate to graph.

## Test families (§8.6)

14 families. Every test belongs to one. Coverage rule: a playbook ships only when every applicable family has ≥ 1 test.

| # | Family |
|---|---|
| 1 | Ingest completeness |
| 2 | Needle-in-haystack retrieval |
| 3 | Synthetic gold doc end-to-end |
| 4 | Reachability invariant |
| 5 | Negative / refusal |
| 6 | Referential integrity (citations) |
| 7 | Robustness / edge inputs |
| 8 | Adversarial / security |
| 9 | Verifier calibration |
| 10 | Determinism / reproducibility |
| 11 | Performance / cost gates |
| 12 | Failure resilience |
| 13 | Incremental update |
| 14 | Concurrency |

## Performance / cost targets (§9)

- Ingest: 30s – 2min per 10k-line doc.
- Storage: 50 – 200 MB per 10k-line doc.
- QA latency: 5 – 30s end-to-end.
- Audit latency (100 items): 2 – 8 min.
- Cost per audit: $3 – $10.
- MacBook RAM peak: ≤ 8 GB.

## The five non-negotiables (§13)

1. Eval harness exists before any code that it scores.
2. No LLM call ever sees the raw full document.
3. Every claim is cited or refused.
4. Every playbook is stateless per task.
5. Storage layer is abstracted; SQLite is the MVP backend, not the architecture.

If any of these slip, the product loses its core guarantee.
