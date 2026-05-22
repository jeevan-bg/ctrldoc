# ctrldoc — Architecture

This document is an opinionated walkthrough of the system. For the normative specification, see [SPEC.md](SPEC.md).

## The one principle

The LLM never sees the raw document. It sees only:

- retrieved spans (mediated by L1 + L2),
- structured findings (produced by other LLM tasks),
- or distilled state (provenance, glossary, skeleton).

Every other property of the system — no rot, no drift, no dilution, no hallucination — follows from this principle plus stateless per-task execution.

## Layers

```
┌──────────────────────────────────────────────────────────┐
│  CLI / Python API                                        │
├──────────────────────────────────────────────────────────┤
│  L5  Playbooks: qa, coverage_audit, quality_audit,       │
│      analytical_review, anomaly_scan, relation_map       │
├──────────────────────────────────────────────────────────┤
│  L4  Orchestrator: stateless tasks, structured output,   │
│      batching, tiered routing, prompt caching, synthesis │
├──────────────────────────────────────────────────────────┤
│  L3  Verifier: claim decomposition, NLI, LLM-judge,      │
│      refuse-on-failure                                   │
├──────────────────────────────────────────────────────────┤
│  L2  Retrieval: planner, fusion, reranker, evidence pack │
├──────────────────────────────────────────────────────────┤
│  L1  Multi-View Index: tree, dense vectors, BM25,        │
│      entity index, skeleton                              │
├──────────────────────────────────────────────────────────┤
│  L0  Ingest: parse, coref, NER, semantic chunk, embed    │
└──────────────────────────────────────────────────────────┘
```

Each layer has a stable contract (Pydantic models defined in `src/ctrldoc/models.py`). Layers can be reimplemented in isolation as long as the contracts hold.

## The two pillars

**Pillar 1 — Stateless tasks.** Every sub-task is an independent API call with a fresh context window: `{system_prompt, doc_skeleton, entity_glossary, evidence_pack, task_input}`. Outputs are JSON validated against a schema. The orchestrator collects JSON results and feeds only the *distilled findings* to a final synthesis call — never raw documents, never prior reasoning.

**Pillar 2 — Shared prompt cache.** Every sub-task in a session begins with the same cacheable prefix (`system_prompt + skeleton + glossary`). Anthropic's prompt cache keys on content, not session, so N sub-tasks share the same cache entry. Fresh sessions become nearly free on the prefix.

Together: fresh contexts (isolation) at the cost of only the small task-specific tail (economy).

## Storage

SQLite, single file per document collection:

- `chunks(id, section_id, text, char_start, char_end, token_count, embedding_id, hash)`
- `sections(id, parent_id, title, summary)`
- `entities(id, type, aliases_json)`
- `entity_mentions(entity_id, chunk_id, char_start, char_end)`
- `embeddings(...)` via [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
- BM25 index via [`tantivy`](https://github.com/quickwit-oss/tantivy)

The storage layer hides behind a `Store` trait in `src/ctrldoc/store/`. Switching to Qdrant or FalkorDB is a config change, not a rewrite.

## Provenance

Every output carries a `Provenance` record: run ID, timestamp, playbook + version, schema version, index hash, model identifiers, tokenizer name. This is what makes reproducible audit trails possible.

## What the system explicitly does *not* do (MVP)

- Multi-document analysis beyond pairwise audit.
- Multilingual processing.
- Image/figure/diagram understanding.
- Live collaboration.
- Hosted SaaS.

See [SPEC.md](SPEC.md) §11 for the complete out-of-scope list.
