# ctrldoc — Architecture

This document is an opinionated walkthrough of the v1 system. For the normative specification, see [SPEC.md](SPEC.md).

## The one principle

A document is a noisy observation of a latent ontology. Reading it = jointly inferring concepts, relations, and claims. Multiple documents = noisy observations of a shared ontology — comparing them is aligning them in that shared space.

This is the v1 generalisation of the v0.3 "the LLM never sees the raw document" principle, which still holds: the LLM sees only retrieved spans, structured findings, or distilled state.

## Layers

```
┌──────────────────────────────────────────────────────────┐
│  CLI  ·  Python API  ·  MCP server (stdio JSON-RPC)      │
├──────────────────────────────────────────────────────────┤
│  L6   Trace renderer: spans → claims → edges →           │
│       verdict + calibrated confidence                     │
├──────────────────────────────────────────────────────────┤
│  L5   Universal operations: one optimal-transport engine  │
│       drives compare / coverage / merge / list_check /    │
│       map / qa                                            │
├──────────────────────────────────────────────────────────┤
│  L4   Tool-using orchestrator: forced tool calls, no      │
│       free-form reasoning; append-only verdict ledger;    │
│       replay within ±0.02                                 │
├──────────────────────────────────────────────────────────┤
│  L3   Probabilistic edge inference: tuple logic + NLI +   │
│       LLM-judge + paraphrase voting + isotonic            │
│       calibration; shipped ECE per backend                │
├──────────────────────────────────────────────────────────┤
│  L2.5 Workspace: N docs share one Galois concept lattice; │
│       cross-doc edges (aligned_with / entails_across /    │
│       contradicts_across) are lazy, cached, linear in     │
│       |A| × k                                             │
├──────────────────────────────────────────────────────────┤
│  L2   Retrieval: planner, dense + BM25 + entity +         │
│       personalized PageRank over typed edges, reranker    │
├──────────────────────────────────────────────────────────┤
│  L1.5 Claim graph: universal claim tuple as the logic     │
│       floor (§6.2); span / claim / concept triplane        │
├──────────────────────────────────────────────────────────┤
│  L1   Multi-view index: tree + dense vectors + BM25 +     │
│       entity index + claims / concepts / typed_edges      │
├──────────────────────────────────────────────────────────┤
│  L0   Adaptive ingest: parse → coref → NER → chunk →      │
│       embed → index; per-doc schema co-induction          │
└──────────────────────────────────────────────────────────┘
```

Each layer has a stable contract (Pydantic models in `src/ctrldoc/models.py` for the v0.3 substrate and `src/ctrldoc/models_v1.py` for the v1 additions). Layers can be reimplemented in isolation as long as the contracts hold.

## The two pillars (preserved verbatim from v0.3)

**Pillar 1 — Stateless tasks.** Every sub-task is an independent API call with a fresh context window: `{system_prompt, doc_skeleton, entity_glossary, evidence_pack, task_input}`. Outputs are JSON validated against a schema. The orchestrator collects JSON results and feeds only the distilled findings to a final synthesis call — never raw documents, never prior reasoning.

**Pillar 2 — Shared prompt cache.** Every sub-task in a session begins with the same cacheable prefix (`system_prompt + skeleton + glossary`). Anthropic's prompt cache keys on content, not session, so N sub-tasks share the same cache entry. Fresh sessions become nearly free on the prefix.

Together: fresh contexts (isolation) at the cost of only the small task-specific tail (economy).

## The universal claim tuple

`Claim = (subject, predicate, object, polarity, modality, qualifier, span_refs, confidence)` per §6.2. This is the logic floor — every extractor (heuristic, Tier-2 SVO, LLM) emits this shape. Contradiction is a polarity flip; stronger-than is qualifier ordering (the §6.3 Galois lattice computes the partial order); the tuple is always extracted even when adapter-specific extraction fails. This is non-negotiable #6 in §13.

## Schema co-induction (per-doc adapters that emerge)

Per-document ontologies emerge via an EM loop (§6.4): the universal tuple is the floor; an LLM proposes typed nodes / edges from a fixed primitive library (10 element closed set: Entity, Event, Process, Property, Quantity, Definition, Assertion, Obligation, Citation, Relation); the extractor runs under the proposed schema; the residual rate is measured; the schema is re-induced if `unmatched_claim_rate > 0.20`. The final schema is cached as YAML alongside the document. There is no hardcoded per-doc-type code.

## Probabilistic edges and calibration

Every edge in the v1 graph carries a calibrated `confidence ∈ [0, 1]`. Sources are heuristic / NLI / LLM. The §6.5 calibration pipeline runs paraphrase voting (3–5 paraphrases of the hypothesis, agreement-rate correlates with correctness ρ ≥ 0.5) and isotonic regression to produce a calibrated probability. The release gate is `ECE ≤ 0.05` per backend on the held-out eval — non-negotiable #9.

## Optimal-transport core

`compare` / `coverage` / `merge` / `list_check` reduce to one min-cost-flow / Sinkhorn engine on the claim-pair edges weighted by `1 − NLI_entail`. Sinkhorn handles soft variants; min-cost flow handles exact hard-assignment variants and many-to-one transport. The `merge` operation respects a hard loss invariant — every input claim maps to exactly one output cluster — non-negotiable #10, asserted in CI.

## Workspace (L2.5)

A workspace is N documents sharing one canonical concept lattice. Cross-doc edges (`aligned_with`, `entails_across`, `contradicts_across`, `stronger_than`) are produced by the L3 inference pipeline against candidate-retrieval pairs only — linear in `|A| × k` per ordered doc pair, never quadratic. Every cross-doc edge cites a source span in *both* docs (non-negotiable #11).

## Storage

SQLite, single file per index. The v1 schema adds six tables to the v0.3 layout:

- v0.3: `chunks`, `sections`, `entities`, `entity_mentions`, `embeddings` (via [`sqlite-vec`](https://github.com/asg017/sqlite-vec)), BM25 (via [`tantivy`](https://github.com/quickwit-oss/tantivy)).
- v1: `claims`, `concepts`, `typed_edges`, `workspaces`, `cross_doc_edges`, `verdict_ledger`.

`SCHEMA_VERSION` is bumped `0.1.0` → `0.2.0`; v0.3 indexes refuse to open under v1. The storage layer hides behind a `Store` trait in `src/ctrldoc/store/`. Switching to Qdrant or FalkorDB is a config change, not a rewrite.

## Tool surface (the MCP integration)

The §6.10 tool surface is a 13-method closed alphabet: `lookup_concept`, `get_claim`, `traverse`, `entails`, `subsumes`, `optimal_transport`, `coverage`, `compare`, `merge`, `list_check`, `map`, `qa`, `calibration`. Each tool is a paired Pydantic input/output schema with `extra = "forbid"` and `frozen = True`; the `TOOL_SURFACE` dict aggregates them in declaration order and `TOOL_SURFACE_VERSION` is a semver-pinned string the MCP handshake reports so any host can detect schema drift (non-negotiable #14).

The `ctrldoc mcp serve` command runs an MCP server over stdio (JSON-RPC 2.0). It implements three methods (`initialize`, `tools/list`, `tools/call`) directly — no `mcp` SDK dependency — and is wire-compatible with any MCP-aware host (Claude Desktop, Claude CLI). See ADR-0007 for the in-house-vs-SDK decision.

## Verdict ledger

Every L4 verdict appends one row to the `verdict_ledger` table: `(operation, inputs_json, output_json, model_versions_json, persisted_confidence, workspace_id, paraphrase_vote_json?, created_at)`. `VerdictLedger.replay(entry_id, replayer)` hands the persisted inputs dict to the supplied replayer and scores the result against the §6.5 ±0.02 determinism gate. The facade exposes only `append`, `get`, `list_entries`, and `replay`; no UPDATE/DELETE SQL is emitted at the storage layer — the append-only contract is non-negotiable #13.

## Provenance

Every output carries a `Provenance` record: run ID, timestamp, operation + version, schema version, index hash, model identifiers, tokenizer name. This is what makes reproducible audit trails possible.

## What the system explicitly does *not* do (v1)

- Privacy mode / on-device-only deployment with no Anthropic option.
- Active learning loops on top of verdicts.
- Cross-modal claims (image / audio / table).
- Sheaf-theoretic global consistency proofs across N docs.
- Merkle-DAG provenance with cryptographic verification.
- Workspaces of 100+ documents.
- Formal-system interop (Lean / TLA+).

See [SPEC.md](SPEC.md) §15 for the complete out-of-scope list; everything above is queued for v2.
