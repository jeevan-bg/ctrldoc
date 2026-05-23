# ctrldoc — Specification Digest (v1)

A compact summary of `docs/SPEC.md`. Optimized for fast loading and stable content. For any detail, follow the SPEC-REF back to the full document.

## The one principle (§3)

A document is a noisy observation of a latent ontology. Reading it = jointly inferring concepts, relations, and claims. Multiple documents = noisy observations of a shared ontology — comparing them is aligning them in that shared space.

## Five first-principles axioms (§3)

1. Unit of meaning = the atomic claim, not the chunk.
2. Structure beats similarity — vectors find candidates; logic decides verdicts.
3. Documents are graphs; collections are graphs of graphs sharing a latent ontology.
4. Every verdict is replayable — output is an auditable proof trace.
5. The LLM is a perception layer, not a reasoning engine.

## Operations (§6, §9)

| Op | Question it answers |
|---|---|
| `qa` | What does the doc say about X — with citations? |
| `coverage` | Does target B cover source A? Per-claim verdict + confidence. |
| `compare` | Strengths / weaknesses / gaps across N docs. |
| `merge` | Lossless synthesis of N docs — every claim in one output cluster. |
| `list_check` | Per-item verdict against a doc. |
| `map` | Render the concept graph (Mermaid / JSON-LD). |

All six dispatch to one engine: optimal transport on the probability-weighted claim graph.

## Architecture (§5)

| Layer | Purpose |
|---|---|
| L0 Adaptive ingest | parse, coref, NER, chunk; **schema co-induction** (NEW) |
| L1 Multi-view index | + `claims`, `concepts`, `typed_edges` tables (NEW) |
| L1.5 Claim graph | **span / claim / concept triplane** (NEW primitive) |
| L2 Retrieval | dense + BM25 + entity + **personalized PageRank** (NEW) + rerank |
| L2.5 Workspace | **shared latent ontology over N docs** (NEW) |
| L3 Probabilistic edge inference | tuple logic + NLI + LLM judge + **paraphrase vote + ECE** (NEW) |
| L4 Tool-using orchestrator | forced tool calls + verdict ledger (NEW shape) |
| L5 Universal operations | `compare`/`coverage`/`merge`/`list_check`/`map`/`qa` over the engine |
| L6 Trace renderer | proof trace: spans → claims → edges → verdict + confidence |

## Two pillars (§4.1, preserved verbatim from v0.3)

- **Stateless tasks.** Every sub-task is a fresh API call with input `{system_prompt, doc_skeleton, glossary, evidence, task}` → structured JSON.
- **Shared prompt cache.** Identical prefix across sub-tasks; cache hits N times.

## Universal claim tuple (§6.2)

`Claim = (subject, predicate, object, polarity, modality, qualifier, span_refs, confidence)` — the logic floor. Contradiction = polarity flip. Stronger-than via qualifier ordering. Always extracted, even when adapter-specific extraction fails.

## Schema co-induction (§6.4)

Per-doc ontology *emerges* via EM loop: universal tuple as floor → LLM proposes typed nodes/edges from a primitive library → extract under schema → measure residual → re-induce if unmatched > 0.20 → cache as YAML. No hardcoded adapters.

## Galois lattice for "stronger than" (§6.3)

Partial order via logical entailment. Operations: `join`, `meet`, `incomparable` (first-class). Bidirectional NLI on natural text; predicate logic on formal text.

## Probabilistic edges + calibration (§6.5)

Every edge: calibrated `confidence ∈ [0,1]`. Sources: heuristic / NLI / LLM. Paraphrase voting (3–5 paraphrases) + isotonic regression → calibrated probability. **Shipped ECE per backend** with release gate `ECE ≤ 0.05`.

## Optimal-transport core (§6.6)

`compare`/`coverage`/`merge`/`list_check` = variants of min-cost flow on claim-pair edges weighted by `1 − NLI_entail`. Sinkhorn for soft variants. Many-to-one transport supported. `merge` loss invariant: every input claim ID → exactly one output cluster.

## Workspace (§6.7)

N docs share one concept lattice. Cross-doc edges (`aligned_with`, `entails_across`, `contradicts_across`, `stronger_than`) lazy + cached + linear in `|A|·k` via candidate retrieval.

## Tool surface (§6.10) — the MCP integration

`lookup_concept`, `traverse`, `entails`, `subsumes`, `optimal_transport`, `coverage`, `compare`, `merge`, `list_check`, `map`, `qa`, `calibration`. **Forced tool calls only.** Schemas versioned.

## Data model (§7)

Adds `Claim`, `Concept`, `TypedEdge`, `Workspace`, `CoverageReport`, `CoverageVerdict` to the v0.3 Pydantic models. `schema_version` 1 → 2.

## Storage (§8)

Adds tables: `claims`, `concepts`, `typed_edges`, `workspaces`, `cross_doc_edges`, `verdict_ledger`. Guarded by `IndexVersions` bump.

## CLI surface (§9)

v0.3 commands kept (now thin wrappers). v1 additions: `workspace {create|add|list|info}`, `compare`, `coverage`, `merge`, `list-check`, `graph {show|query}`, `schema {show|pin}`, `calibration`, `ledger {list|show|replay}`, `mcp serve`.

## Test families (§14, kept from v0.3)

14 families. Every test belongs to one. Coverage rule: an operation ships only when every applicable family has ≥ 1 test.

## The 14 Non-Negotiables (§13)

**v0.3 (1–5):** eval-first; LLM never sees raw doc; every claim cited or refused; every op stateless per task; storage layer abstracted.

**v1 (6–14):** universal tuple always extracted; residual rate observable; edges carry calibrated confidence; ECE ≤ 0.05 release gate; merge loss invariant; cross-doc edges cite source spans both docs; orchestrator uses forced tool calls; verdict ledger append-only & replayable; MCP tool schemas versioned.

## Performance / cost targets (§16 end state)

Unchanged from v0.3 per-operation budgets (qa $0.10/30s, coverage_audit $5/5min, …) — see `.ctrldoc/BUDGET.md`. v1 adds: ECE backend evaluation cost (one-shot, off the hot path) and cross-doc edge computation (linear in `|A|·k`, k=5).

## Out of scope for v1 (§15)

Privacy mode, active learning, cross-modal claims, sheaf-theoretic global consistency, Merkle-DAG provenance, 100+ doc workspaces, formal interop (Lean/TLA). All deferred to v2.
