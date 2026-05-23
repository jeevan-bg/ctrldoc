# ctrldoc — v1.0 Spec

*Universal claim-graph substrate for citation-grounded, calibrated, multi-document analysis.*

> This document supersedes the v0.3 spec (archived at
> `.ctrldoc/SPEC_v0.3_ARCHIVE.md`) at the architectural level. The two pillars in
> §3.1 and the five non-negotiables in §13 are reproduced verbatim from v0.3 so
> this spec is self-contained. Everything else is **additive or a refactor that
> preserves them**.

---

## 1. Vision

A user uploads N documents — a spec, a textbook, three RFCs, a stack of contracts, a
novel, a research paper, anything — and asks comparative, coverage, synthesis, or
relational questions ("does B cover A?", "what's the difference?", "merge these
losslessly", "where do they contradict?", "graph the concepts"). The system answers
with **per-claim verdicts, calibrated confidence, and citations to source spans**,
and it does so without context rot, dilution, drift, or hallucination — by
construction.

The product is **the auditable substrate for high-stakes multi-document reasoning**.
The deliverable a user touches is either the CLI, the Python API, or an **MCP server**
that any Claude-compatible client can plug into.

---

## 2. Problems Solved

| Failure mode | Mechanism |
|---|---|
| **Hallucination** | LLM never sees raw doc; only retrieved spans + decomposed claims, each verified back to source. |
| **Context rot** | Every sub-task is a fresh stateless API call. Nothing accumulates. |
| **Context dilution** | Retrieval (vectors + structure + graph walk) narrows to focused evidence before the LLM is invoked. |
| **Drift** | No multi-turn reasoning; each judgment is independent and replayable. |
| **Token burn** | Prefix cache + tiered routing (free deterministic → small local model → frontier LLM only when needed) + per-call evidence budget ≤ 6k tokens. |
| **Single-doc ceiling** | New **Workspace** primitive: N docs share one latent ontology. |
| **Hardcoded schemas** | New **Schema co-induction**: per-doc ontology *emerges* from the content, no domain code. |
| **Boolean verdicts** | New **Probabilistic edges**: every finding carries calibrated confidence with shipped ECE. |
| **Heuristic similarity** | New **Universal claim layer + Galois lattice + optimal transport**: logic replaces vibes. |

---

## 3. First-Principles Reframe

> **A document is a noisy observation of a latent ontology. Reading it means jointly
> inferring (a) what concepts exist, (b) how they relate, and (c) what claims the doc
> makes about them. Multiple documents are multiple noisy observations of an
> overlapping ontology — comparing them is aligning them in the shared latent space.**

Five axioms follow:

1. **Unit of meaning is the atomic claim, not the chunk.** Chunks remain physical storage; claims become semantics.
2. **Structure beats similarity.** Vectors find candidates; logic decides verdicts. Negation, quantification, conditionals are bit flips and typed edges, not vector nudges.
3. **Documents are graphs; collections are graphs of graphs sharing a latent ontology.** Coverage, contradiction, merge, diff, alignment are graph algorithms with NLI / LLM as edge oracles.
4. **Every verdict is replayable.** Output is not "an answer" — it is an auditable proof trace with provenance, models, paraphrase votes, calibrated probability.
5. **The LLM is a perception layer, not a reasoning engine.** It extracts and adjudicates at ingest; the graph reasons at query time.

These five replace v0.3's playbook-centric framing.

---

## 4. What v0.3 Keeps (Preserved Verbatim)

### 4.1 Two Architectural Pillars

The system's guarantees (no rot, no dilution, no drift, bounded cost) come from two pillars working together. Everything else is implementation.

**Pillar 1 — Stateless Tasks.** Every sub-task (per-item judge, per-section sweep, per-claim verify) is an **independent API call with fresh context**. No accumulation across tasks. Each call sees only what it needs: `{system_prompt, doc_skeleton, entity_glossary, evidence_pack, task_input}`. Output is structured JSON. The orchestrator collects JSON results and feeds only the *distilled findings* to a synthesis call — never raw doc, never prior reasoning. This is what eliminates context rot and drift by construction.

**Pillar 2 — Shared Prompt Cache.** The cacheable prefix `{system_prompt, doc_skeleton, entity_glossary}` is identical across every sub-task. Anthropic's prompt cache keys on content prefix, not session ID — so N fresh sub-tasks hit the same cache N times. Effective cost of the prefix: ~10% of nominal after first call.

**Why they combine well.** Naively, "fresh sessions" sounds expensive (no context reuse). With prefix caching, fresh sessions are nearly free on the prefix and only pay full cost on the small task-specific tail. You get isolation *and* economy. This is the unlock that makes large-doc analysis affordable.

Practical rule: every operation is a *map* of stateless tasks followed by a single *reduce* over their structured outputs. Never carry context across tasks.

### 4.2 Other v0.3 substrate kept verbatim

- **L0 ingest** (parse → coref → NER → chunk → embed → BM25 → skeleton).
- **L1 multi-view index** (structural tree + dense + BM25 + entity).
- **L2 retrieval** planner + executor + reranker + evidence pack ≤ 6k tokens.
- **L3 verifier** (claim decomposition + NLI + LLM judge + refusal).
- **L4 orchestrator** (batching, tiered routing, synthesis).
- **Storage trait + SQLite + sqlite-vec + Tantivy.**
- **Three runtime profiles** (heuristic, thrifty, production).
- **Run artifacts in `runs/<run_id>/`.**
- **The 14 test families.**
- **`schema_version` discipline.**

The **playbook layer (L5)** is **demoted to a thin renderer** over the universal substrate; see §6.

---

## 5. New Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  CLI · Python API · MCP server                                   │
├──────────────────────────────────────────────────────────────────┤
│  L6  Trace renderer (Markdown / JSON / Mermaid)                  │
│      proof trace: spans → claims → edges → verdict + confidence  │
├──────────────────────────────────────────────────────────────────┤
│  L5  Universal operations (replaces playbook layer)              │
│      compare · coverage · merge · list_check · map · qa          │
│      all dispatch to the same optimal-transport / graph engine   │
├──────────────────────────────────────────────────────────────────┤
│  L4  Tool-using orchestrator                                     │
│      stateless · forced-tool calls · tiered routing · synthesis  │
├──────────────────────────────────────────────────────────────────┤
│  L3  Probabilistic edge inference                                │
│      tuple logic · NLI · LLM judge · paraphrase vote · ECE       │
├──────────────────────────────────────────────────────────────────┤
│  L2.5  Workspace (NEW — shared latent ontology over N docs)      │
│      concept canonicalization across docs · cross-doc edges      │
│      lazy + cached · linear via candidate retrieval              │
├──────────────────────────────────────────────────────────────────┤
│  L2  Retrieval (hybrid + graph walk)                             │
│      planner · dense ⊕ BM25 ⊕ entity ⊕ personalized PageRank     │
│      reranker · evidence pack ≤ 6k tokens                        │
├──────────────────────────────────────────────────────────────────┤
│  L1.5  Claim graph (NEW — promoted from L5 playbook to primitive)│
│      span layer · claim layer · concept layer + is_a/part_of     │
│      lattice · typed edges with calibrated confidence            │
├──────────────────────────────────────────────────────────────────┤
│  L1  Multi-view index (kept) + claim/edge/concept tables (new)   │
├──────────────────────────────────────────────────────────────────┤
│  L0  Adaptive ingest (kept) + schema co-induction (new)          │
│      perplexity-boundary chunking (new, optional)                │
│      schema induction loop (new, Tier-3 LLM, batched at ingest)  │
└──────────────────────────────────────────────────────────────────┘
```

Three additions, no removals: **L1.5 claim graph**, **L2.5 workspace**, **schema co-induction**.

---

## 6. The Universal Substrate (kills the playbook layer)

In v0.3, each use case was its own playbook with its own code path. In v1, **every
analysis is a function over the same data structure**: the claim graph (single doc) or
the workspace (N docs).

### 6.1 Triplane graph (per document)

Not one graph — three coupled layers, each with its own job:

| Layer | Job | Backed by |
|---|---|---|
| **Span layer** | Where the words physically live. | `chunks` table (kept from v0.3). |
| **Claim layer** | Atomic propositions in the universal tuple. | New `claims` table. |
| **Concept layer** | Canonical entities organized in a lattice (`is_a` / `part_of`). | New `concepts` + `concept_edges` tables. |

Every node and edge carries provenance back to spans. Every layer is queryable
independently; operations dispatch to the layer they need.

### 6.2 The universal claim tuple (logic floor)

Every claim, regardless of doc type, is represented as:

```
Claim = (subject, predicate, object, polarity, modality, qualifier, span_refs, confidence)
```

This is the **logic substrate**: contradiction = polarity flip; stronger-than =
qualifier ordering; equivalence = bidirectional NLI entailment. Works on any
sentence. Always extracted, even when adapter-specific extraction fails.

### 6.3 Galois lattice for "stronger than"

A claim `C₁` is stronger than `C₂` (`C₁ ⊑ C₂`) iff `C₁` logically entails `C₂`. This
defines a **partial order** over claims; the resulting structure is a lattice with
operations:

- `join(C₁, C₂)`  — weakest claim both imply.
- `meet(C₁, C₂)`  — strongest claim that implies both.
- `incomparable(C₁, C₂)` — first-class verdict when neither subsumes (e.g., "OAuth" vs "SAML").

For formal text: exact via predicate logic. For natural text: approximate via NLI
with bidirectional entailment + qualifier-ordering heuristics + paraphrase vote.
Approximations are tagged in the edge metadata so consumers know which is which.

### 6.4 Schema co-induction (emergent adapters)

Rejected: hardcoded per-doc-type schemas. Adopted: **the schema is data, emerges per doc, refines on residuals**.

Algorithm (EM-style, 2–3 iterations):

1. **Floor:** extract universal tuples for the whole doc (Tier 1+2, no LLM).
2. **Propose:** sample 8–12 representative chunks (max-entropy sampling on the embedding cloud → diverse coverage). One Tier-3 LLM call: *"Given these excerpts, propose typed nodes and typed edges, composed only from the primitive library {Entity, Event, Process, Property, Quantity, Definition, Assertion, Obligation, Citation, Relation}. Output JSON."*
3. **Extract under schema:** Tier-2 dependency parser + NER + a small LLM pass typed by the proposed schema, over the whole doc.
4. **Residual:** measure `unmatched_claim_rate` = fraction of universal tuples that didn't bind to any typed slot. If `> τ_residual` (default 0.20), re-propose with residual claims as evidence, then re-extract *only over affected sections*.
5. **Cache:** the converged schema is stored as YAML alongside the doc index. Reusable for sibling docs (workspace-level schema reuse).

Why this is safe:
- Universal tuple is always the floor — if induction collapses, logic still works.
- Library of atomic primitives bounds the type space (no infinite category proliferation).
- Residual rate is a first-class observable, surfaced in the UI; high residual → user warned.
- Schema is data, not code; the user can inspect, edit, override, pin.

### 6.5 Probabilistic edges + calibration

Every edge carries `confidence ∈ [0, 1]`. Sources:

- **Heuristic** (Hearst pattern, heading tree): ~0.9 fixed.
- **NLI** (DeBERTa): raw model score.
- **LLM judge**: raw judge score.

Verdict propagation along edges uses **Bayesian product** under the local-independence assumption:

```
P(verdict | path) = ∏ confidence(edge_i)
```

Calibration: paraphrase voting (run NLI / judge on 3–5 paraphrases of the claim; agreement → cheap high-confidence; disagreement → escalate) plus **isotonic regression** fitted on the eval set, mapping raw scores → calibrated probabilities.

Shipped metric: **Expected Calibration Error (ECE)** per backend. v1 release gate: `ECE ≤ 0.05` on the held-out eval. ECE is surfaced via the CLI (`ctrldoc calibration`) and the trace renderer.

### 6.6 Optimal-transport operations (one algorithm, five queries)

The user-facing operations — `compare`, `coverage`, `merge`, `list_check`, `map`, `qa` — collapse into variants of one mathematical primitive:

> **Optimal transport between two probability-weighted claim distributions over the shared concept lattice.**

| Operation | Reduction |
|---|---|
| `coverage(A → B)` | Min-cost transport of B's claim-mass onto A's claim-mass, cost = `1 - NLI_entail(A, B)`. Unmoved mass = uncovered. |
| `compare(A, B)` | Asymmetric transport in both directions; per-concept-cluster cost summary = strengths/weaknesses. |
| `merge({A, B, C})` | Centroid distribution over the union; per-cluster strongest claim (Galois join) emitted; topological order via `depends_on` + `part_of`. Loss invariant: every input claim ID maps to exactly one output cluster. |
| `list_check(items, D)` | List parsed as a tiny doc; `coverage(items → D)`. |
| `map(D)` | Render the concept layer of D's triplane graph (Mermaid / JSON-LD). |
| `qa(D, query)` | Graph-walk retrieval seeded at query → evidence pack → verifier (unchanged from v0.3 modulo seeding). |

Implementation: min-cost flow (sparse, polynomial) or Sinkhorn for soft variants. Edges where transport cost > τ_uncovered → labelled `Missing`. Many-to-one transport (one claim in B jointly covered by several in A) is supported natively.

### 6.7 Workspace = shared latent ontology

A **Workspace** is a typed collection of doc-graphs with:

- A **shared concept lattice**: concepts across docs are canonicalized into one ID via the ER pipeline below.
- **Cross-doc edges** (lazy, cached): `aligned_with`, `entails_across`, `contradicts_across`, `stronger_than`, `equivalent_to`.
- **Provenance tags** on every claim: which doc, which span, which section.

The shared concept space *emerges from the docs* (schema co-induction across the workspace's docs at workspace-create time). The map is shared; the evidence stays labelled. Queries can filter ("show only doc-A claims," "show consensus regions," "show divergence").

### 6.8 Entity resolution (canonicalization)

Concepts across docs are merged into one canonical ID via the standard ER recipe:

1. **Blocking** on embedding cosine ≥ τ_block (default 0.85) generates candidate pairs cheaply.
2. **LLM judge** on candidate pairs only, with the source spans in context: *equivalent / subsumes / subsumed_by / incomparable*. JSON output.
3. **Union-find** over equivalence verdicts → canonical cluster IDs.
4. **Subsumption edges** (Galois lattice) stored as typed edges between canonical concepts.

Scales linearly thanks to blocking; ER cost stays a small fraction of total ingest.

### 6.9 Graph-walk retrieval

L2 retrieval is upgraded:

1. **Seed:** query → entity-link → seed concept nodes.
2. **Walk:** personalized PageRank along typed edges, with edge-type weights:
   - `depends_on`, `refines`, `prerequisite_of`: high (precision-relevant).
   - `is_a`, `part_of`: medium (abstraction-relevant).
   - `related_to`: low.
3. **Harvest:** chunks anchored to the top-N concepts by stationary probability.
4. **Fuse** with existing dense ⊕ BM25 ⊕ entity; rerank as before.

Catches the two-hop coverage that pure vector retrieval physically cannot find.

### 6.10 Tool-using orchestrator (replaces playbook plumbing)

L4 exposes a fixed tool surface to a top-level LLM (or to direct API callers):

```
lookup_concept(name) → ConceptId | None
get_claim(claim_id) → Claim
traverse(node_id, edge_type, direction, hops) → list[NodeId]
entails(claim_a, claim_b) → {verdict, confidence}
subsumes(claim_a, claim_b) → {verdict, confidence}
optimal_transport(distribution_a, distribution_b, cost_fn) → TransportPlan
coverage(workspace, target_doc_id, source_doc_id) → CoverageReport
compare(workspace, doc_ids) → CompareReport
merge(workspace, doc_ids) → MergedDoc
list_check(items, doc_id) → list[Verdict]
map(doc_id, filters) → Mermaid
qa(doc_id_or_workspace, query) → AnswerWithTrace
calibration() → {ECE_per_backend, sample_sizes}
```

**Forced tool calls only** — no free-form reasoning at the orchestrator level. Every answer comes back as a structured object with embedded provenance.

---

## 7. Data Model (additions to `models.py`)

```python
class Claim(_Strict):
    id: str                          # content-hashed
    doc_id: str
    text: str                        # canonical form
    subject: str | None
    predicate: str
    object: str | None
    polarity: Literal["+", "-"]
    modality: Literal["assert", "must", "may", "should", "shall", "neg"] | None
    qualifier: dict[str, object]
    span_refs: list[Span]
    section_id: str
    concept_ids: list[str]           # concepts this claim binds to
    typed_slots: dict[str, object]   # adapter-specific fields (Event.timepoint, …)
    confidence: UnitInterval         # extraction confidence

class Concept(_Strict):
    id: str                          # canonical cluster id
    canonical_name: str
    aliases: list[str]
    primitive_type: Literal[         # the atomic library
        "Entity", "Event", "Process", "Property", "Quantity",
        "Definition", "Assertion", "Obligation", "Citation", "Relation"
    ]
    mention_claim_ids: list[str]
    doc_ids: list[str]               # which docs this concept appears in

class TypedEdge(_Strict):
    src_id: str
    dst_id: str
    type: Literal[
        "entails", "contradicts", "refines", "instantiates",
        "depends_on", "prerequisite_of", "part_of", "is_a",
        "example_of", "alternative_to", "equivalent_to", "related_to",
        # cross-doc only:
        "aligned_with", "entails_across", "contradicts_across", "stronger_than"
    ]
    confidence: UnitInterval
    raw_score: float                 # pre-calibration model output
    citations: list[Span]
    source: Literal["heuristic", "nli", "llm", "induction"]
    paraphrase_votes: int | None     # number of paraphrases that voted; None for non-NLI

class Workspace(_Strict):
    id: str
    name: str
    doc_ids: list[str]
    induced_schema: dict[str, object]   # the shared, co-induced ontology
    provenance: Provenance

class CoverageReport(_Strict):
    workspace_id: str
    target_doc_id: str
    source_doc_id: str
    per_claim: list[CoverageVerdict]
    summary: CoverageSummary             # rates of Covered/Partial/Missing/Contradicted

class CoverageVerdict(_Strict):
    target_claim_id: str
    verdict: Literal["Covered", "Partial", "Missing", "Contradicted"]
    aligned_source_claims: list[str]
    transport_cost: float
    calibrated_confidence: UnitInterval
    trace: ProofTrace                   # full chain
```

Schema migration bumps `schema_version` from `1` → `2`. The existing v0.3 records continue to read under v1 with new tables alongside.

---

## 8. Storage Schema (SQL additions)

```sql
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    subject TEXT, predicate TEXT NOT NULL, object TEXT,
    polarity TEXT NOT NULL CHECK (polarity IN ('+', '-')),
    modality TEXT,
    qualifier_json TEXT NOT NULL DEFAULT '{}',
    span_refs_json TEXT NOT NULL,
    section_id TEXT NOT NULL,
    concept_ids_json TEXT NOT NULL DEFAULT '[]',
    typed_slots_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL
);
CREATE INDEX idx_claims_doc ON claims(doc_id);
CREATE INDEX idx_claims_section ON claims(section_id);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    primitive_type TEXT NOT NULL,
    mention_claim_ids_json TEXT NOT NULL DEFAULT '[]',
    doc_ids_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS typed_edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_score REAL NOT NULL,
    citations_json TEXT NOT NULL,
    source TEXT NOT NULL,
    paraphrase_votes INTEGER,
    PRIMARY KEY (src_id, dst_id, type)
);
CREATE INDEX idx_edges_src ON typed_edges(src_id, type);
CREATE INDEX idx_edges_dst ON typed_edges(dst_id, type);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    doc_ids_json TEXT NOT NULL,
    induced_schema_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cross_doc_edges (
    workspace_id TEXT NOT NULL,
    src_claim_id TEXT NOT NULL,
    dst_claim_id TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_score REAL NOT NULL,
    citations_json TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (workspace_id, src_claim_id, dst_claim_id, type)
);

CREATE TABLE IF NOT EXISTS verdict_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    operation TEXT NOT NULL,             -- "coverage" | "compare" | …
    inputs_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    calibrated_confidence REAL NOT NULL,
    model_versions_json TEXT NOT NULL,
    paraphrase_votes_json TEXT,
    timestamp TEXT NOT NULL
);
```

All new tables guarded by `IndexVersions` bump; v0.3 indexes refuse to open under v1 without explicit migration.

---

## 9. CLI Surface

```bash
# v0.3 commands kept (now thin wrappers over the universal layer):
ctrldoc ingest <doc>                       # also runs claim graph + schema induction
ctrldoc qa <doc-or-workspace> <query>
ctrldoc audit <target> --against <source>
ctrldoc review <doc>
ctrldoc scan <doc>
ctrldoc map <doc>

# v1 additions:
ctrldoc workspace create <name>
ctrldoc workspace add <name> <doc>
ctrldoc workspace list
ctrldoc workspace info <name>

ctrldoc compare <workspace>                # symmetric N-doc comparison
ctrldoc coverage --workspace <ws> --target <doc> --source <doc>
ctrldoc merge   --workspace <ws> --output <merged.md>
ctrldoc list-check <list.md> <doc>

ctrldoc graph show <doc>                   # Mermaid + JSON-LD
ctrldoc graph query <doc> --concept <name>
ctrldoc schema show <doc>                  # the induced schema YAML
ctrldoc schema pin --workspace <ws> --from <doc>

ctrldoc calibration                        # shipped ECE per backend
ctrldoc ledger list / show / replay        # the verdict ledger

ctrldoc mcp serve                          # MCP server on stdio for Claude Desktop / CLI
```

Every command writes `runs/<run_id>/report.md` + `runs/<run_id>/result.json` per v0.3 contract.

---

## 10. Python API (the plug-in surface)

```python
from ctrldoc import Workspace, ingest, profile

ws = Workspace("audit-2026")
ws.add(ingest("rfc7231.md", profile="thrifty"))
ws.add(ingest("our-impl.md", profile="thrifty"))

report = ws.coverage(target="our-impl.md", source="rfc7231.md")
for v in report.per_claim:
    print(v.target_claim_id, v.verdict, v.calibrated_confidence, v.trace.citations)

print(ws.calibration())   # {"nli": ECE=0.04, "judge": ECE=0.06}
```

Every return value is a Pydantic model with `.json()` and a `.trace` attribute that walks back to source spans.

---

## 11. MCP Server (the Claude integration)

`ctrldoc mcp serve` exposes the L4 tool surface (§6.10) as MCP tools over stdio.
Claude Desktop, Claude CLI, or any MCP-compatible client can:

- Create / open a workspace.
- Add docs (paths or URLs).
- Call `coverage`, `compare`, `merge`, `list_check`, `map`, `qa`.
- Receive structured `CoverageReport` / `CompareReport` / etc. with proof traces.

The server is stateless per call (workspaces are persisted on disk). Calls are
idempotent and use content-hashed cache keys, so Claude can call the same operation
repeatedly without recomputation cost.

A `tools.json` schema is published alongside the binary so the MCP host autocompletes
calls.

---

## 12. v0.3 → v1.0 Comparison

| Dimension | v0.3 | v1.0 |
|---|---|---|
| Primitive | Chunk + EvidencePack | Claim graph (triplane) + Workspace |
| Multi-doc | Single doc per run | Native N-doc via Workspace |
| Schema | Implicit, none beyond entity glossary | Co-induced per doc; YAML, inspectable |
| Adapters | None / hardcoded NER labels | Emerges per doc from primitive library |
| Verdict shape | `{Covered, Partial, NotCovered, Ambiguous}` boolean | Probabilistic + calibrated + proof trace |
| Reliability metric | Refusal-on-fail | **ECE** shipped; refusal threshold principled |
| Comparison algo | Per-playbook, ad-hoc | One algorithm (optimal transport) |
| Subsumption | Not modeled | Galois lattice via `is_a` / qualifier ordering |
| Retrieval | Dense ⊕ BM25 ⊕ entity + rerank | + Personalized PageRank on typed graph |
| Use case API | Six playbooks (L5) | Six L5 operations dispatched to one engine |
| Audit artifact | Run folder | Run folder + verdict ledger (replayable) |
| Integration | CLI / Python | CLI / Python / **MCP server** |
| Cross-doc edges | None | Lazy, cached, linear via candidate retrieval |
| Contradiction detection | Verifier-level only | Intra-doc cycle detection (graph-native) |

What is **kept verbatim**: the two pillars, L0–L4 substrate, three profiles, 14 test families, schema_version discipline, run-folder format.

What is **deleted**: the per-playbook L5 code paths (replaced by a thin renderer + a single optimal-transport engine).

What is **net new**: claim/concept/edge tables, workspace, schema co-induction, Galois lattice, optimal-transport engine, calibrated probabilistic edges, ECE pipeline, paraphrase voting, graph-walk retrieval, tool-using orchestrator, MCP server, verdict ledger.

---

## 13. The Non-Negotiables

The five v0.3 non-negotiables are reproduced verbatim. The nine v1 additions extend, not replace.

**From v0.3 (1–5):**

1. **Eval harness exists before any code that it scores.**
2. **No LLM call ever sees the raw full document.** Only skeleton, glossary, retrieved spans, or structured findings.
3. **Every claim cited or refused.** No uncited prose in outputs.
4. **Every operation is stateless per task.** Fresh context per sub-call.
5. **Storage layer is abstracted.** SQLite is the MVP backend, not the architecture.

**New in v1 (6–14):**

6. **Universal tuple is always extracted** — never replaced by adapter-only output.
7. **Residual rate is observable** — schema co-induction emits `unmatched_claim_rate`; CLI / report surfaces it.
8. **Edges carry calibrated confidence** — no boolean edges in shipped output.
9. **ECE is shipped per backend** — release blocks if `ECE > 0.05` on the held-out eval.
10. **Optimal-transport ops respect the loss invariant** — `merge` maps every input claim to exactly one output cluster; CI checks.
11. **Every cross-doc edge has a source-span citation** in both docs.
12. **Tool-using orchestrator uses forced tool calls** — no free-form reasoning at L4.
13. **Verdict ledger is append-only and replayable.**
14. **MCP tool schemas are versioned** and bumped on breaking change.

If any of these slip, the product loses its core guarantee.

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Claim extraction noise → poisoned graph | Dual extraction (rule + LLM consensus); hard gate `claim_F1 ≥ 0.85`; universal tuple as floor. |
| NLI domain drift (legal, biomed) | Paraphrase voting + isotonic calibration; per-domain ECE tracked separately. |
| Schema-induction misses on long-tail docs | Residual loop with re-propose; user-overridable YAML schema. |
| Cross-doc edge explosion (N² claims) | Candidate retrieval (vectors + entity overlap) → top-k; bipartite matching, not all-pairs. |
| `stronger_than` partially undefined | `incomparable` is a first-class verdict; never fabricated. |
| Graph maintenance under doc edits | Edges tagged with source span IDs; edge invalidated iff source span invalidated (extends v0.3 family 13). |
| LLM-induced schema hallucination | Sample max-entropy chunks (representative); residual rate caught silently; user warning surfaces. |
| Optimal transport on large graphs slow | Sparse min-cost flow; Sinkhorn for soft; cap claim count per call; offload to background jobs. |

---

## 15. Out of Scope for v1

Explicitly deferred to v2:

- **Privacy mode** (local-only LLM tier; no spans leave the machine).
- **Active learning** (LoRA fine-tune from user disagreements).
- **Cross-modal claims** (tables, figures, equations as first-class).
- **Sheaf-theoretic global consistency** (gluing local sections; nice-to-have).
- **Merkle-DAG provenance** (per-claim integrity proofs).
- **100+ doc workspaces** (sharded graph; current target is 2–10 docs).
- **Formal interop** (Lean / TLA — explicitly removed per user direction).

---

## 16. End State

A user runs:

```bash
ctrldoc workspace create due-diligence
ctrldoc workspace add due-diligence company-spec.pdf
ctrldoc workspace add due-diligence security-policy.pdf
ctrldoc workspace add due-diligence soc2-report.pdf
ctrldoc coverage --workspace due-diligence --target soc2-report.pdf --source security-policy.pdf
```

…and gets a Markdown report + JSON payload where every line is `(claim, verdict, calibrated_confidence, citations_in_both_docs)`. They can run `ctrldoc ledger replay` six months later and reproduce every verdict. Or they run `ctrldoc mcp serve` and ask Claude in chat: *"compare these three docs"* and Claude gets back the same structured result, traceable to source spans, with shipped ECE telling them how much to trust each number.

That is the v1 product. Everything in `PLAN_V1.md` is sized to reach this end state.
