# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_(Empty — the v1.0.0 line below freezes the v1.0 arc; subsequent slices accumulate here.)_

## [1.0.0] — 2026-05-24

**Universal claim-graph substrate.** The v0.3 per-use-case playbook
layer is gone; every operation now flows through one claim graph
(L1.5) + workspace (L2.5) primitive plus the optimal-transport core
(L5). The v0.3 surface is preserved unchanged — `ingest`, `qa`,
`scan`, `map`, `audit`, `review` all keep their reports and JSON
payloads — and is now layered on top of the universal substrate.

**v1 additions over the v0.3 release (`0.3.0`, see below):**

- **L1.5 claim graph.** Universal claim tuple as the logic floor
  (`(subject, predicate, object, polarity, modality, qualifier,
  span_refs, confidence)` per SPEC §6.2). Tier-1 deterministic
  extractor (Hearst patterns + heading-tree containment + PMI
  related-to + lexical-identity coref); Tier-2 SVO extractor over
  spaCy `en_core_web_sm` with copular / passive / modal-passive /
  agent-PP / advcl-xcomp coverage; Tier-2 NLI edge inferer with
  candidate retrieval (k=5 default) — never quadratic. Galois
  subsumption lattice (`equivalent` / `subsumes` / `subsumed_by` /
  `incomparable`) with `claim_join` (LUB) and `claim_meet` (GLB).
  Entity resolution (cosine blocking + `ERJudge` Protocol +
  union-find + canonical concept emission).
- **L2.5 workspace.** `Workspace` CRUD (`create` / `add` / `list` /
  `info`) over `SQLiteStore` with content-derived ids for replay
  stability; order-preserving doc membership; shared concept lattice
  via doc-id intersection. Cross-doc edge inferer (`aligned_with`,
  `entails_across`, `contradicts_across`) with linear `k * |A|` cost
  per ordered doc pair; source-span citations in both endpoint docs
  (SPEC §13 non-negotiable #11). `ctrldoc workspace` sub-app.
- **L0 schema co-induction.** Max-entropy chunk sample → single
  batched LLM call → JSON schema cached as deterministic block-style
  YAML. Residual EM loop re-induces on `unmatched_claim_rate > 0.20`,
  region-scoped re-extraction over affected sections only.
- **L3 probabilistic edges + calibration.** Paraphrase voting
  (3–5 paraphrases of the hypothesis; agreement rate vs binary
  correctness correlates with ρ ≥ 0.5 per SPEC §6.5). Isotonic
  calibration via pool-adjacent-violators; `CalibratedNLIScorer` wraps
  any raw backend; one-shot per-backend ECE measurement with release
  gate `CALIBRATION_ECE_THRESHOLD = 0.05` (SPEC §13 non-negotiable #9).
- **L5 optimal-transport engine.** `TransportProblem` (balanced
  bipartite input shape) + two solvers: `min_cost_transport`
  (successive shortest paths with Dijkstra-and-potentials on the
  residual graph; many-to-one transport) and `sinkhorn` (entropy-
  regularised Sinkhorn-Knopp on the Gibbs kernel). Pure-Python,
  stdlib-only, byte-deterministic across runs.
- **L5 universal operations.** `coverage` + `list_check` via the
  transport reduction with slack column priced at `1 - threshold`;
  `compare` (per-cluster `StrengthA` / `StrengthB` / `Gap`) with
  Galois floor first and asymmetric NLI fallback in both directions;
  `merge` with union-find + Galois-join representative selection,
  satisfying the §13 loss-invariant non-negotiable #10 by
  construction (every input claim maps to exactly one output cluster).
- **L2 graph-walk retrieval.** Personalized PageRank over typed
  claim-graph edges with the §6.9 three-tier edge-weight ladder.
  `GraphWalkRetriever` harvests top-k concepts by stationary
  probability and maps to anchored chunk ids for the existing
  dense ⊕ BM25 ⊕ entity fusion step. Release gate
  `PPR_RECALL_LIFT_THRESHOLD = 0.10` ("≥ 10 % recall lift on
  multi-hop queries").
- **L4 tool-using orchestrator + verdict ledger.** §6.10 13-tool
  surface (`lookup_concept`, `get_claim`, `traverse`, `entails`,
  `subsumes`, `optimal_transport`, `coverage`, `compare`, `merge`,
  `list_check`, `map`, `qa`, `calibration`) with strict frozen
  Pydantic input/output schemas + semver-pinned
  `TOOL_SURFACE_VERSION` (SPEC §13 non-negotiable #14). Append-only
  `verdict_ledger` table with `VerdictLedger` facade (`append` /
  `get` / `list_entries` / `replay`); replay determinism gated at
  `REPLAY_TOLERANCE = 0.02` (SPEC §13 non-negotiable #13).
  `ctrldoc ledger {list, show, replay}` sub-app.
- **MCP server.** `ctrldoc mcp serve` — Model Context Protocol over
  stdio (JSON-RPC 2.0). Implements `initialize` / `tools/list` /
  `tools/call` directly over the L4 dispatcher; `serverInfo.version`
  ships `TOOL_SURFACE_VERSION` so hosts detect schema drift at the
  handshake. Wire-compatible with stock MCP clients (Claude Desktop,
  Claude CLI). ADR-0007 records the in-house-vs-SDK transport
  decision.
- **Storage schema v2.** Six new tables — `claims`, `concepts`,
  `typed_edges`, `workspaces`, `cross_doc_edges`, `verdict_ledger` —
  provisioned alongside the v0.3 chunk / section / entity layout.
  `SCHEMA_VERSION` bumped `"0.1.0"` → `"0.2.0"` so v0.3 indexes
  refuse to open under v1 and must be re-ingested. See
  [`MIGRATION_v0.3_to_v1.0.md`](MIGRATION_v0.3_to_v1.0.md) for the
  upgrade walkthrough.
- **Real-doc shakedown corpus + smoke.** `tests/fixtures/real_docs/`
  ships a hand-built realistically-shaped corpus spanning every §16
  doc-type axis (spec, legal, academic, educational, narrative,
  spec-vs-impl pair). `scripts/real_doc_smoke.sh` runs every doc
  through the full L0 ingest + scan + workspace pipeline on the
  heuristic profile in ~20 seconds. No LLM, no Ollama, no network.
- **README + ARCHITECTURE rewrites.** README advertises the v1
  surface (workspace / coverage / compare / merge / mcp commands);
  ARCHITECTURE describes the v1 9-layer stack including L1.5 claim
  graph, L2.5 workspace, the optimal-transport core, and the MCP
  integration.
- **`examples/v1/`.** Three runnable v1 walkthroughs (workspace,
  coverage-via-transport, merge-via-transport) — hermetic, no LLM
  credentials required.

**Breaking changes:**

- Storage `SCHEMA_VERSION` bumped `"0.1.0"` → `"0.2.0"`; v0.3 indexes
  refuse to open under v1. No in-place data migration — re-ingest.
- The v0.3 per-use-case L5 package was removed from disk. Every
  symbol is re-homed under `ctrldoc.ops.*` along its CLI-aligned name
  — see the rename table in [`MIGRATION_v0.3_to_v1.0.md`](MIGRATION_v0.3_to_v1.0.md).

**Preserved verbatim from v0.3:**

- The two pillars (stateless tasks + shared prompt cache, SPEC §4.1).
- Non-negotiables #1–5 (no raw doc to LLM; every claim cited or
  refused; every sub-task is stateless; storage is abstracted;
  every output carries provenance).
- The full v0.3 CLI surface (`ingest`, `qa`, `scan`, `map`, `audit`,
  `review`) — now thin adapters over the universal substrate.
- The 14 test families (SPEC §8.6 / §14); pytest markers unchanged.

### Added (per-slice detail, S-119 through S-147)

- Real-doc shakedown corpus + smoke script (SPEC §16). A new
  `tests/fixtures/real_docs/` directory ships a hand-built corpus
  spanning every §16 doc-type axis — a system specification
  (`spec_lighthouse.md`), a service-terms excerpt (`legal_terms.md`),
  an academic-style writeup (`academic_paper.md`), an explanatory
  tutorial (`educational_guide.md`), a short narrative
  (`narrative.md`), and a spec-vs-impl pair (`pair_spec.md` +
  `pair_impl.md`, linked by `pair_id: tideline`). `MANIFEST.yaml`
  declares each row's `doc_id`, `type`, `role`, `pair_id`, `path`,
  `title`, and `summary`, and is the oracle every downstream test
  reads instead of hard-coding paths. `scripts/real_doc_smoke.sh`
  drives every entry through the v1 substrate on the heuristic
  profile (no LLM, no Ollama, no network): the Python driver
  `ctrldoc.eval.real_doc_smoke` runs the full L0 ingest, the
  deterministic anomaly-scan battery, a determinism rerun (re-ingest
  every doc into a sibling tree and assert byte-identical signature
  hashes), and a workspace build from the spec-vs-impl pair. The
  driver writes a single `summary.json` the script then validates and
  surfaces as a per-doc table. Hermetic by construction, CI-safe,
  ~20 seconds locally.
- `ctrldoc.mcp` + `ctrldoc mcp serve` — Model Context Protocol server
  exposing the §6.10 13-tool surface over stdio JSON-RPC 2.0 (SPEC
  §11). `MCPServer` owns the protocol layer: envelope parsing, the
  three required methods (`initialize` / `tools/list` / `tools/call`),
  and per-call dispatch through the existing `ToolDispatcher` from
  S-142. `tools/list` derives each tool's `inputSchema` from the
  registered Pydantic `input_model.model_json_schema()` so the
  catalogue is byte-deterministic across calls and stays in lock-step
  with the L4 surface. `serverInfo.version` ships
  `TOOL_SURFACE_VERSION` so any MCP host can detect schema drift via
  the handshake (§13 non-negotiable 14). Tool-level failures (unwired
  handler, validation error, handler exception) surface as
  `CallToolResult.isError=true` with a text content block — never a
  silent no-op (§13 non-negotiable 3); transport-level failures
  (malformed envelope, unknown method) return a JSON-RPC `error`
  envelope with a reserved-band code. `serve_stdio` is the
  line-framed reader/writer loop that drives the protocol over
  `sys.stdin` / `sys.stdout`; blank lines are skipped, malformed JSON
  surfaces a parse-error envelope without killing the loop, EOF
  terminates cleanly. The integration test spawns
  `python -m ctrldoc mcp serve` as a real subprocess and drives the
  full `initialize` → `tools/list` → `tools/call` round-trip from a
  stock JSON-RPC 2.0 client written inline, proving the wire format
  is the standard MCP stdio transport a third-party host (Claude
  Desktop, Claude CLI) would speak. ADR-0007 documents the in-house-
  vs-`mcp`-SDK transport decision: the surface is JSON-RPC 2.0 with
  three methods, implementing it directly keeps the dependency graph
  small and the integration test deterministic.
- `ctrldoc.orch.ledger` — L4 verdict ledger + `ctrldoc ledger {list,
  show, replay}` CLI (SPEC §6.5, §11). `VerdictLedger` is a thin
  facade over `SQLiteStore` exposing `append`, `get`, `list_entries`,
  and `replay`. Persists one row per L4 verdict into the §8
  `verdict_ledger` table; `LedgerAppendRequest` and `LedgerEntry`
  mirror the table column-for-column. Append-only contract is
  enforced at both layers: the facade has no `update` / `delete` /
  `clear` method, and the storage helpers (`append_ledger_row`,
  `get_ledger_row`, `iter_ledger_rows`) emit only INSERT and SELECT
  against the table. `replay(entry_id, replayer)` hands the persisted
  `inputs` dict verbatim to a caller-supplied `Replayer` callback and
  scores the result against the §6.5 ±0.02 determinism gate:
  `REPLAY_TOLERANCE = 0.02`, with a `1e-9` boundary slack so deltas
  that land exactly on the threshold (e.g. via floating-point
  arithmetic like `0.40 + 0.02`) still pass. `ReplayOutcome` surfaces
  the persisted vs. replayed confidences, the raw delta, the
  tolerance, and a `is_deterministic` flag for the CLI to render the
  pass/fail verdict alongside the distance. The CLI sub-app routes
  through `<runs_path>/ledger.db` (one-file-per-substrate, like the
  workspaces DB): `ctrldoc ledger list [--workspace-id WS]` shows
  rows in append order; `ctrldoc ledger show <id>` returns the full
  entry with inputs / output / model versions / timestamp; `ctrldoc
  ledger replay <id>` runs an identity replayer over the persisted
  confidence to round-trip the gate plumbing end-to-end. Per-op
  replayers will plug in via the L4 tool dispatcher when the MCP
  server lands.

- `ctrldoc.retrieval.graph_walk` — personalized PageRank over typed
  claim-graph edges (SPEC §6.9). `EDGE_TYPE_WEIGHTS` encodes the §6.9
  three-tier ladder verbatim: `depends_on` / `refines` /
  `prerequisite_of` (high), `is_a` / `part_of` (medium), `related_to`
  (low). Other typed-edge types in the alphabet are silently skipped
  by the walker because the spec scopes L2 retrieval to the listed
  subset. `personalized_pagerank(edges, seeds, alpha, max_iter, tol)`
  computes the stationary distribution via power iteration; per-step
  walk probability over outgoing edges is proportional to
  `EDGE_TYPE_WEIGHTS[edge.type] * edge.confidence`. Dangling nodes
  teleport their mass to the seed distribution so probability is
  conserved. Defaults: `alpha = 0.85` (classic PageRank persistence),
  `max_iter = 50`, `tol = 1e-8` L1 distance; convergence is fast on
  the sparse claim graph. Pure-Python (no numpy / scipy),
  byte-deterministic across runs because the adjacency is built into
  a sorted-key dict. `GraphWalkRetriever(edges, concept_to_chunks,
  config)` wraps the primitive with a harvest stage: top-N concepts
  by stationary probability map to their anchored chunk ids, deduped
  in concept-rank order then within-concept input order, ready for
  the existing dense ⊕ BM25 ⊕ entity fusion step. `GraphWalkConfig`
  pins `alpha` / `max_iter` / `tol` / `harvest_k` (default 10).
  `recall_at_k(retrieved, gold, k)` is a stdlib-only helper that
  returns the §6.9 release-gate metric. `PPR_RECALL_LIFT_THRESHOLD =
  0.10` names the spec's "≥ 10 % recall lift on multi-hop queries"
  gate; the test suite proves it end-to-end on a 2-hop synthetic
  topology where gold concepts are reachable from the query seed
  only through intermediate concepts (seed-only baseline recovers
  zero gold; the walker recovers all three).

### Changed

- **Storage schema v2** (SPEC §8). `SQLiteStore` now provisions six
  new tables — `claims`, `concepts`, `typed_edges`, `workspaces`,
  `cross_doc_edges`, `verdict_ledger` — through the same idempotent
  `_init_schema` block as the v0.3 chunk/section/entity layout.
  `SCHEMA_VERSION` is bumped from `"0.1.0"` to `"0.2.0"`, so any
  index written by a v0.3 install will refuse to open under v1 and
  must be re-ingested. `clear_all` now truncates the new tables as
  part of its destructive reset.

### Added

- `ctrldoc.ops.transport` — optimal-transport engine on claim-pair
  edges (SPEC §6.6). `TransportProblem(source_weights, target_weights,
  cost_matrix)` is the balanced bipartite input shape (validators
  reject mismatched shape, negative entries, or unbalanced total
  mass within `1e-9`). Two solvers consume the same shape and emit
  the same `TransportPlan(flow, total_cost)`: `min_cost_transport`
  solves the transportation problem exactly via successive shortest
  paths (Dijkstra with potentials on the residual graph) producing
  sparse hard-assignment plans with possibly many-to-one transport —
  the foundation for `coverage` and `list_check`. `sinkhorn(problem,
  *, epsilon, max_iter, tol)` solves the entropy-regularised variant
  via Sinkhorn-Knopp matrix scaling on the Gibbs kernel
  `exp(-cost/epsilon)`, converging when both row and column marginal
  residuals drop under `tol` simultaneously — the foundation for
  `compare` and `merge`. Smaller `epsilon` sharpens the Sinkhorn plan
  toward the exact solution. Pure-Python, stdlib-only (no scipy or
  numpy dependency), byte-deterministic across runs. Slack mass for
  `Missing` / `Contradicted` verdicts is the caller's responsibility
  via explicit dummy claims so the engine itself stays a clean
  single-purpose primitive. 25 unit cases under `family_determinism`
  cover validation, exact solver correctness (one-to-one,
  many-to-one, marginals preserved, zero-weight rows/cols, zero-cost
  diagonal, off-diagonal optimum, chained split, 3x3 known-optimal,
  empty problem, inner-product bookkeeping, repeat-run
  determinism), and Sinkhorn behaviour (marginals preserved,
  epsilon-shrink convergence to the exact plan, repeat-run
  determinism, epsilon ≤ 0 rejection, strictly-positive flow at high
  epsilon, inner-product bookkeeping, tight-tolerance convergence).
- `ctrldoc.ops.coverage` — `coverage` + `list_check` operations via the
  optimal-transport reduction (SPEC §6.6). `coverage(source, target,
  scorer)` emits one `Covered` / `Missing` verdict per target claim in
  input order; `list_check(items, doc, scorer)` is the same primitive
  with items as targets and doc claims as sources, matching §6.6's
  "list parsed as a tiny doc; `coverage(items → D)`" framing.
  `TransportCoverageVerifier` adapts the same reduction onto the §14
  `CrossDocCoverageVerifier` protocol so the existing eval substrate
  grades it directly. The §6.6 entailment threshold (default `0.5`,
  configurable via `CoverageConfig.entailment_threshold`) is encoded
  as a slack column priced at `1 - threshold` — any real source with
  entailment confidence above the threshold strictly beats slack and
  the target reads `Covered`; otherwise the target's mass routes to
  the slack column and reads `Missing`. Total source mass is
  rebalanced with an absorption column so the `TransportProblem`
  validator's balanced-mass check holds. Cost contract: exactly
  `|sources| * |targets|` NLI scorer calls per `coverage` call.
  `COVERAGE_VERDICT_ACCURACY_THRESHOLD` re-exports the eval
  substrate's `CROSS_DOC_COVERAGE_THRESHOLD = 0.85` so callers gate
  by the §6.6 release-gate constant; the release contract is asserted
  end-to-end against the shipped 12-case fixture under a gold-aligned
  NLI oracle that isolates the transport reduction's correctness from
  any real backend's quality. 16 unit cases under
  `family_determinism`, `family_verifier_calibration`, and
  `family_performance_cost` cover empty-input short-circuits, hard
  Covered/Missing verdicts, polarity-flip contradiction, many-to-one
  transport, scorer-call accounting, repeat-run determinism, the
  `list_check` mirror surface, the `CrossDocCoverageVerifier` protocol
  shape, and the release-gate eval roll-up.
- `ctrldoc.extract.isotonic_calibration` — isotonic regression for the
  §6.5 calibration pipeline. `IsotonicCalibrator.fit(raw_scores,
  correct)` runs the pool-adjacent-violators algorithm against
  `(raw_score, binary_correctness)` pairs to learn a monotonic
  step function; `transform(raw_score)` maps any new raw score to a
  calibrated probability by linear interpolation between fitted
  breakpoints, with extrapolation clamping to the nearest endpoint
  and the output clipped to `[0, 1]`. `CalibratedNLIScorer` wraps any
  `NLIScorer`: it preserves the inner backend's argmax label,
  replaces the top-label confidence with the calibrated value, and
  redistributes the remaining mass over the non-top labels in
  proportion to their raw ratios (even-split fallback when both
  non-top raw masses are zero). `fit_per_backend_ece(raw_scores,
  correct)` is the one-shot release-gate helper: it fits a
  calibrator on the first half of a labelled set and reports
  held-out ECE on the second half. `ece_within_release_gate(ece)`
  names the §6.5 threshold (`CALIBRATION_ECE_THRESHOLD = 0.05`) so
  callers gate by intent. Stdlib-only — no scipy dependency. A
  miscalibrated 200-case synthetic backend (raw top-confidence
  inflated by 0.20) drops from pre-fit ECE 0.10 to post-fit ECE
  under the 0.05 release gate end-to-end in the test suite.
- `ctrldoc.extract.paraphrase_voting` — paraphrase voting for the §6.5
  calibration pipeline. `ParaphraseVoter.vote(premise, hypothesis)`
  asks the injected `Paraphraser` for `num_paraphrases` re-wordings of
  the hypothesis (band pinned to the §6.5 `[3, 5]` envelope; default
  3), scores each `(premise, paraphrase)` pair under the underlying
  `NLIScorer` exactly once, and aggregates the per-paraphrase argmax
  labels into a `ParaphraseVote` carrying `majority_label`,
  `agreement_rate` (fraction of paraphrases voting for the majority
  label), `mean_top_confidence` (averaged across only the majority
  paraphrases so dissenters do not drag the score toward the wrong
  label), `num_paraphrases`, and a complete-shape `label_votes` dict
  over `{entailment, contradiction, neutral}`. The agreement rate is
  the confidence proxy the isotonic-regression calibration layer
  consumes downstream; voting alone does not ship a calibrated
  probability. `spearman_rank_correlation(xs, ys)` is a stdlib-only
  helper (no scipy dependency) that computes the rank correlation via
  Pearson on the average-rank vectors and handles ties via the
  standard average-rank convention; it returns `0.0` on a constant
  sequence and raises on length-mismatched or single-pair inputs.
  `PARAPHRASE_CORRELATION_THRESHOLD = 0.5` pins the §6.5 acceptance
  gate: across a labelled batch, agreement rate vs binary correctness
  must clear Spearman rho >= 0.5; a 10-case fixture in the test suite
  exercises the gate end-to-end on confident-correct and
  hard-disagreeing paraphrase votes.
- `ctrldoc.extract.schema_proposer` — L0 schema proposer per SPEC §6.4
  step 2. `max_entropy_sample(chunks, embeddings, *, k)` runs greedy
  farthest-point selection on the embedding cloud: the seed is the
  chunk whose embedding is furthest from the cloud's centroid, and each
  subsequent pick maximises the minimum cosine distance to anything
  already picked. Ties break by input ordinal so the output is
  byte-stable across runs, length-mismatched inputs raise, and the
  function returns at most `min(k, len(chunks))` (empty input → empty
  output). `SchemaProposer.propose(chunks, doc_id)` wires the
  8-to-12-chunk sample through one batched `TaskClient.call` whose
  system prompt enumerates the closed 10-element `PrimitiveTypeLiteral`
  library — `Entity` / `Event` / `Process` / `Property` / `Quantity` /
  `Definition` / `Assertion` / `Obligation` / `Citation` / `Relation` —
  and whose user message carries only the sampled excerpts plus the doc
  id, never the full document. The returned JSON is parsed into a
  `SchemaProposal` of `TypedNodeSpec` and `TypedEdgeSpec` rows that
  reject unknown primitives and blank fields at the Pydantic boundary
  so a hallucinated primitive raises before the proposal reaches the
  workspace cache. `dump_schema_yaml(proposal, path)` and
  `load_schema_yaml(path)` round-trip the proposal as deterministic
  block-style YAML (one mapping of `nodes:` + `edges:`, each holding
  either `[]` or single-line key/value entries with double-quoted
  scalars; parent directories are created on dump). The dumper uses the
  stdlib only — no new project dependency — so the cache key (file
  hash) is reproducible across runs and environments and sibling docs
  in the same workspace can reuse the cached per-doc schema without
  paying for a second LLM round-trip.
- `ctrldoc.extract.galois` — Galois subsumption lattice over the
  universal claim tuple per SPEC §6.3. `claim_subsumption(left, right)`
  returns one of `equivalent` / `subsumes` / `subsumed_by` /
  `incomparable`; `claim_join` is the lattice LUB (the weakest claim
  both operands imply) and `claim_meet` is the GLB (the strongest
  claim that implies both), each returning `None` for incomparable
  pairs that share no common weakening / strengthening at the
  structural floor. The deterministic ordering reasons on the six
  §6.2 universal-tuple slots only: surface-form SVO inequality after
  the `normalize_text` pipeline, polarity flips, or cross-axis
  modality pairs collapse to `incomparable`. Modalities map to three
  axes — the deontic chain `obligatory ⊐ recommended ⊐ permitted`
  (RFC-2119 `MUST ⊐ SHOULD ⊐ MAY`), the prohibitive chain
  `prohibited ⊐ recommended ⊐ permitted` reached under negative
  polarity, and the singleton axes `asserted` (descriptive) and
  `hypothetical` (conditional). Within a same-axis pair an empty
  qualifier is strictly stronger than any scoped qualifier (the
  universal claim entails every narrowed instance), and two distinct
  non-empty qualifiers do not order — semantic scope reasoning is the
  upcoming NLI/LLM path's job, which calls this floor first and only
  escalates when it returns `incomparable`. Pure-function module; no
  I/O, no LLM, no state; output byte-identical across repeat calls.
- `ctrldoc.extract.entity_resolution` — entity-resolution
  canonicalizer per SPEC §6.8. `EntityResolver` runs the standard
  four-step ER recipe over a batch of `ConceptMention` rows:
  embedding-cosine blocking (any `Embedder` backend, default
  `tau_block = 0.85`, restricted to same-`PrimitiveTypeLiteral`
  pairs); an `ERJudge` Protocol returning the four-class verdict
  `equivalent` / `subsumes` / `subsumed_by` / `incomparable` per
  surviving pair; union-find (smaller-root-wins, path-compressed)
  over equivalence verdicts into canonical `Concept` rows whose
  `canonical_name` is the most-frequent surface form among the
  cluster's mentions (ties broken lexicographically); and
  subsumption verdicts rewritten onto canonical-concept endpoints
  and emitted as deduplicated `is_a` `TypedEdge` rows with
  `source = "llm"`. `EntityResolution` returns the concept list,
  the parallel mention-id cluster partition (so callers can
  rebuild membership for scoring), the subsumption edges, and the
  judge-call count for budget bookkeeping. `cluster_precision_recall`
  is the pairwise scoring helper used to gate the §6.8 release
  thresholds (`ER_PRECISION_THRESHOLD = 0.90`,
  `ER_RECALL_THRESHOLD = 0.85`) — verified on an inline 12-mention
  7-cluster gold fixture.
- `ctrldoc.extract.tier2_nli` — Tier-2 NLI edge inferer per SPEC
  §6.5. `Tier2NLIEdgeInferer` consumes a list of universal `ClaimTuple`
  rows (the Tier-2 SVO extractor's output) and emits `TypedEdge` rows
  of type `entails` / `contradicts` between pairs whose top-label
  NLI confidence crosses the default 0.70 threshold. The cost
  contract is the §6.5 candidate-retrieval bound: at most
  `k_candidates * N` ordered pairs reach the backend, where the
  default `k = 5` matches the spec's `5N` envelope; quadratic
  enumeration is explicitly forbidden. The candidate ranker is a
  token-overlap Jaccard over a lower-cased word-token bag, with
  ties broken on the lexicographic claim id for full determinism.
  Edges carry `source="nli"`, the raw top-label probability as
  both `confidence` and `raw_score` (the upcoming isotonic-
  calibration layer fits against the latter), and synthetic
  premise / hypothesis citation spans whose `chunk_id` is prefixed
  `tier2-nli:` so the trace renderer can surface the provenance.
  `render_claim_text` lifts a `ClaimTuple` into a natural-language
  surface with polarity-aware copula flips (`is` → `is not`, etc.)
  and trailing qualifier; `claim_id` is content-hashed over the
  six logical fields for stable cross-run identity.
- `ctrldoc.extract.tier2` + `ctrldoc.extract.tier2_spacy` — Tier-2
  SVO claim-tuple extractor per SPEC §6.4. The pure-Python helpers in
  `tier2` carry the modality lexicon (`must` / `shall` / `should` /
  `may` / `if` / `when` / `cannot` ...), a `NEGATION_TOKENS`
  frozenset, a `classify_modality` priority ladder that resolves the
  `shall not` -> prohibited rewrite, a `merge_modality_with_polarity`
  reconciliation rule, and a `lemmatize_predicate` helper that does
  subject-verb agreement on the verb lemma. `tier2_spacy.SpacyTier2-
  SVOExtractor` drives a spaCy `en_core_web_sm` dependency parser to
  fill the `Claim = (subject, predicate, object, polarity, modality,
  qualifier)` tuple from §6.2, satisfying the `ClaimExtractor`
  Protocol so the new extractor drops straight into the
  `ClaimExtractionEvalRunner` shipped by S-119. The backend handles
  copular `acomp` / `attr` decomposition, passive `auxpass`
  constructions (including `shall be resolved` modal-passive
  periphrasis), past-tense preservation for narrative prose, agent-PP
  object capture (`is governed by California law`), prep-tail trim
  with an argument-PP carve-out (`with` / `by` / `from` PPs stay
  inside the object when they directly modify it), `advcl` / `xcomp`
  purpose-clause qualifiers, sentence-prefix conditional scanning,
  and `if` -> `when` qualifier normalisation. The release gate is
  `TIER2_F1_THRESHOLD = 0.75` on the SVO-amenable subset of
  `tests/eval/claim_extraction_eval.jsonl` (single-tuple cases with
  the gold subject head word in the source sentence, no
  reporting-verb paraphrase or modal-periphrasis collapse — the
  patterns excluded here are routed to the Tier-3 LLM-mediated pass
  queued by S-129+).
- `ctrldoc.extract.tier1` — deterministic claim-graph floor per SPEC
  §6.4. Four heuristic pattern families compose in a single pass:
  Hearst lexico-syntactic patterns (`X such as Y`, `X including Y`,
  `X is a Y`) emit `example_of` / `is_a` edges; heading-tree
  containment over `Section.parent_id` emits `part_of` edges;
  sliding-window PMI over chunk tokens (configurable window,
  min-count, log2 threshold) emits `related_to` edges; lexical-
  identity coref over repeated proper-noun-ish surface forms emits
  `equivalent_to` self-edges. Public surface: `extract_tier1`,
  `Tier1Mention`, `Tier1Concept`, `Tier1Edge`, `Tier1Extraction`,
  `Tier1Config`, `HEURISTIC_CONFIDENCE` (the §6.5 fixed prior, 0.9).
  Every edge cites its producing span; concept ids are content-hashed
  on the canonical surface form so identical input bytes produce
  byte-identical output across runs.
- `ctrldoc.models_v1` — v1 substrate Pydantic models that mirror the
  v2 storage tables and carry the calibrated-edge graph the
  universal-transport operations traverse. Exports `Claim`, `Concept`,
  `TypedEdge`, `Workspace`, `CoverageReport`, `CoverageVerdict`,
  `CoverageSummary`, and `ProofTrace`, plus the public literal
  aliases `PolarityLiteral`, `ModalityLiteral`, `PrimitiveTypeLiteral`,
  `TypedEdgeTypeLiteral`, `EdgeSourceLiteral`, and `VerdictLiteral`.
  Every model is frozen with `extra='forbid'`; all confidences are
  unit-interval bounded; `CoverageSummary`'s four rates must form a
  probability mass (sum to 1.0 ± 1e-6). Lives in its own module so
  the v0.3 surface in `ctrldoc.models` keeps compiling unchanged
  through the v1 build-out.
- `ctrldoc.eval.claim_extraction` — universal-claim-tuple extraction
  eval substrate. Exports `ClaimTuple`, `ClaimExtractor` Protocol,
  `ClaimExtractionEvalRunner`, and `precision_recall_f1`. The
  `tests/eval/claim_extraction_eval.jsonl` starter set ships 120
  hand-curated sentence→tuple pairs across six doc types (spec,
  runbook, RFC, legal, academic, narrative); the runner gates each
  case on the F1 ≥ 0.85 floor.
- `ctrldoc.eval.calibration` — NLI calibration eval substrate per
  SPEC §6.5. Exports `NLIScore` (3-way softmax with sum-to-1
  validation), `CalibrationScorer` Protocol, `CalibrationEvalRunner`,
  and the metric primitives `label_accuracy`,
  `expected_calibration_error` (Guo et al. 2017, equal-width binning
  on top-label confidence), and `per_label_recall`. The
  `tests/eval/calibration_eval.jsonl` starter set ships 200
  hand-authored premise/hypothesis cases balanced across
  {entailment, contradiction, neutral} and spanning six doc types;
  the runner gates each case on `label_accuracy ≥ 0.85` AND the v1
  release-gate threshold `ECE ≤ 0.05`.
- `scripts/eval_v1_*.py` — five baseline measurement scripts (one
  per v1 eval substrate: claim_extraction, cross_doc_coverage,
  compare, merge, calibration). Each drives a degenerate Protocol
  stub through the substrate's runner against the shipped JSONL
  fixture and prints a single-line JSON summary on stdout.
- `scripts/run_v1_smoke.sh` — sequential aggregator over the five
  baseline scripts. Exits non-zero if any baseline fails to execute
  or emit a parseable summary; prints a per-substrate `OK | FAIL`
  table plus the raw JSON summaries. Intended as a CI wiring check
  for the v1 eval substrates.
- `ctrldoc.ops.cross_doc_edges` — L2.5 cross-doc edge inferer per SPEC
  §6.7. `CrossDocEdgeInferer` bridges N workspace member docs with
  `aligned_with`, `entails_across`, and `contradicts_across` typed
  edges produced by an NLI scorer. For every ordered pair of distinct
  member docs `(A, B)` the inferer picks the top-`k` target-doc
  candidates per source claim via deterministic token-overlap Jaccard,
  then issues exactly one NLI call per surviving pair through the same
  `NLIScorer` Protocol that `Tier2NLIEdgeInferer` uses. Threshold
  ladder: contradiction or entailment at or above `0.70` emits the hard
  cross-doc edge; entailment in `[0.50, 0.70)` emits the soft
  `aligned_with` band so paraphrase-style equivalences surface without
  being mis-labelled as strict entailment. Cost contract: scorer calls
  grow linearly at `k * |A|` per ordered doc pair, i.e.
  `k * sum(|d|) * (n_docs - 1)` total, never quadratic. Endpoint
  identity reuses the persisted `Claim.id` verbatim so the
  optimal-transport engine (Phase 18) reads back the same ids it loaded
  from the claim store; emitted edges sort by
  `(type, src_id, dst_id)` for byte-stable diffs.

## [0.3.0] — 2026-05-23

End-to-end CLI release. Every UC playbook drives real documents
through the production stack via `ctrldoc <cmd> --profile thrifty
--target <md>`. The 5-doc smoke run lives at `runs/cli_smoke/`
with `SUMMARY.md` aggregating per-doc verdict counts.

### Summary table

| Slice | Subcommand → playbook | Module |
|---|---|---|
| S-110 | `TaskClient` ← Qwen2.5-7B via Ollama | `src/ctrldoc/orch/task_ollama.py` |
| S-111 | `BackendBundle` factory + `--profile` selector | `src/ctrldoc/backends.py` |
| S-112 | `ctrldoc ingest` ← L0 pipeline + sqlite-vec persistence | `src/ctrldoc/cli.py` |
| S-113 | `ctrldoc audit` ← UC2 `CoverageAuditPlaybook` | `src/ctrldoc/cli_audit.py` |
| S-114 | `ctrldoc qa` ← UC1 `QAPlaybook` | `src/ctrldoc/cli_qa.py` |
| S-115 | `ctrldoc review` ← UC4 `AnalyticalReviewPlaybook` | `src/ctrldoc/cli_review.py` |
| S-116 | `ctrldoc scan` ← UC5 `AnomalyScanPlaybook` | `src/ctrldoc/cli_scan.py` |
| S-117 | `ctrldoc map` ← UC6 `RelationMapPlaybook` (+ Mermaid) | `src/ctrldoc/cli_map.py` |
| S-118 | End-to-end smoke against 5 phase-0 docs + this tag | `scripts/aggregate_smoke.py`, `runs/cli_smoke/` |

### Fixes shipped during the smoke

- `OllamaTaskClient` now sets `num_ctx=16384` so the cacheable
  prefix + evidence pack fits without silent truncation. Default
  `num_ctx=2048` silently cut the LLM-judge call's view of the
  doc skeleton; every per-item judge response came back empty
  before this was raised. Now configurable via constructor.
- `SequentialBatchedRunner` (in `src/ctrldoc/cli_audit.py`) — a
  drop-in replacement for `BatchedTaskRunner` that issues one
  per-item Ollama call instead of one batched call. Local 7B
  models routinely fail the batched array shape; the sequential
  shim trades batching for prompt simplicity and accepts a
  per-item `on_error` callback so one bad parse doesn't abort
  the whole audit (the smoke uses an `Ambiguous` fallback).
- `_resolve_optional_ner()` in `backends.py` falls back to
  `StubNERTagger` when `gliner` isn't installed. Entity-based
  retrieval degrades gracefully; the rest of the pipeline still
  works.

### Smoke result

| # | checklist | items | Covered | Partial | NotCovered | Ambiguous |
|---|---|---:|---:|---:|---:|---:|
| 01 | `01_adversary_catalog.md` | 16 | 6 | 0 | 10 | 0 |
| 02 | `02_trust_assumptions.md` | 35 | 4 | 0 | 26 | 5 |
| 03 | `03_property_catalog.md` | 24 | 7 | 0 | 16 | 1 |
| 04 | `04_exclusions_and_L1_contract.md` | 60 | 2 | 0 | 45 | 13 |
| **Total** | | **135** | **19** | **0** | **97** | **19** |

No Anthropic calls fired in any audit (thrifty profile reserves
Opus for synthesis, which `coverage_audit` does not invoke).

## [0.2.8] — 2026-05-23

### Added

- `ctrldoc map --target <md>` is wired end-to-end through
  `RelationMapPlaybook` (UC6). New helpers in
  `src/ctrldoc/cli_map.py`:
  - `StoreEntityConceptExtractor` — pulls top-N entities by
    mention count, bounded by `--max-concepts` (default 10) so
    the O(M²) pair fan-out stays sane.
  - `BundleCoOccurrenceRetriever` — wraps the shared
    `BundleRetriever` (S-113) and sanitises BM25-hostile
    punctuation in concept names before retrieval.
  - `LLMRelationClassifier` — routes through the bundle's
    `local` tier (Ollama Qwen in thrifty), returns `None` for
    `unrelated` so the playbook drops the pair, resolves
    citation chunk_ids against the evidence pack, clamps
    confidence to `[0, 1]`.
  - `render_map_markdown` emits a Markdown adjacency table +
    Mermaid `graph LR` block with typed edges and standalone
    nodes.

### Notes

- Heuristic profile rejected for `map` (no LLM seam).
- Two obsolete map stub tests removed from `tests/test_cli.py`.

## [0.2.7] — 2026-05-23

### Added

- `ctrldoc scan --target <md>` is wired end-to-end through
  `AnomalyScanPlaybook` (UC5). Ingests the target inline, runs
  the deterministic detector battery (`HedgeWordDetector` +
  `EmptySummaryDetector` — §5.5 baseline), writes a Markdown
  triage report grouped by detector + a JSON payload. No LLM
  dependency: works in every profile including `heuristic`.
- `render_scan_markdown` emits per-detector groups sorted
  critical → warn → info, plus a per-severity summary table.

### Notes

- README Quickstart step 2 now invokes
  `ctrldoc --profile heuristic scan --target <gold_doc>` (the
  command previously took no args).
- One obsolete scan stub test removed from `tests/test_cli.py`.

## [0.2.6] — 2026-05-23

### Added

- `ctrldoc review <doc_type> --target <md>` is wired end-to-end
  through `AnalyticalReviewPlaybook` (UC4). Enumerates the
  canonical 5-lens set via `HeuristicLensGenerator`, fans out one
  LLM sweep per lens through the bundle's `local` tier (Ollama
  Qwen in thrifty), then a single synthesis call routed through
  the `opus` tier — the only Opus call per playbook run in
  thrifty mode.
- `LLMLensSweeper` (`src/ctrldoc/cli_review.py`) implements the
  `LensSweeper` protocol against the shared `BundleRetriever`.
  Hallucinated citation chunk_ids are silently dropped; the
  sweeper short-circuits to an empty result when retrieval
  returns no spans.
- `render_review_markdown` emits the synthesis narrative
  (headline + summary + sections) then per-lens groups ordered
  by severity (critical → warn → info), plus a per-lens summary
  table at the foot.

### Notes

- Heuristic profile rejected for `review` (no LLM seam).
- Two obsolete stub-style review tests removed from
  `tests/test_cli.py`.

## [0.2.5] — 2026-05-23

### Added

- `ctrldoc qa <query> --target <md>` is wired end-to-end through
  `QAPlaybook` (UC1). Ingests the target inline, runs the bundle's
  retriever + planner + executor + RRF + reranker, generates an
  answer via the `local` task tier, decomposes into atomic claims,
  and runs each claim through `ClaimVerifier` (NLI + judge with
  one repair pass).
- `VerifierRetriever` (`src/ctrldoc/cli_qa.py`) adapts the shared
  `BundleRetriever` (S-113) to the verifier's `Retriever` protocol.
  `depth` is currently a no-op for both `normal` and `broad`;
  documented as a future-widening hook.
- `render_qa_markdown` renders the Markdown answer + per-claim
  verification table (verified, confidence, NLI, judge,
  citations) plus a citation appendix with chunk-id snippets.

### Notes

- Heuristic profile rejected for `qa` (no LLM seam in heuristic).
- Three obsolete stub-style qa tests removed from
  `tests/test_cli.py`.

## [0.2.4] — 2026-05-23

### Added

- `ctrldoc audit --checklist <md> --target <md>` is wired end-to-end
  through `CoverageAuditPlaybook` (UC2). Checklist items are
  extracted via a deterministic Markdown-section parser
  (`parse_checklist_markdown`): each `## H2` / `### H3` heading +
  first paragraph becomes one item, with `topic_key` inheriting
  from the nearest parent section.
- `BundleRetriever` (`src/ctrldoc/cli_audit.py`) adapts a
  `BackendBundle` to the `QARetriever` / `CoverageRetriever`
  protocols: bundle planner → executor → RRF fusion → bundle
  reranker → `build_evidence_pack`. Reused by qa / review /
  map in later slices.
- `render_coverage_markdown` groups verdicts (`Covered`,
  `Partial`, `NotCovered`, `Ambiguous`) with a summary table and
  per-item citations rendered as `[chunk_id] snippet`.
- The CLI rejects `--profile heuristic` for `audit` with a clear
  error — the playbook needs an LLM seam and heuristic mode has
  no `task_client_router`.

### Notes

- Per-item batched judging routes to the bundle's `local` tier
  (Ollama Qwen2.5-7B in thrifty mode); Opus is reserved for
  synthesis calls (none in coverage_audit yet).

## [0.2.3] — 2026-05-23

### Added

- `ctrldoc ingest <path>` is wired end-to-end through the
  `BackendBundle`: heuristic profile keeps the deterministic
  in-memory L0 substrate (matches the S-090 canary baseline);
  thrifty / production profiles drive the L0 pipeline through
  `OllamaEmbedder` + `SQLiteStore` + `sqlite-vec` + Tantivy BM25
  and persist a per-doc index at
  `<runs_path>/indexes/<doc_hash>.{db,vec.db,bm25/}`.
- Global CLI options on a `@app.callback`: `--config <path>`
  (falls back to a built-in default when the file is absent),
  `--profile heuristic|thrifty|production` (default `thrifty`),
  `--format markdown|json|both` (default `markdown`), and
  `--max-cost-usd FLOAT` (default 5.00).
- `.env` is parsed on every CLI invocation and entries are
  promoted to `os.environ` (existing values are not overwritten;
  the value is never echoed back).
- Per-run artefacts now land at `<runs_path>/<run_id>/report.md`
  (Markdown) and `<runs_path>/<run_id>/result.json` (structured
  payload incl. signature + signature_hash). Legacy
  `<doc_id>__ingest_signature.json` + `__ingest_stats.json` files
  remain next to the run dir so the S-090 canary path stays
  whole.

### Notes

- The Quickstart in README.md now uses `ctrldoc --profile
  heuristic ingest …` so the no-credentials install path still
  works under the new default thrifty profile.

## [0.2.2] — 2026-05-23

### Added

- `BackendBundle` + `build_bundle(config, profile)` in
  `src/ctrldoc/backends.py` — typed wiring for the three runtime
  profiles. `heuristic` returns deterministic reference impls (no
  LLM, no model loading). `thrifty` uses the production retrieval
  / verifier infra (Ollama embedder, sqlite-vec, BGE reranker,
  fastcoref, GLiNER, DeBERTa NLI) but keeps every per-item /
  per-claim LLM seam on local Qwen2.5-7B; the `task_client_router`
  still binds Opus on the `opus` tier so playbooks can spend
  exactly one synthesis call per run. `production` upgrades the
  planner / claim decomposer / summarizer / judge to Anthropic-
  backed equivalents. Heavy backends are lazy-imported per
  profile so `heuristic` mode never pulls Ollama or transformers.
  `build_bundle_from_toml(path, profile)` is the one-line CLI
  entry point (SPEC-REF §4.5, §4.7).

## [0.2.1] — 2026-05-23

### Added

- `OllamaTaskClient` in `src/ctrldoc/orch/task_ollama.py` — implements
  the `TaskClient` protocol against `qwen2.5:7b-instruct-q4_K_M` via
  a local Ollama service, mirroring `AnthropicTaskClient`. Enables
  the tier-1 (`local`) route through `TaskClientRouter` so the
  `BatchedTaskRunner` can fan per-item judging out to the local 7B
  while reserving Opus for synthesis (SPEC-REF §4.5).

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
