# Migrating from ctrldoc v0.3 to v1.0

ctrldoc v1.0 reframes the system as a **universal claim-graph
substrate**. The v0.3 per-use-case playbook layer is gone; every
operation now flows through one claim graph + workspace primitive plus
the optimal-transport core. This guide is the shortest path from a
v0.3 install to a green v1 install.

## TL;DR

1. **Re-ingest every document.** Storage `SCHEMA_VERSION` bumped from
   `"0.1.0"` to `"0.2.0"`; v0.3 indexes refuse to open under v1.
2. **Update imports.** The deprecated v0.3 package is gone; every
   former symbol lives under `ctrldoc.ops.*` along its CLI-aligned
   name (see the rename table below).
3. **Adopt the new commands.** `workspace`, `coverage`, `compare`,
   `merge`, `list-check`, `graph`, `schema`, `calibration`, `ledger`,
   and `mcp serve` are new. The v0.3 commands (`ingest`, `qa`, `scan`,
   `map`, `audit`, `review`) keep working unchanged.

## Breaking changes

### Storage: `SCHEMA_VERSION` 0.1.0 → 0.2.0

`SQLiteStore` now provisions six new tables — `claims`, `concepts`,
`typed_edges`, `workspaces`, `cross_doc_edges`, `verdict_ledger` —
alongside the v0.3 chunk / section / entity layout. The version gate
*is* the migration boundary: there is no in-place data migration. A
v0.3 index opened against v1 will raise on the version check.

**Action:** re-ingest. The L0 pipeline is fully backward-compatible at
the input layer (a v0.3 ingest command still works against the same
documents).

```bash
ctrldoc ingest <doc.md> --output-dir ./runs --doc-id <id>
```

### Package removed: former v0.3 playbook surface

The v0.3 `playbooks/` package was the per-use-case L5 code path. v1
collapses every operation onto the universal claim graph + workspace
surface. The package is gone from disk; every previous symbol is
re-homed under `ctrldoc.ops.*` along its CLI-aligned name.

| v0.3 import (gone)                       | v1 import (new home)                    |
| ---------------------------------------- | --------------------------------------- |
| `from ctrldoc.playbooks.anomaly import …`  | `from ctrldoc.ops.scan import …`        |
| `from ctrldoc.playbooks.qa import …`       | `from ctrldoc.ops.qa import …`          |
| `from ctrldoc.playbooks.review import …`   | `from ctrldoc.ops.review import …`      |
| `from ctrldoc.playbooks.relations import …`| `from ctrldoc.ops.map import …`         |
| `from ctrldoc.playbooks.coverage import …` | `from ctrldoc.ops.audit import …`       |
| `from ctrldoc.playbooks.quality import …`  | `from ctrldoc.ops.quality import …`     |

Note the one rename that is not a straight copy: the v0.3
`coverage_audit` symbol now lives under `ctrldoc.ops.audit`. The bare
`ctrldoc.ops.coverage` name is taken by the v1 optimal-transport
coverage operation (§6.6) — see the new-surface section below.

**Action:** a single find-and-replace per import row above is
sufficient. The class names and behaviour are unchanged.

### CLI surface: unchanged commands, new flags

The v0.3 commands (`ingest`, `qa`, `scan`, `map`, `audit`, `review`)
still emit the same Markdown report + JSON payload to
`runs/<run_id>/`. They are now thin adapters over the universal
substrate; behaviour is preserved.

## New v1 surface

### `ctrldoc workspace {create | add | list | info}`

Workspaces are the L2.5 primitive: N documents sharing one concept
lattice. Build one once, then run any cross-doc operation against it.

```bash
ctrldoc workspace create due-diligence
ctrldoc workspace add due-diligence company-spec.pdf
ctrldoc workspace add due-diligence security-policy.pdf
ctrldoc workspace info due-diligence
```

### `ctrldoc coverage --workspace <name> --target <doc> --source <doc>`

Per-target-claim verdicts via the §6.6 optimal-transport reduction.
Every line is `(claim, verdict, calibrated_confidence,
citations_in_both_docs)`. The v1 coverage operation replaces the v0.3
`audit` playbook as the workhorse cross-doc check; the v0.3 `audit`
command is preserved unchanged for callers that depend on its
single-doc framing.

### `ctrldoc compare` and `ctrldoc merge`

`compare` emits per-cluster `{StrengthA, StrengthB, Gap}` verdicts;
`merge` returns one cluster per equivalence class with a strongest
representative via the Galois GLB. Both reduce to optimal transport
on the claim-pair graph.

### `ctrldoc list-check`

Per-item verdicts against a document — `list_check(items, doc)`
under §6.6's "list parsed as a tiny doc" framing.

### `ctrldoc ledger {list | show | replay}`

Append-only verdict ledger; every L4 verdict is replayable inside the
§6.5 ±0.02 determinism gate.

### `ctrldoc mcp serve`

Model Context Protocol server over stdio (JSON-RPC 2.0). Exposes the
13-tool §6.10 surface (`lookup_concept`, `traverse`, `entails`,
`subsumes`, `optimal_transport`, `coverage`, `compare`, `merge`,
`list_check`, `map`, `qa`, `calibration`, plus `get_claim`) so any
MCP-aware host — Claude Desktop, Claude CLI — can drive the substrate
directly.

```bash
ctrldoc mcp serve
```

### `ctrldoc graph {show | query}` and `ctrldoc schema {show | pin}`

Surface the per-doc claim graph and the per-doc induced schema (the
YAML cache from the §6.4 schema co-induction loop) for inspection and
manual pinning.

### `ctrldoc calibration`

One-shot ECE measurement for any NLI backend; the v1 release gate
blocks if `ECE > 0.05` per backend.

## Conceptual changes worth understanding

### Universal claim tuple is now the logic floor

Every extractor — heuristic, Tier-2 SVO, LLM — produces tuples of
shape `Claim = (subject, predicate, object, polarity, modality,
qualifier, span_refs, confidence)` per §6.2. Contradiction is a
polarity flip; stronger-than is qualifier ordering. The tuple is
always extracted, even when adapter-specific extraction fails — this
is non-negotiable #6 in §13.

### Edges carry calibrated confidence

Every edge in the v1 graph has a `confidence ∈ [0, 1]`. Sources are
heuristic / NLI / LLM. The §6.5 calibration pipeline (paraphrase
voting + isotonic regression) ships calibrated confidences with a
release gate `ECE ≤ 0.05` per backend. Boolean edges are gone from
shipped output (non-negotiable #8).

### Optimal transport replaces per-op code paths

`compare`, `coverage`, `merge`, and `list_check` collapse into one
min-cost-flow / Sinkhorn engine on the claim-pair edges weighted by
`1 - NLI_entail`. The `merge` operation respects a hard loss
invariant — every input claim maps to exactly one output cluster
(non-negotiable #10).

### Workspaces share one concept lattice

N documents in a workspace share the same canonical concept set
(§6.3 Galois lattice + §6.8 entity resolution). Cross-doc edges
(`aligned_with`, `entails_across`, `contradicts_across`,
`stronger_than`) are lazy, cached, and linear in `|A| × k` via
candidate retrieval — never quadratic.

### MCP is the integration

The v1 server is the canonical way to drive the substrate from
Claude (Desktop, CLI, or any MCP-aware host). Forced tool calls
only — no free-form L4 reasoning (non-negotiable #12).

## What's gone

The v0.3 spec is archived; the live `docs/SPEC.md` is the v1 spec.
v0.3 pillars (§4.1: stateless tasks + shared prompt cache) and v0.3
non-negotiables (#1–5) are reproduced verbatim inside the v1 spec —
nothing about the v0.3 substrate's guarantees was relaxed. The v0.3
release notes live in `CHANGELOG.md`.

## Where to look next

- `CHANGELOG.md` — the per-release notes, including the full v1.0.0
  surface roll-up.
- `docs/SPEC.md` — the live v1 normative specification.
- `docs/ARCHITECTURE.md` — the system overview, refreshed for the v1
  9-layer stack.
- `examples/v1/` — runnable v1 walkthroughs of the workspace,
  coverage-via-transport, and merge surfaces, against in-memory
  fixtures (no LLM credentials required).
- `docs/decisions/INDEX.md` — every architectural decision recorded
  during the v1 build.
