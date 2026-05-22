# Testing strategy

Tests in `ctrldoc` are organized around the spec's 14-family taxonomy (see [SPEC.md](SPEC.md) §8.6). Every test belongs to at least one family; the family is declared with a pytest marker.

## Test kinds used in this project

`ctrldoc` mixes seven kinds of testing. Each test is one or more.

| Kind | Tooling | Purpose |
|---|---|---|
| **Unit** | `pytest` | A function or class in isolation; in-memory; fast; no I/O. |
| **Integration** | `pytest` + temp SQLite | Two or more layers wired together against a real DB and real fixtures. |
| **Property-based** | `hypothesis` | Generate inputs across a domain; assert invariants. Used on chunkers, fusers, planners, verifiers. |
| **End-to-end** | `pytest` on the synthetic gold doc | All six playbooks against `gold.yaml`. The ground truth oracle. |
| **Adversarial** | `pytest -m family_adversarial` | Source-doc attacks: prompt injection, homoglyphs, zero-width, RTL, malformed PDFs, jailbreak strings. |
| **Performance / cost** | `pytest -m family_performance_cost` + baselines | Token, latency, RAM, and dollar regressions. CI fails on >20% drift. |
| **Determinism / replay** | `lens replay <run_id>` + snapshot tests | Same input, same seed → byte-identical output. Trace-replay reproduces every LLM call from cached responses. |

Post-MVP: mutation testing (`mutmut`) and chaos testing on the orchestrator's failure-injection layer.

## Markers — every test declares its family

| Marker | Family | Catches |
|---|---|---|
| `family_ingest_completeness` | 1 | Lost chunks, broken parsing |
| `family_niah` | 2 | Retrieval blind spots |
| `family_synthetic_gold` | 3 | Playbook correctness end-to-end |
| `family_reachability` | 4 | Orphaned content (sections never surfaced) |
| `family_negative_refusal` | 5 | Fabrication / refusal correctness |
| `family_referential_integrity` | 6 | Hallucinated citations |
| `family_robustness` | 7 | Parser/chunker on edge inputs |
| `family_adversarial` | 8 | Source-doc attacks |
| `family_verifier_calibration` | 9 | FP/FN rates on verifier |
| `family_determinism` | 10 | Silent drift |
| `family_performance_cost` | 11 | Token/latency/RAM regressions |
| `family_failure_resilience` | 12 | Crashes, corruption, partial data |
| `family_incremental_update` | 13 | Stale or broken state after edits |
| `family_concurrency` | 14 | Race conditions, async cancellation |

Run a single family:

```bash
pytest -m family_niah
```

## The synthetic gold fixture

`tests/fixtures/synthetic/` contains a hand-built ~1.1k-token document (Aurora distributed cache) with paired ground truth in `gold.yaml`. It is the canary for every playbook. Before any playbook ships, its outputs must be consistent with `gold.yaml`.

The fixture is deliberately designed to exercise hard cases:

- nested sections with native IDs (§1, §2.1) for structural parsing,
- named entities with aliases for coreference and the entity index,
- a deliberate contradiction across two sections for `analytical_review`,
- a deliberate definition gap (a term used but not specified) for `anomaly_scan`,
- directed relations between concepts for `relation_map`,
- known answerable questions for `qa`,
- known unanswerable questions for `qa` refusal,
- a six-item threat checklist that drives `coverage_audit`,
- needle-in-haystack sentinels at known locations for retrieval recall.

`gold.yaml` declares the expected outputs in machine-readable form so all six playbooks can be scored without human judgment.

## Real-world fixtures

`scripts/fetch_fixtures.sh` downloads three real public-domain documents to `tests/fixtures/downloaded/` (gitignored):

1. A technical RFC.
2. An open-access scientific paper.
3. A long-form public-domain text.

These exercise prose, structured technical content, and citations at scale.

## LLM-as-judge eval

`analytical_review` and `anomaly_scan` produce partly subjective outputs that cannot be fully scored against gold. See [SPEC.md](SPEC.md) §8.7. The judge model must differ from the producer; bias controls (blind shuffle, position swap, multi-seed averaging) are mandatory; a 10% human-labeled sample anchors agreement.

## Cost / latency gates

Per-playbook baselines live in `tests/baselines/`. CI fails on > 20 % regression. See [SPEC.md](SPEC.md) §8.4 for targets.

## CI gates (every PR must pass)

1. Full suite green.
2. `ruff check` and `ruff format --check` clean.
3. `mypy --strict src/ctrldoc/` clean.
4. Public-leak scan clean.
5. Spec-trace coverage: every MVP-required spec section has ≥ 1 mapped test.

## Coverage rule (hard)

Before any playbook ships, it must have at least one test from every applicable family. See [SPEC_TRACE.md](SPEC_TRACE.md) for current coverage.
