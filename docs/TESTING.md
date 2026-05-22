# Testing strategy

Tests in `ctrldoc` are organized around the spec's 14-family taxonomy (see [SPEC.md](SPEC.md) §8.6). Every test belongs to at least one family; the family is declared with a pytest marker.

## Markers

| Marker | Family | Purpose |
|---|---|---|
| `family_ingest_completeness` | 1 | Lost chunks, broken parsing |
| `family_niah` | 2 | Needle in haystack — retrieval blind spots |
| `family_synthetic_gold` | 3 | End-to-end correctness on the synthetic gold doc |
| `family_reachability` | 4 | Orphaned content |
| `family_negative_refusal` | 5 | Fabrication / refusal correctness |
| `family_referential_integrity` | 6 | Hallucinated citations |
| `family_robustness` | 7 | Parser/chunker on edge inputs |
| `family_adversarial` | 8 | Source-doc attacks |
| `family_verifier_calibration` | 9 | Verifier misjudgment |
| `family_determinism` | 10 | Silent drift |
| `family_performance_cost` | 11 | Token/latency/RAM regressions |
| `family_failure_resilience` | 12 | Crashes, corruption, data loss |
| `family_incremental_update` | 13 | Stale or broken state |
| `family_concurrency` | 14 | Race conditions |

Run a single family:

```bash
pytest -m family_niah
```

## The synthetic gold fixture

`tests/fixtures/synthetic/` contains a hand-built ~5k-token document with paired ground-truth annotations in `gold.yaml`. It is the canary for every playbook. Before any playbook ships, it must produce outputs consistent with `gold.yaml` on every relevant assertion.

The fixture is deliberately designed to exercise the hard cases:

- nested sections with native IDs (§1, §2.1, etc.) for structural parsing,
- named entities with aliases for coreference and the entity index,
- a deliberate contradiction across two sections for `analytical_review`,
- a deliberate gap (a term introduced but not defined) for `anomaly_scan`,
- a directed relation between two concepts for `relation_map`,
- known answerable questions for `qa`,
- known unanswerable questions for `qa` refusal,
- a checklist of seeded "requirements" so a paired threats-style doc can drive `coverage_audit`.

`gold.yaml` declares the expected outputs in a machine-readable form so all six playbooks can be scored without human judgment.

## Real-world fixtures

`scripts/fetch_fixtures.sh` downloads three real, public-domain documents to `tests/fixtures/downloaded/` (gitignored):

1. A technical RFC.
2. An open-access scientific paper.
3. A long-form public-domain text.

These exercise the system on prose, structured technical content, and citations.

## Property-based testing

Hypothesis is used for chunkers, mergers, retrievers, and verifiers — anywhere the input domain is large and edge cases are easy to miss.

## Performance and cost gates

Per-playbook cost and latency budgets are declared in `tests/baselines/`. CI fails if a regression exceeds 20%.

## LLM-as-judge eval

For partly subjective outputs (`analytical_review`, `anomaly_scan`), see [SPEC.md](SPEC.md) §8.7. Judge model must differ from the producer; bias controls (blind shuffle, position swap, multi-seed averaging) are mandatory; a 10% human-labeled sample anchors agreement.

## CI gates (every PR must pass)

1. Tests green.
2. `ruff check` and `ruff format --check` clean.
3. `mypy --strict src/ctrldoc/` clean.
4. Public-leak scan clean (no internal language in public files).
5. Spec-trace coverage: every spec section listed as MVP-required has at least one mapped test.
