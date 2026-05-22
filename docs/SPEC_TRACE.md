# Spec → Code Traceability Matrix

Every MVP-required section of `docs/SPEC.md` maps to at least one slice. This file is the source of truth for spec coverage and is mechanically checked by `scripts/spec_trace_check.py`.

Row format:

```
| §spec | slice | src | tests | status |
```

`status` is one of `pending`, `partial`, `covered`.

## Coverage table (filled as slices land)

| §X.Y | Slice | Source files | Test files | Status |
|---|---|---|---|---|
| §3 (architecture) | S-001 .. S-103 | src/ctrldoc/ | tests/ | partial |
| §4.0 (data model) | S-010..S-013 | src/ctrldoc/models.py | tests/test_models.py | pending |
| §4.1 (ingest) | S-030..S-039 | src/ctrldoc/ingest/ | tests/test_ingest_*.py | pending |
| §4.2 (multi-view index) | S-020..S-026 | src/ctrldoc/store/ | tests/test_store_*.py | pending |
| §4.3 (retrieval) | S-040..S-046 | src/ctrldoc/retrieval/ | tests/test_retrieval_*.py | pending |
| §4.4 (verifier) | S-050..S-055 | src/ctrldoc/verify/ | tests/test_verify_*.py | pending |
| §4.5 (orchestrator) | S-060..S-067 | src/ctrldoc/orch/ | tests/test_orch_*.py | pending |
| §4.7 (cross-cutting) | S-002..S-007, S-014 | src/ctrldoc/{config,trace,budget,provenance,tokenizer}.py | tests/test_*.py | pending |
| §5.1 (UC1 qa) | S-070 | src/ctrldoc/playbooks/qa.py | tests/test_qa.py | pending |
| §5.2 (UC2 coverage) | S-071 | src/ctrldoc/playbooks/coverage.py | tests/test_coverage.py | pending |
| §5.3 (UC3 quality) | S-072 | src/ctrldoc/playbooks/quality.py | tests/test_quality.py | pending |
| §5.4 (UC4 review) | S-073 | src/ctrldoc/playbooks/review.py | tests/test_review.py | pending |
| §5.5 (UC5 anomaly) | S-074 | src/ctrldoc/playbooks/anomaly.py | tests/test_anomaly.py | pending |
| §5.6 (UC6 relations) | S-075 | src/ctrldoc/playbooks/relations.py | tests/test_relations.py | pending |
| §8.1 (eval sets) | S-080..S-085 | tests/eval/ | tests/eval/ | pending |
| §8.5 (adversarial) | S-086 | tests/adversarial/ | tests/adversarial/ | pending |
| §8.6 family 1 | S-038 | src/ctrldoc/ingest/ | tests/families/test_ingest_completeness.py | pending |
| §8.6 family 2 | S-046 | src/ctrldoc/retrieval/ | tests/families/test_niah.py | pending |
| §8.6 family 3 | S-070..S-075 | src/ctrldoc/playbooks/ | tests/families/test_synthetic_gold.py | pending |
| §8.6 family 4 | S-073 | src/ctrldoc/playbooks/review.py | tests/families/test_reachability.py | pending |
| §8.6 family 5 | S-054 + S-082 | src/ctrldoc/verify/ | tests/families/test_refusal.py | pending |
| §8.6 family 6 | S-044 | src/ctrldoc/retrieval/ | tests/families/test_referential_integrity.py | pending |
| §8.6 family 7 | S-030..S-033 | src/ctrldoc/ingest/ | tests/families/test_robustness.py | pending |
| §8.6 family 8 | S-086 | n/a | tests/families/test_adversarial.py | pending |
| §8.6 family 9 | S-055 | src/ctrldoc/verify/ | tests/families/test_verifier_calibration.py | pending |
| §8.6 family 10 | S-087 | n/a | tests/families/test_determinism.py | pending |
| §8.6 family 11 | S-088 | n/a | tests/families/test_perf_cost.py | pending |
| §8.6 family 12 | S-066 | src/ctrldoc/orch/ | tests/families/test_resilience.py | pending |
| §8.6 family 13 | S-039 | src/ctrldoc/ingest/ | tests/families/test_incremental.py | pending |
| §8.6 family 14 | S-064 | src/ctrldoc/orch/ | tests/families/test_concurrency.py | pending |
| §8.7 (LLM judge) | S-089 | tests/eval/judge/ | tests/eval/judge/ | pending |
