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
| §12 (build order) | S-001 | (toolchain) | tests/test_toolchain.py | covered |
| §4.0 (data model) | S-010..S-013 | src/ctrldoc/models.py | tests/test_models_*.py | partial |
| §4.0 (Chunk/Section/Span) | S-010 | src/ctrldoc/models.py | tests/test_models_chunk_section_span.py | covered |
| §4.0 (Entity/EntityGlossary) | S-011 | src/ctrldoc/models.py | tests/test_models_entity.py | covered |
| §4.0 (EvidencePack/Claim/Verdict/Finding/RelationEdge) | S-012 | src/ctrldoc/models.py | tests/test_models_output.py | covered |
| §4.0 (PlaybookOutput) | S-013 | src/ctrldoc/models.py | tests/test_models_playbook_output.py | covered |
| §4.7 (versioning / integrity hash) | S-014 | src/ctrldoc/versioning.py | tests/test_versioning.py | covered |
| §4.1 (ingest) | S-030..S-039 | src/ctrldoc/ingest/ | tests/test_ingest_*.py | partial |
| §4.1 (Markdown parser) | S-030 | src/ctrldoc/ingest/parser.py | tests/test_parser_markdown.py | covered |
| §4.1 (PDF parser) | S-031 | src/ctrldoc/ingest/pdf.py | tests/test_parser_pdf.py | covered |
| §4.1 (code parser, Python) | S-032 | src/ctrldoc/ingest/code.py | tests/test_parser_code.py | covered |
| §4.1 (semantic chunker) | S-033 | src/ctrldoc/ingest/chunker.py | tests/test_chunker.py | covered |
| §4.1 (coref — interface) | S-034 | src/ctrldoc/ingest/coref.py | tests/test_coref.py | covered |
| §4.1 (coref — fastcoref backend) | S-034b | src/ctrldoc/ingest/coref.py | tests/test_coref.py | blocked |
| §4.1 (NER + canonicalisation) | S-035 | src/ctrldoc/ingest/ner.py, src/ctrldoc/ingest/ner_gliner.py | tests/test_ner.py, tests/test_ner_gliner.py | covered |
| §4.1/§4.2 (embedder — interface) | S-036 | src/ctrldoc/ingest/embedder.py | tests/test_embedder.py | covered |
| §4.1 (embedder — BGE-M3 via Ollama) | S-036b | src/ctrldoc/ingest/embedder.py | tests/test_embedder.py | blocked |
| §4.1/§3.1 (section summariser) | S-037 | src/ctrldoc/ingest/summarizer.py, src/ctrldoc/ingest/summarizer_anthropic.py | tests/test_summarizer.py, tests/test_summarizer_anthropic.py | covered |
| §4.1 (ingest end-to-end pipeline) | S-038 | src/ctrldoc/ingest/pipeline.py | tests/families/test_ingest_completeness.py | covered |
| §8.6 family 1 (ingest completeness) | S-038 | src/ctrldoc/ingest/ | tests/families/test_ingest_completeness.py | covered |
| §4.1 / §8.6 family 13 (incremental update) | S-039 | src/ctrldoc/ingest/pipeline.py, src/ctrldoc/store/ | tests/families/test_incremental_update.py | covered |
| §4.2 (multi-view index) | S-020..S-026 | src/ctrldoc/store/ | tests/test_store_*.py | partial |
| §10/§13 (Store protocol) | S-020 | src/ctrldoc/store/__init__.py, src/ctrldoc/store/memory.py | tests/test_store.py | covered |
| §4.2 (SQLite tables) | S-021 | src/ctrldoc/store/sqlite.py | tests/test_store_sqlite.py | covered |
| §4.2 (dense vectors — interface) | S-022 | src/ctrldoc/store/vectors.py | tests/test_vector_index.py | covered |
| §4.2 (dense vectors — sqlite-vec) | S-022b | src/ctrldoc/store/sqlite.py | tests/test_store_sqlite.py | blocked |
| §4.2 (BM25 lexical) | S-023 | src/ctrldoc/store/bm25.py | tests/test_bm25.py | covered |
| §4.2 (entity inverted index) | S-024 | src/ctrldoc/store/{__init__,memory,sqlite}.py | tests/test_store_entity_index.py | covered |
| §3.1/§4.2 (cacheable prefix) | S-025 | src/ctrldoc/assembler.py | tests/test_skeleton_glossary.py | covered |
| §4.7 (index integrity + backup) | S-026 | src/ctrldoc/store/sqlite.py | tests/test_store_integrity.py | covered |
| §4.3 (retrieval) | S-040..S-046 | src/ctrldoc/retrieval/ | tests/test_retrieval_*.py | partial |
| §4.3 (retrieval DSL) | S-040 | src/ctrldoc/retrieval/dsl.py | tests/test_retrieval_dsl.py | covered |
| §4.3 (retrieval executor) | S-041 | src/ctrldoc/retrieval/executor.py | tests/test_retrieval_executor.py | covered |
| §4.3 (Reciprocal Rank Fusion) | S-042 | src/ctrldoc/retrieval/fusion.py | tests/test_retrieval_fusion.py | covered |
| §4.4 (verifier) | S-050..S-055 | src/ctrldoc/verify/ | tests/test_verify_*.py | pending |
| §4.5 (orchestrator) | S-060..S-067 | src/ctrldoc/orch/ | tests/test_orch_*.py | pending |
| §4.7 (cross-cutting) | S-002..S-007, S-014 | src/ctrldoc/{config,trace,budget,provenance,tokenizer}.py | tests/test_*.py | partial |
| §4.7 (pre-commit gates) | S-002 | .pre-commit-config.yaml | tests/test_pre_commit_config.py | covered |
| §4.7 (tokenizer) | S-003 | src/ctrldoc/tokenizer.py | tests/test_tokenizer.py | covered |
| §4.7 (configuration) | S-004 | src/ctrldoc/config.py | tests/test_config.py | covered |
| §4.7 (provenance) | S-005 | src/ctrldoc/provenance.py | tests/test_provenance.py | covered |
| §4.7 (observability) | S-006 | src/ctrldoc/trace.py | tests/test_trace.py | covered |
| §4.7 (cost / budget) | S-007 | src/ctrldoc/budget.py | tests/test_budget.py | covered |
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
