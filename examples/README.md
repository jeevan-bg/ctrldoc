# Examples

Runnable, hermetic walkthroughs of every UC playbook. Each file uses
deterministic stubs in place of a real LLM client, so you can read
the wiring pattern and run the script with no API key.

| File | Use case |
|---|---|
| `01_qa.py` | UC1 trustworthy QA (retrieve, generate, decompose, verify) |
| `02_coverage_audit.py` | UC2 coverage audit with topic-clustered batched judging |
| `03_quality_audit.py` | UC3 quality audit (criteria generation delegates to coverage) |
| `04_analytical_review.py` | UC4 analytical review (lens fan-out + synthesis reduce) |
| `05_anomaly_scan.py` | UC5 anomaly scan with the deterministic detector battery |
| `06_relation_map.py` | UC6 concept-pair classification into a relation graph |

Run any example directly:

```bash
python examples/01_qa.py
```

Or invoke the CLI against the bundled synthetic doc:

```bash
ctrldoc ingest tests/fixtures/synthetic/gold_doc.md -o ./runs --doc-id aurora
ctrldoc scan
```

The CLI is exercised in `tests/test_cli.py`; each example here is
exercised end-to-end in `tests/test_examples_smoke.py` so a
regression in any playbook surface fails the build.
