# v1 walkthroughs

Runnable walkthroughs of the v1 universal-substrate surface. Each
example is hermetic — no LLM, no Ollama, no network — so they run
from a clean clone without any credentials.

| Script | What it shows | SPEC |
| --- | --- | --- |
| [`01_workspace.py`](01_workspace.py) | Build a workspace (L2.5 primitive). | §6.7 |
| [`02_coverage_transport.py`](02_coverage_transport.py) | Per-target-claim `Covered` / `Missing` verdicts via the optimal-transport reduction. | §6.6 |
| [`03_merge_transport.py`](03_merge_transport.py) | Lossless multi-doc merge: Galois floor first, NLI fallback for paraphrases. | §6.6 |

```bash
python examples/v1/01_workspace.py
python examples/v1/02_coverage_transport.py
python examples/v1/03_merge_transport.py
```

To drive the operations against a real NLI backend, swap the
synthetic scorer in each example for any `NLIScorer` Protocol
implementation — `ctrldoc.extract.isotonic_calibration.CalibratedNLIScorer`
wraps any raw backend with the §6.5 calibration pipeline (release
gate `ECE ≤ 0.05`).

For the v0.3 per-playbook walkthroughs (still functional) see the
parent [`examples/`](../) directory.
