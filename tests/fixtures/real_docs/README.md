# Real-doc corpus

Realistically-shaped documents spanning the §16 doc-type axes used by
the v1 substrate's end-to-end shakedown. Every doc is hand-built,
small enough to run through the full L0 pipeline in seconds, and
committed to the repo so the smoke is hermetic.

Doc-type axes covered:

- **spec** — `spec_lighthouse.md`
- **legal** — `legal_terms.md`
- **academic** — `academic_paper.md`
- **educational** — `educational_guide.md`
- **narrative** — `narrative.md`
- **spec_vs_impl** — `pair_spec.md` + `pair_impl.md` (linked by `pair_id: tideline`)

`MANIFEST.yaml` is the authoritative oracle: doc IDs, types, file
paths, and per-doc summaries live there. New tests should read the
manifest rather than hard-coding paths.

The smoke entrypoint is `scripts/real_doc_smoke.sh`. It drives every
doc through `ctrldoc ingest` and `ctrldoc scan` on the heuristic
profile and builds a workspace from the spec-vs-impl pair. The
script writes a `summary.json` so test code can inspect outcomes
mechanically.
