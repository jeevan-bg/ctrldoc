# Synthetic gold fixture

This directory contains a hand-built reference document and its ground-truth annotations.

- `gold_doc.md` — a fictional ~1.1k-token specification ("Aurora distributed cache") with seeded structure, entities, contradictions, gaps, and relations.
- `gold.yaml` — machine-readable ground truth for every playbook.

The fixture is deliberately small enough to run all six playbooks against in CI and rich enough to exercise the hard cases. New tests should consult `gold.yaml` rather than hand-coding expected values, so the oracle stays in one place.
