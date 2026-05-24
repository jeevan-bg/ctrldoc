"""L1.5 claim-graph extraction.

The `extract` package owns the per-doc claim-graph construction: the
deterministic floor (`tier1`) plus the LLM-mediated typed-edge layers
that build on it. See `docs/SPEC.md` §6.4 for the schema co-induction
algorithm this package is the floor of.

SPEC-REF: §6.4 (claim-graph extraction)
"""
