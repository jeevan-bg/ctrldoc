"""L2.5 / L5 universal-substrate operations.

The `ops` package owns every post-ingest operation: the v1 substrate
itself (`workspace`, `cross_doc_edges`, `coverage`, `compare`, `merge`,
`transport`) plus the v0.3 CLI-aligned operations (`scan`, `qa`,
`review`, `map`, `audit`, `quality`). One package, one import path;
every CLI command routes through `ctrldoc.ops.*` and nothing else.

SPEC-REF: §6 (universal substrate kills the playbook layer), §6.6, §6.7, §9
"""
