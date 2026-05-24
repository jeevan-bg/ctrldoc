"""L2.5 / L5 universal-substrate operations.

The `ops` package owns the post-ingest operations that read the claim
graph and the workspace primitive: workspace CRUD today (S-134),
optimal-transport coverage / compare / merge / list_check later (Phase
18). It is the seam the CLI commands route through as the v1
substrate replaces the per-playbook L5 code paths in
`ctrldoc.playbooks/` (deleted at S-146).

SPEC-REF: §6.6, §6.7, §9
"""
