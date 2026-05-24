"""v1 workspace primitive — create, add, info, list.

Workspaces are the L2.5 substrate primitive: N documents sharing one
canonical concept lattice. Once a workspace is built every cross-doc
operation in v1 (`coverage`, `compare`, `merge`, `list_check`,
`map`, `qa`) is driven against it.

This walkthrough is hermetic: it provisions a fresh SQLite store in
a temp directory, creates a workspace, attaches two doc ids, and
prints the resulting info bundle. No LLM, no Ollama, no network.

Run:

    python examples/v1/01_workspace.py

SPEC-REF: §6.7
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ctrldoc.ops.workspace import WorkspaceManager
from ctrldoc.store.sqlite import SQLiteStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "workspaces.db"
        store = SQLiteStore(db_path)
        try:
            workspaces = WorkspaceManager(store=store)

            # 1. Create a workspace. Name is unique; ID is content-derived
            # so a re-creation under the same name is replay-stable.
            ws = workspaces.create("due-diligence")
            print(f"created workspace {ws.name!r} with id {ws.id!r}")

            # 2. Attach two docs. Membership is order-preserving so the
            # cross-doc-edge enumeration walks pairs deterministically.
            workspaces.add("due-diligence", "company-spec")
            workspaces.add("due-diligence", "security-policy")

            # 3. Read back the aggregate view.
            info = workspaces.info("due-diligence")
            print(
                json.dumps(
                    {
                        "workspace_id": info.workspace.id,
                        "name": info.workspace.name,
                        "doc_ids": list(info.workspace.doc_ids),
                        "doc_count": info.doc_count,
                        "shared_concept_count": info.concept_count,
                    },
                    indent=2,
                )
            )

            # 4. The list surface is the registry of every workspace.
            registry = workspaces.list()
            print(f"\n{len(registry)} workspace(s) total")
            for w in registry:
                print(f"  - {w.name} ({len(w.doc_ids)} doc(s))")
        finally:
            store.close()


if __name__ == "__main__":
    main()
