"""Tree-sitter Python code parser.

Reads a `.py` source file and emits one `ParsedSection` per top-level
function or class, plus one per method inside each class. Nested
functions inside other functions stay inline (their text remains in
the outer function's body) so the chunker doesn't over-split.

SPEC-REF: §4.1 (ingest step 1, code parsing)
"""

from __future__ import annotations

from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Node
from tree_sitter import Parser as TSParser

from ctrldoc.ingest.parser import ParsedSection

_PY_LANGUAGE = Language(tree_sitter_python.language())


class CodeParser:
    """Parse Python source into a function/class section tree."""

    def __init__(self) -> None:
        self._parser = TSParser(_PY_LANGUAGE)

    def parse(self, source: str | Path) -> list[ParsedSection]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        tree = self._parser.parse(raw)
        sections: list[ParsedSection] = []
        used_ids: set[str] = set()
        for child in tree.root_node.children:
            self._collect(child, raw, parent_id=None, sections=sections, used_ids=used_ids)
        return sections

    def _collect(
        self,
        node: Node,
        source: bytes,
        *,
        parent_id: str | None,
        sections: list[ParsedSection],
        used_ids: set[str],
    ) -> None:
        if node.type == "function_definition":
            name = _name_of(node, source) or "<anonymous>"
            section_id = _allocate_id(used_ids, name, parent_id)
            text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            sections.append(
                ParsedSection(
                    id=section_id,
                    parent_id=parent_id,
                    title=name,
                    text=text,
                    char_start=node.start_byte,
                    char_end=node.end_byte,
                )
            )
            # Nested functions stay inside the parent body — don't recurse.
            return
        if node.type == "class_definition":
            name = _name_of(node, source) or "<anonymous>"
            section_id = _allocate_id(used_ids, name, parent_id)
            text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            sections.append(
                ParsedSection(
                    id=section_id,
                    parent_id=parent_id,
                    title=name,
                    text=text,
                    char_start=node.start_byte,
                    char_end=node.end_byte,
                )
            )
            # Recurse into the class body to capture methods.
            body = next((c for c in node.children if c.type == "block"), None)
            if body is not None:
                for child in body.children:
                    self._collect(
                        child,
                        source,
                        parent_id=section_id,
                        sections=sections,
                        used_ids=used_ids,
                    )
            return
        # Anything else (imports, assignments, comments) is not a section.


def _name_of(node: Node, source: bytes) -> str | None:
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return None


def _allocate_id(used: set[str], name: str, parent_id: str | None) -> str:
    base = f"code/{name}" if parent_id is None else f"{parent_id}/{name}"
    candidate = base
    n = 2
    while candidate in used:
        candidate = f"{base}-{n}"
        n += 1
    used.add(candidate)
    return candidate


__all__ = ["CodeParser"]
