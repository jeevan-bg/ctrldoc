"""Contract tests for the tree-sitter Python code parser.

`CodeParser` produces a `ParsedSection` for every top-level
function/class and every method nested inside a class. Section
titles are the identifier names, ids include the lexical path, and
parent_ids preserve class membership so the chunker can keep methods
attached to their owning class.

SPEC-REF: §4.1 (ingest step 1, code parsing)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.ingest.code import CodeParser
from ctrldoc.ingest.parser import Parser


def _write(tmp_path: Path, source: str, *, name: str = "module.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_satisfies_protocol() -> None:
    assert isinstance(CodeParser(), Parser)


def test_top_level_function(tmp_path: Path) -> None:
    src = _write(tmp_path, "def foo(x):\n    return x + 1\n")
    sections = CodeParser().parse(src)
    assert len(sections) == 1
    assert sections[0].title == "foo"
    assert sections[0].parent_id is None
    assert "return x + 1" in sections[0].text


def test_multiple_top_level_functions(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "def alpha():\n    pass\n\ndef beta():\n    pass\n\ndef gamma():\n    pass\n",
    )
    sections = CodeParser().parse(src)
    titles = [s.title for s in sections]
    assert titles == ["alpha", "beta", "gamma"]
    for s in sections:
        assert s.parent_id is None


def test_class_with_methods_nests_methods(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "class Foo:\n    def m1(self):\n        pass\n    def m2(self):\n        pass\n",
    )
    sections = CodeParser().parse(src)
    by_title = {s.title: s for s in sections}
    assert set(by_title) == {"Foo", "m1", "m2"}
    assert by_title["Foo"].parent_id is None
    assert by_title["m1"].parent_id == by_title["Foo"].id
    assert by_title["m2"].parent_id == by_title["Foo"].id


def test_mixed_top_level_and_class(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "def free():\n    pass\n\nclass Bar:\n    def method(self):\n        return 1\n",
    )
    sections = CodeParser().parse(src)
    titles = [s.title for s in sections]
    # Order: depth-first per top-level node.
    assert titles == ["free", "Bar", "method"]
    by_title = {s.title: s for s in sections}
    assert by_title["method"].parent_id == by_title["Bar"].id


def test_section_text_contains_signature_and_body(tmp_path: Path) -> None:
    src = _write(tmp_path, "def adder(a, b):\n    return a + b\n")
    section = CodeParser().parse(src)[0]
    assert section.text.startswith("def adder")
    assert "return a + b" in section.text


def test_section_char_range_matches_source(tmp_path: Path) -> None:
    raw = "def x():\n    return 1\n"
    src = _write(tmp_path, raw)
    section = CodeParser().parse(src)[0]
    assert raw[section.char_start : section.char_end].startswith("def x()")


def test_ignores_imports_and_module_level_assignments(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "import os\nX = 1\n\ndef thing():\n    return X\n",
    )
    sections = CodeParser().parse(src)
    titles = [s.title for s in sections]
    assert titles == ["thing"]


def test_empty_file_returns_no_sections(tmp_path: Path) -> None:
    src = _write(tmp_path, "")
    assert CodeParser().parse(src) == []


def test_module_with_only_comments_returns_no_sections(tmp_path: Path) -> None:
    src = _write(tmp_path, "# nothing here\n# really\n")
    assert CodeParser().parse(src) == []


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CodeParser().parse(tmp_path / "missing.py")


def test_str_path_accepted(tmp_path: Path) -> None:
    src = _write(tmp_path, "def f():\n    pass\n")
    sections = CodeParser().parse(str(src))
    assert len(sections) == 1


def test_deterministic_across_calls(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "class A:\n    def x(self):\n        pass\n\ndef y():\n    pass\n",
    )
    first = [(s.id, s.title) for s in CodeParser().parse(src)]
    second = [(s.id, s.title) for s in CodeParser().parse(src)]
    assert first == second


def test_duplicate_names_get_unique_ids(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "def dup():\n    pass\n\ndef dup():\n    pass\n",
    )
    sections = CodeParser().parse(src)
    ids = [s.id for s in sections]
    assert len(ids) == len(set(ids))
    assert all(s.title == "dup" for s in sections)


def test_nested_function_inside_function_is_skipped(tmp_path: Path) -> None:
    """Sections are top-level + class methods only; locally-defined functions stay in their parent's body."""
    src = _write(
        tmp_path,
        "def outer():\n    def inner():\n        pass\n    return inner\n",
    )
    sections = CodeParser().parse(src)
    titles = [s.title for s in sections]
    assert titles == ["outer"]
    assert "def inner()" in sections[0].text
