"""Contract tests for the coreference-resolver protocol and identity reference.

Coref replaces pronouns and other anaphoric references with canonical
mentions ("It is helpful." → "Claude is helpful."). Downstream NER and
retrieval slices need the protocol contract to compile against. The
identity reference is a passthrough that the rest of the pipeline can
use until a working fastcoref backend lands.

SPEC-REF: §4.1 (ingest step 2 — coref)
"""

from __future__ import annotations

from ctrldoc.ingest.coref import (
    CorefResolver,
    IdentityCorefResolver,
    resolve_sections,
)
from ctrldoc.ingest.parser import ParsedSection


def _section(text: str, *, section_id: str = "sec/a", char_start: int = 0) -> ParsedSection:
    return ParsedSection(
        id=section_id,
        parent_id=None,
        title="Title",
        text=text,
        char_start=char_start,
        char_end=char_start + len(text),
    )


def test_satisfies_protocol() -> None:
    assert isinstance(IdentityCorefResolver(), CorefResolver)


def test_identity_resolve_returns_input_unchanged() -> None:
    text = "Claude was created by Anthropic. It is a helpful assistant."
    assert IdentityCorefResolver().resolve(text) == text


def test_identity_resolve_empty_string() -> None:
    assert IdentityCorefResolver().resolve("") == ""


def test_identity_resolve_unicode_preserved() -> None:
    text = "数字 + emoji 🚀 + RTL: مرحبا"
    assert IdentityCorefResolver().resolve(text) == text


def test_resolve_sections_yields_same_count() -> None:
    parsed = [
        _section("First section text. Pronouns here.", section_id="sec/a"),
        _section("Second section. Body sentence.", section_id="sec/b"),
    ]
    resolved = resolve_sections(parsed, IdentityCorefResolver())
    assert len(resolved) == len(parsed)
    assert [s.id for s in resolved] == ["sec/a", "sec/b"]


def test_resolve_sections_preserves_structural_fields() -> None:
    parsed = [
        _section("body", section_id="sec/a", char_start=10),
    ]
    original = parsed[0]
    resolved = resolve_sections(parsed, IdentityCorefResolver())[0]
    assert resolved.id == original.id
    assert resolved.parent_id == original.parent_id
    assert resolved.title == original.title
    assert resolved.char_start == original.char_start
    assert resolved.char_end == original.char_end


def test_resolve_sections_text_passed_through_by_identity() -> None:
    parsed = [_section("Claude is here. It is helpful.")]
    resolved = resolve_sections(parsed, IdentityCorefResolver())
    assert resolved[0].text == parsed[0].text


def test_resolve_sections_uses_resolver_output_when_text_differs() -> None:
    class UppercaseResolver:
        """Stand-in resolver to prove `resolve_sections` actually calls the resolver."""

        def resolve(self, text: str) -> str:
            return text.upper()

    parsed = [_section("hello world")]
    resolved = resolve_sections(parsed, UppercaseResolver())
    assert resolved[0].text == "HELLO WORLD"


def test_identity_is_deterministic_across_calls() -> None:
    text = "Some text. Another sentence."
    resolver = IdentityCorefResolver()
    assert resolver.resolve(text) == resolver.resolve(text)


def test_resolve_sections_accepts_arbitrary_iterable() -> None:
    parsed = [_section("body", section_id=f"sec/{i}") for i in range(3)]
    resolved = list(resolve_sections(iter(parsed), IdentityCorefResolver()))
    assert {s.id for s in resolved} == {"sec/0", "sec/1", "sec/2"}
