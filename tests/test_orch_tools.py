"""L4 tool surface — Pydantic schemas + forced-tool-call dispatcher.

SPEC §6.10 defines a fixed 13-tool surface that the L4 orchestrator
(and the §11 MCP server) exposes to a top-level LLM. The "forced tool
calls only" constraint means three things at the dispatcher layer:

1. **Closed alphabet.** Calling a name not in the registered surface
   raises `UnknownToolError` — there is no free-form fallback path the
   model can sneak into.
2. **Validated inputs.** Every invocation pre-validates the raw input
   dict against the tool's `input_model`. A malformed payload raises
   `ToolValidationError` before the handler runs.
3. **Validated outputs.** Handlers may return either a `BaseModel`
   instance of the registered `output_model` or a raw dict; the
   dispatcher re-validates either way so downstream callers always see
   a typed object.

The test below pins:

* The 13 tool names from §6.10 are registered out of the box.
* Each tool exposes a Pydantic input_model / output_model pair.
* The dispatcher rejects unknown tools and malformed inputs.
* The dispatcher validates handler outputs and returns a typed model.
* Handlers for engines that have not yet shipped (Phase 18+) raise
  `ToolNotImplementedError` cleanly — they do NOT silently no-op or
  invent fake answers, which would violate §13 non-negotiable 3
  (every claim cited or refused).
* Tool schemas carry a `schema_version` field so future shape changes
  break loudly (§13 non-negotiable 14 — MCP tool schemas versioned).

SPEC-REF: §6.10 (tool-using orchestrator, forced tool calls)
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from ctrldoc.orch.tools import (
    TOOL_SURFACE,
    TOOL_SURFACE_VERSION,
    ToolDispatcher,
    ToolNotImplementedError,
    ToolSpec,
    ToolValidationError,
    UnknownToolError,
    default_dispatcher,
)

# ---------------------------------------------------------------------------
# §6.10 enumerates exactly these 13 tools. Any drift is a spec drift.
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = frozenset(
    {
        "lookup_concept",
        "get_claim",
        "traverse",
        "entails",
        "subsumes",
        "optimal_transport",
        "coverage",
        "compare",
        "merge",
        "list_check",
        "map",
        "qa",
        "calibration",
    }
)


# ---------------------------------------------------------------------------
# Surface coverage
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_surface_matches_spec_exactly() -> None:
    """The registered tool names equal the §6.10 list — no extras, no gaps."""
    assert set(TOOL_SURFACE.keys()) == EXPECTED_TOOL_NAMES


@pytest.mark.family_referential_integrity
def test_every_tool_carries_pydantic_models() -> None:
    """Each registered tool exposes `input_model` and `output_model` Pydantic types."""
    for name, spec in TOOL_SURFACE.items():
        assert isinstance(spec, ToolSpec), f"{name} is not a ToolSpec"
        assert issubclass(spec.input_model, BaseModel), f"{name}.input_model is not Pydantic"
        assert issubclass(spec.output_model, BaseModel), f"{name}.output_model is not Pydantic"
        assert spec.description.strip(), f"{name} missing description"


@pytest.mark.family_referential_integrity
def test_surface_version_is_pinned() -> None:
    """`TOOL_SURFACE_VERSION` is a non-empty semver string (§13 non-negotiable 14)."""
    assert isinstance(TOOL_SURFACE_VERSION, str)
    parts = TOOL_SURFACE_VERSION.split(".")
    assert len(parts) == 3 and all(
        p.isdigit() for p in parts
    ), f"expected semver MAJOR.MINOR.PATCH; got {TOOL_SURFACE_VERSION!r}"


# ---------------------------------------------------------------------------
# Dispatcher behaviour
# ---------------------------------------------------------------------------


@pytest.mark.family_referential_integrity
def test_unknown_tool_rejected() -> None:
    """Forced tool calls means the dispatcher refuses out-of-surface names."""
    dispatcher = default_dispatcher()
    with pytest.raises(UnknownToolError):
        dispatcher.dispatch(tool_name="not_a_tool", raw_input={})


@pytest.mark.family_referential_integrity
def test_malformed_input_rejected() -> None:
    """Bad payloads raise `ToolValidationError` before any handler runs."""
    dispatcher = default_dispatcher()
    # `lookup_concept` requires a `name` field; passing `{}` is malformed.
    with pytest.raises(ToolValidationError):
        dispatcher.dispatch(tool_name="lookup_concept", raw_input={})


@pytest.mark.family_referential_integrity
def test_handler_can_return_dict_and_dispatcher_validates() -> None:
    """Handlers may return raw dicts; the dispatcher re-validates them."""
    dispatcher = default_dispatcher()

    def lookup_handler(payload: BaseModel) -> dict[str, str | None]:
        # Return a raw dict — dispatcher must coerce + validate.
        return {"concept_id": "concept-test-001"}

    dispatcher.register_handler("lookup_concept", lookup_handler)
    result = dispatcher.dispatch(
        tool_name="lookup_concept",
        raw_input={"name": "Photosynthesis"},
    )
    spec = TOOL_SURFACE["lookup_concept"]
    assert isinstance(result, spec.output_model)
    assert result.concept_id == "concept-test-001"  # type: ignore[attr-defined]


@pytest.mark.family_referential_integrity
def test_handler_returning_wrong_shape_raises_validation_error() -> None:
    """Handler outputs are validated; wrong shapes raise `ToolValidationError`."""
    dispatcher = default_dispatcher()

    def bad_handler(payload: BaseModel) -> dict[str, str]:
        return {"not_a_real_field": "oops"}

    dispatcher.register_handler("lookup_concept", bad_handler)
    with pytest.raises(ToolValidationError):
        dispatcher.dispatch(
            tool_name="lookup_concept",
            raw_input={"name": "Anything"},
        )


@pytest.mark.family_referential_integrity
def test_unregistered_handler_raises_not_implemented() -> None:
    """Tools without a wired handler raise `ToolNotImplementedError` — never fake answers."""
    dispatcher = default_dispatcher()
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(
            tool_name="coverage",
            raw_input={
                "workspace_id": "ws-abc",
                "target_doc_id": "doc-1",
                "source_doc_id": "doc-2",
            },
        )


@pytest.mark.family_referential_integrity
def test_handler_can_return_validated_model_directly() -> None:
    """If the handler returns an `output_model` instance, the dispatcher passes it through."""
    dispatcher = default_dispatcher()
    spec = TOOL_SURFACE["lookup_concept"]

    def lookup_handler(payload: BaseModel) -> BaseModel:
        return spec.output_model(concept_id=None)

    dispatcher.register_handler("lookup_concept", lookup_handler)
    result = dispatcher.dispatch(
        tool_name="lookup_concept",
        raw_input={"name": "Unknown"},
    )
    assert isinstance(result, spec.output_model)
    assert result.concept_id is None  # type: ignore[attr-defined]


@pytest.mark.family_referential_integrity
def test_register_handler_for_unknown_tool_rejected() -> None:
    """Even at wire-up time, only known tool names accept handlers."""
    dispatcher = default_dispatcher()
    with pytest.raises(UnknownToolError):
        dispatcher.register_handler("nope", lambda payload: {})


# ---------------------------------------------------------------------------
# Per-tool schema sanity — every input/output round-trips through validation.
# ---------------------------------------------------------------------------


@pytest.mark.family_determinism
def test_traverse_input_validates_edge_type() -> None:
    """`traverse` accepts only registered edge types from `TypedEdgeTypeLiteral`."""
    spec = TOOL_SURFACE["traverse"]
    # Good case: every typed-edge literal works.
    spec.input_model(
        node_id="claim-x",
        edge_type="entails",
        direction="forward",
        hops=1,
    )
    # Bad case: bogus edge type rejected by Pydantic.
    with pytest.raises(ValidationError):
        spec.input_model(
            node_id="claim-x",
            edge_type="not-an-edge",
            direction="forward",
            hops=1,
        )


@pytest.mark.family_determinism
def test_entails_output_carries_verdict_and_confidence() -> None:
    """`entails` output shape matches the §6.10 `{verdict, confidence}` spec."""
    spec = TOOL_SURFACE["entails"]
    obj = spec.output_model(verdict="entailment", confidence=0.91)
    assert obj.verdict == "entailment"  # type: ignore[attr-defined]
    assert obj.confidence == pytest.approx(0.91)  # type: ignore[attr-defined]
    # Verdict label is closed: invalid labels rejected.
    with pytest.raises(ValidationError):
        spec.output_model(verdict="maybe", confidence=0.5)
    # Confidence is unit-interval clamped.
    with pytest.raises(ValidationError):
        spec.output_model(verdict="entailment", confidence=1.5)


@pytest.mark.family_determinism
def test_calibration_output_has_per_backend_map() -> None:
    """`calibration()` returns `{ECE_per_backend, sample_sizes}` keyed by backend."""
    spec = TOOL_SURFACE["calibration"]
    obj = spec.output_model(
        ece_per_backend={"nli": 0.04, "llm": 0.06},
        sample_sizes={"nli": 200, "llm": 200},
    )
    assert obj.ece_per_backend["nli"] == pytest.approx(0.04)  # type: ignore[attr-defined]
    assert obj.sample_sizes["llm"] == 200  # type: ignore[attr-defined]


@pytest.mark.family_determinism
def test_dispatcher_is_isolated_per_instance() -> None:
    """Registering a handler on one dispatcher does not leak to a fresh one."""
    a = default_dispatcher()
    b = default_dispatcher()
    a.register_handler("lookup_concept", lambda payload: {"concept_id": "x"})
    with pytest.raises(ToolNotImplementedError):
        b.dispatch(tool_name="lookup_concept", raw_input={"name": "y"})


@pytest.mark.family_determinism
def test_dispatcher_instance_constructible_directly() -> None:
    """`ToolDispatcher()` builds an empty-handler dispatcher over the default surface."""
    dispatcher = ToolDispatcher()
    assert set(dispatcher.tool_names()) == EXPECTED_TOOL_NAMES
    with pytest.raises(ToolNotImplementedError):
        dispatcher.dispatch(tool_name="qa", raw_input={"target": "doc-1", "query": "what?"})
