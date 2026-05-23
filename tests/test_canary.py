"""Continuous-canary primitive — unit + integration tests.

Tests cover the comparison machinery (`signature_hash_of`,
`compute_drift`, `check_canary`, `save_baseline` / `load_baseline`)
and lay down one real baseline: a sorted chunk-id list from the
ingest pipeline running over the synthetic gold doc. CI fans the
canary out across all six playbooks once each plays nicely with a
deterministic substrate; this module ships the substrate plus the
ingest baseline that's already deterministic today.

SPEC-REF: §8.6 (continuous canary)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctrldoc.canary import (
    DEFAULT_DRIFT_THRESHOLD,
    CanaryBaseline,
    CanaryReport,
    check_canary,
    compute_drift,
    load_baseline,
    save_baseline,
    signature_hash_of,
)
from ctrldoc.ingest.coref import IdentityCorefResolver
from ctrldoc.ingest.embedder import HashEmbedder
from ctrldoc.ingest.ner import StubNERTagger
from ctrldoc.ingest.parser import MarkdownParser
from ctrldoc.ingest.pipeline import ingest_document
from ctrldoc.ingest.summarizer import HeuristicSummarizer
from ctrldoc.store.bm25 import TantivyBM25Index
from ctrldoc.store.memory import InMemoryStore
from ctrldoc.store.vectors import InMemoryVectorIndex

# --- signature_hash_of ---


def test_signature_hash_stable_across_input_order() -> None:
    """The hash sorts each value list before hashing, so two equal
    signatures emitted in different orders share a hash."""
    a = {"chunk_ids": ["c1", "c2", "c3"]}
    b = {"chunk_ids": ["c3", "c1", "c2"]}
    assert signature_hash_of(a) == signature_hash_of(b)


def test_signature_hash_distinguishes_distinct_signatures() -> None:
    a = {"chunk_ids": ["c1", "c2"]}
    b = {"chunk_ids": ["c1", "c2", "c3"]}
    assert signature_hash_of(a) != signature_hash_of(b)


def test_signature_hash_is_deterministic() -> None:
    sig = {"chunk_ids": ["c1", "c2"], "section_ids": ["s/1"]}
    assert signature_hash_of(sig) == signature_hash_of(sig)


def test_signature_hash_empty_signature_is_consistent() -> None:
    assert signature_hash_of({}) == signature_hash_of({})


# --- CanaryBaseline.from_signature ---


def test_from_signature_sorts_values_and_records_hash() -> None:
    baseline = CanaryBaseline.from_signature(
        doc_id="gold",
        playbook="ingest",
        signature={"chunk_ids": ["c3", "c1", "c2"]},
    )
    assert baseline.signature == {"chunk_ids": ["c1", "c2", "c3"]}
    assert baseline.signature_hash == signature_hash_of(baseline.signature)


# --- compute_drift ---


def test_compute_drift_no_change_returns_zero_fractions() -> None:
    sig = {"chunk_ids": ["c1", "c2", "c3"]}
    drifts = compute_drift(sig, sig)
    assert all(d.fraction == 0.0 for d in drifts)


def test_compute_drift_single_added_id() -> None:
    """Symmetric diff size 1 over union size 4 ⇒ 0.25."""
    baseline = {"chunk_ids": ["c1", "c2", "c3"]}
    current = {"chunk_ids": ["c1", "c2", "c3", "c4"]}
    drifts = compute_drift(baseline, current)
    assert len(drifts) == 1
    assert drifts[0].key == "chunk_ids"
    assert drifts[0].fraction == pytest.approx(0.25)
    assert drifts[0].symmetric_difference == 1
    assert drifts[0].baseline_size == 3
    assert drifts[0].current_size == 4


def test_compute_drift_completely_different_returns_one() -> None:
    baseline = {"chunk_ids": ["c1", "c2"]}
    current = {"chunk_ids": ["c3", "c4"]}
    drifts = compute_drift(baseline, current)
    assert drifts[0].fraction == pytest.approx(1.0)


def test_compute_drift_key_only_in_baseline_is_full_drift() -> None:
    baseline = {"chunk_ids": ["c1"], "section_ids": ["s/1"]}
    current = {"chunk_ids": ["c1"]}
    drifts = compute_drift(baseline, current)
    by_key = {d.key: d for d in drifts}
    assert by_key["chunk_ids"].fraction == pytest.approx(0.0)
    assert by_key["section_ids"].fraction == pytest.approx(1.0)
    assert by_key["section_ids"].current_size == 0


def test_compute_drift_key_only_in_current_is_full_drift() -> None:
    """Symmetric — a new key in current is a 1.0 drift on that key."""
    baseline = {"chunk_ids": ["c1"]}
    current = {"chunk_ids": ["c1"], "entity_ids": ["e/1"]}
    drifts = compute_drift(baseline, current)
    by_key = {d.key: d for d in drifts}
    assert by_key["entity_ids"].fraction == pytest.approx(1.0)


# --- check_canary ---


def _baseline(sig: dict[str, list[str]]) -> CanaryBaseline:
    return CanaryBaseline.from_signature(doc_id="gold", playbook="ingest", signature=sig)


def test_check_canary_hash_match_when_identical() -> None:
    sig = {"chunk_ids": ["c1", "c2"]}
    report = check_canary(_baseline(sig), sig)
    assert report.hash_match is True
    assert report.passed is True
    assert report.flagged_keys == []
    assert report.max_drift == pytest.approx(0.0)


def test_check_canary_drift_within_threshold_passes() -> None:
    """Default threshold is 0.10; a 1/12 ≈ 0.083 drift is fine."""
    baseline = _baseline({"chunk_ids": [f"c{i}" for i in range(12)]})
    current = {"chunk_ids": [f"c{i}" for i in range(12)] + ["c-new"]}
    report = check_canary(baseline, current)
    assert report.passed is True
    assert report.max_drift == pytest.approx(1 / 13, abs=1e-9)
    assert report.flagged_keys == []


def test_check_canary_drift_above_threshold_flags_key() -> None:
    """Default threshold is 0.10; ~25% drift flags."""
    baseline = _baseline({"chunk_ids": ["c1", "c2", "c3"]})
    current = {"chunk_ids": ["c1", "c2", "c3", "c4"]}
    report = check_canary(baseline, current)
    assert report.passed is False
    assert report.flagged_keys == ["chunk_ids"]


def test_check_canary_custom_threshold_can_be_strict() -> None:
    baseline = _baseline({"chunk_ids": [f"c{i}" for i in range(20)]})
    current = {"chunk_ids": [f"c{i}" for i in range(20)] + ["c-new"]}
    relaxed = check_canary(baseline, current, threshold=DEFAULT_DRIFT_THRESHOLD)
    strict = check_canary(baseline, current, threshold=0.01)
    assert relaxed.passed is True
    assert strict.passed is False


def test_check_canary_multi_key_signature() -> None:
    baseline = _baseline(
        {
            "chunk_ids": ["c1", "c2", "c3", "c4", "c5"],
            "section_ids": ["s/1", "s/2"],
        }
    )
    # One chunk renamed; sections unchanged. chunk_ids drift = 2/6 ≈ 0.333
    current = {
        "chunk_ids": ["c1", "c2", "c3", "c4", "c5-renamed"],
        "section_ids": ["s/1", "s/2"],
    }
    report = check_canary(baseline, current)
    assert report.passed is False
    assert "chunk_ids" in report.flagged_keys
    assert "section_ids" not in report.flagged_keys


def test_check_canary_report_carries_threshold_used() -> None:
    sig = {"chunk_ids": ["c1"]}
    report = check_canary(_baseline(sig), sig, threshold=0.05)
    assert report.threshold == pytest.approx(0.05)


# --- baseline round-trip on disk ---


def test_save_load_baseline_round_trip(tmp_path: Path) -> None:
    baseline = CanaryBaseline.from_signature(
        doc_id="aurora",
        playbook="ingest",
        signature={"chunk_ids": ["c1", "c2"], "section_ids": ["s/1"]},
    )
    path = tmp_path / "canary" / "aurora__ingest.json"
    save_baseline(path, baseline)
    assert path.is_file()
    loaded = load_baseline(path)
    assert loaded == baseline


def test_save_baseline_creates_parent_directory(tmp_path: Path) -> None:
    """`tests/canary/baselines/` may not exist yet on a fresh checkout."""
    baseline = CanaryBaseline.from_signature(
        doc_id="x",
        playbook="y",
        signature={"a": ["1"]},
    )
    nested = tmp_path / "deeply" / "nested" / "x__y.json"
    save_baseline(nested, baseline)
    assert nested.is_file()


def test_load_baseline_validates_schema(tmp_path: Path) -> None:
    """A baseline file with extra fields is rejected by `extra='forbid'`."""
    from pydantic import ValidationError

    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"doc_id":"x","playbook":"y","signature":{},"signature_hash":"abc","ghost":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_baseline(bad)


# --- end-to-end: ingest baseline on the synthetic doc ---


@pytest.fixture
def ingest_signature(tmp_path: Path, synthetic_doc_path: Path) -> dict[str, list[str]]:
    """Run the deterministic ingest pipeline once and return its signature."""
    store = InMemoryStore()
    vector_index = InMemoryVectorIndex(dimension=32)
    bm25_index = TantivyBM25Index(path=tmp_path / "bm25")
    ingest_document(
        source=synthetic_doc_path,
        parser=MarkdownParser(),
        coref=IdentityCorefResolver(),
        ner=StubNERTagger({}),
        ner_labels=["person", "system"],
        embedder=HashEmbedder(dimension=32),
        summarizer=HeuristicSummarizer(),
        store=store,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )
    return {
        "chunk_ids": sorted(c.id for c in store.iter_chunks()),
        "section_ids": sorted(s.id for s in store.iter_sections()),
        "entity_ids": sorted(e.id for e in store.iter_entities()),
    }


def test_ingest_signature_is_non_trivial(ingest_signature: dict[str, list[str]]) -> None:
    """Sanity check before the canary asserts equality across re-runs:
    the synthetic doc must produce a non-empty signature, otherwise the
    canary would be vacuously passing on emptiness."""
    assert ingest_signature["chunk_ids"], "ingest yielded zero chunks"
    assert ingest_signature["section_ids"], "ingest yielded zero sections"


def test_ingest_signature_matches_committed_baseline(
    ingest_signature: dict[str, list[str]],
    repo_root: Path,
) -> None:
    """The committed baseline at `tests/canary/baselines/aurora__ingest.json`
    must agree with a fresh ingest run. A drift > 10% fails CI; this
    test asserts byte-equality (drift exactly 0%)."""
    baseline_path = repo_root / "tests" / "canary" / "baselines" / "aurora__ingest.json"
    baseline = load_baseline(baseline_path)
    report = check_canary(baseline, ingest_signature)
    assert report.passed is True, (
        f"ingest canary drifted on keys {report.flagged_keys}; "
        f"max drift {report.max_drift:.3f}. "
        f"If this drift is intentional, re-pin the baseline."
    )
    assert report.hash_match is True, (
        "ingest canary signature hash mismatched but drift was within threshold; "
        "the baseline file should be re-pinned."
    )


def test_ingest_canary_report_shape(
    ingest_signature: dict[str, list[str]],
    repo_root: Path,
) -> None:
    """The `CanaryReport` exposes everything a CI reporter needs."""
    baseline_path = repo_root / "tests" / "canary" / "baselines" / "aurora__ingest.json"
    baseline = load_baseline(baseline_path)
    report = check_canary(baseline, ingest_signature)
    assert isinstance(report, CanaryReport)
    assert report.doc_id == baseline.doc_id
    assert report.playbook == baseline.playbook
    assert report.threshold == pytest.approx(DEFAULT_DRIFT_THRESHOLD)
    # Every signature key should be represented in the drift table.
    drift_keys = {drift.key for drift in report.drifts}
    assert drift_keys == set(baseline.signature) | set(ingest_signature)
