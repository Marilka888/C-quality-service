"""
Polyakov-regression: RetrievedCandidate.unit_type schema gap.

The selector code (`adaptive_candidate_selector.py:102`) reads
`c.unit_type` to prefer atomic PARAGRAPH/LIST_ITEM evidence over
aggregate SECTION_WINDOW units. The retriever
(`retrieve_candidates.py:553`) writes `unit_type=unit.unit_type` when
constructing the candidate. But the field declaration was missing from
`RetrievedCandidate` in c_quality_models.py — pydantic refused the
attribute, the selector hit AttributeError on every parallel
requirement worker, and the run_coverage_analysis exception catch
silently dropped 28 of 31 Polyakov ТЗ requirements.

This test pins the contract: RetrievedCandidate must accept and
expose `unit_type` so the selector path doesn't blow up at runtime.
"""
from __future__ import annotations

from app.domain.c_quality_enums import CoverageUnitType
from app.domain.c_quality_models import RetrievedCandidate


def test_retrieved_candidate_default_unit_type_polyakov() -> None:
    # Default constructor must produce a candidate with a usable
    # unit_type (the selector's `c.unit_type != SECTION_WINDOW` filter
    # must not raise AttributeError).
    c = RetrievedCandidate(
        req_id="r1", unit_id="u1", target_document_id="doc",
    )
    assert hasattr(c, "unit_type")
    assert c.unit_type == CoverageUnitType.PARAGRAPH


def test_retrieved_candidate_explicit_unit_type_polyakov() -> None:
    # Explicit unit_type passes through and is comparable to the enum
    # in selector predicates.
    c = RetrievedCandidate(
        req_id="r1", unit_id="u1", target_document_id="doc",
        unit_type=CoverageUnitType.SECTION_WINDOW,
    )
    assert c.unit_type == CoverageUnitType.SECTION_WINDOW
    assert c.unit_type != CoverageUnitType.PARAGRAPH


def test_retrieved_candidate_unit_type_is_filterable_polyakov() -> None:
    # The selector pattern: split candidates into primary
    # (non-SECTION_WINDOW) and window-only. Without the field this
    # comprehension threw AttributeError on every Polyakov requirement.
    candidates = [
        RetrievedCandidate(req_id="r", unit_id="u1", target_document_id="d",
                           unit_type=CoverageUnitType.PARAGRAPH),
        RetrievedCandidate(req_id="r", unit_id="u2", target_document_id="d",
                           unit_type=CoverageUnitType.SECTION_WINDOW),
        RetrievedCandidate(req_id="r", unit_id="u3", target_document_id="d",
                           unit_type=CoverageUnitType.LIST_ITEM),
    ]
    primary = [c for c in candidates if c.unit_type != CoverageUnitType.SECTION_WINDOW]
    windows = [c for c in candidates if c.unit_type == CoverageUnitType.SECTION_WINDOW]
    assert len(primary) == 2
    assert len(windows) == 1
