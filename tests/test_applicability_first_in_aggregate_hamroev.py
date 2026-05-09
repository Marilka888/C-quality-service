"""
P0 #9 (Хамроев) — applicability_for() must be the FIRST decision in
CoverageAggregator.aggregate(). NOT_APPLICABLE / OUT_OF_SCOPE rows must
NEVER be overwritten by MISSING_NO_EVIDENCE just because the upstream
pipeline produced an empty shortlist.

Risk this test pins:
  * Future refactor reorders aggregate() and the empty-shortlist
    branch (Branch B) starts firing before applicability_for() runs.
    DELIVERY_REQUIREMENT in PMI would then be reported as
    MISSING_NO_EVIDENCE and inflate the C-score's denominator with a
    bogus "missing" — exactly what Хамроев flagged on the demo run
    (USB-носитель / маркировка / комплект поставки sections).

Contract pinned: regardless of judgments / candidates being empty,
DELIVERY_REQUIREMENT in any target role and ARCHITECTURE_IMPLEMENTATION
in PMI must come back with applicability=OUT_OF_SCOPE/NOT_APPLICABLE
and the matching status_subcode.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.aggregate_coverage import (
    CoverageAggregator,
    SUBCODE_NOT_APPLICABLE,
    SUBCODE_OUT_OF_SCOPE,
)
from app.domain.c_quality_enums import (
    Applicability,
    CoverageStatus,
    RequirementType,
)
from app.domain.c_quality_models import RequirementUnit


def _req(req_type: RequirementType) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="любое требование, не важно",
        normalized_text="любое требование, не важно",
        requirement_type=req_type,
    )


@pytest.mark.parametrize("req_type, target_role", [
    # Hamroev: USB-носитель / маркировка / упаковка → DELIVERY in PMI/PZ.
    (RequirementType.DELIVERY_REQUIREMENT, "PMI"),
    (RequirementType.DELIVERY_REQUIREMENT, "PZ"),
    # Process / economic — same OUT_OF_SCOPE class.
    (RequirementType.PROCESS_REQUIREMENT, "PMI"),
])
def test_out_of_scope_wins_over_empty_shortlist_hamroev(
    req_type: RequirementType, target_role: str,
) -> None:
    aggregator = CoverageAggregator()
    result = aggregator.aggregate(
        requirement=_req(req_type),
        judgments=[],            # empty — Branch B used to win here.
        candidates_by_unit_id={},
        units_by_id={},
        target_document_id=f"doc-{target_role.lower()}",
        target_doc_role=target_role,
    )
    assert result.applicability == Applicability.OUT_OF_SCOPE, (
        f"empty-shortlist branch overwrote applicability for "
        f"{req_type.value} in {target_role}: got {result.applicability}"
    )
    assert result.status_subcode == SUBCODE_OUT_OF_SCOPE
    # Critical guarantee: row contributes to neither critical count nor
    # the C-axis grade. That's the whole point of OUT_OF_SCOPE.
    assert result.should_affect_critical is False
    assert result.should_affect_grade is False


def test_not_applicable_wins_over_empty_shortlist_pz_only_in_pmi_hamroev() -> None:
    # ARCHITECTURE_IMPLEMENTATION in PMI is NOT_APPLICABLE (it lives in
    # PZ; PMI may quote it but coverage isn't judged there). Empty
    # shortlist must not promote this to MISSING_NO_EVIDENCE.
    result = CoverageAggregator().aggregate(
        requirement=_req(RequirementType.ARCHITECTURE_IMPLEMENTATION),
        judgments=[],
        candidates_by_unit_id={},
        units_by_id={},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.applicability == Applicability.NOT_APPLICABLE
    assert result.status_subcode == SUBCODE_NOT_APPLICABLE
    assert result.should_affect_grade is False


def test_applicable_with_empty_shortlist_still_falls_to_missing_no_evidence() -> None:
    # Sanity: applicability=APPLICABLE + empty shortlist must STILL hit
    # the MISSING_NO_EVIDENCE branch — we don't want to accidentally
    # demote real APPLICABLE empties to OUT_OF_SCOPE.
    result = CoverageAggregator().aggregate(
        requirement=_req(RequirementType.FUNCTIONAL),
        judgments=[],
        candidates_by_unit_id={},
        units_by_id={},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.applicability == Applicability.APPLICABLE
    # Subcode for empty shortlist when type is REQUIRED.
    assert result.status_subcode in {
        "MISSING_NO_EVIDENCE", "OPTIONAL_NOT_FOUND",
    }
    assert result.status == CoverageStatus.MISSING
