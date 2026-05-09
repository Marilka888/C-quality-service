"""
P0 #10 (all 5 packages): row-level grounding_failed surfaces from
PairJudgment to RequirementCoverageResult so docback can render a
«цитаты не подтверждены» badge in the UI / PDF.

The aggregator already had `low_confidence` propagation (PR-C BUG-3).
This test pins the new row-level `grounding_failed` field — distinct
from low_confidence because:
  * low_confidence fires on weak retrieval too (no LLM dishonesty);
  * grounding_failed fires only when the LLM cited phrases that the
    substring-grounding gate rejected (LLM hallucinated quotes).
"""
from __future__ import annotations

from app.application.use_cases.aggregate_coverage import CoverageAggregator
from app.domain.c_quality_enums import (
    Applicability,
    CoverageStatus,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    CoverageUnit,
    EvidenceItem,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)


def _req() -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Программа должна работать.",
        normalized_text="программа должна работать.",
        requirement_type=RequirementType.FUNCTIONAL,
    )


def _unit(unit_id: str = "u1") -> CoverageUnit:
    return CoverageUnit(
        unit_id=unit_id,
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        text="программа работает.",
        normalized_text="программа работает.",
    )


def _candidate(unit_id: str = "u1", score: float = 0.5) -> RetrievedCandidate:
    return RetrievedCandidate(
        req_id="r1",
        unit_id=unit_id,
        target_document_id="doc-pmi",
        retrieval_score=score,
    )


def test_row_grounding_failed_set_when_any_judgment_failed_grounding_p10() -> None:
    # The aggregator must surface grounding_failed=True onto the row
    # whenever ANY judgment that participated in the verdict had its
    # citations rejected by the grounding gate.
    j_failed = PairJudgment(
        req_id="r1",
        unit_id="u1",
        target_document_id="doc-pmi",
        llm_label=LLMLabel.IRRELEVANT,
        rule_adjusted_label=LLMLabel.IRRELEVANT,
        llm_confidence=0.3,
        grounding_failed=True,
    )
    result = CoverageAggregator().aggregate(
        requirement=_req(),
        judgments=[j_failed],
        candidates_by_unit_id={"u1": _candidate()},
        units_by_id={"u1": _unit()},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.grounding_failed is True
    # status_subcode also propagates so the UI can render a finer badge.
    assert result.status_subcode is not None


def test_row_grounding_failed_false_when_no_judgment_failed_p10() -> None:
    # Sanity: when none of the judgments had grounding_failed, the
    # row's flag stays False even if low_confidence is set elsewhere.
    j_ok = PairJudgment(
        req_id="r1",
        unit_id="u1",
        target_document_id="doc-pmi",
        llm_label=LLMLabel.IRRELEVANT,
        rule_adjusted_label=LLMLabel.IRRELEVANT,
        llm_confidence=0.3,
        grounding_failed=False,
        low_confidence=True,  # weak retrieval, but grounding succeeded
    )
    result = CoverageAggregator().aggregate(
        requirement=_req(),
        judgments=[j_ok],
        candidates_by_unit_id={"u1": _candidate()},
        units_by_id={"u1": _unit()},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.grounding_failed is False
    assert result.low_confidence is True
