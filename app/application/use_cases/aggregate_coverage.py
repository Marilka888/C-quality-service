"""
Stage 7: aggregate per (requirement, target_document) judgments into a
RequirementCoverageResult.

Priority: CONFLICT > COVERED > PARTIAL > MISSING
"""
from __future__ import annotations

from typing import Dict, List

from app.domain.c_quality_enums import CoverageStatus, LLMLabel
from app.domain.c_quality_models import (
    CoverageUnit,
    EvidenceItem,
    PairJudgment,
    RequirementCoverageResult,
    RequirementUnit,
    RetrievedCandidate,
)


def _label_to_status(label: LLMLabel) -> CoverageStatus:
    return {
        LLMLabel.COVERED: CoverageStatus.COVERED,
        LLMLabel.PARTIAL: CoverageStatus.PARTIAL,
        LLMLabel.CONFLICT: CoverageStatus.CONFLICT,
        LLMLabel.IRRELEVANT: CoverageStatus.MISSING,
    }[label]


_STATUS_RANK = {
    CoverageStatus.CONFLICT: 4,
    CoverageStatus.COVERED: 3,
    CoverageStatus.PARTIAL: 2,
    CoverageStatus.MISSING: 1,
}


class CoverageAggregator:
    def aggregate(
        self,
        requirement: RequirementUnit,
        judgments: List[PairJudgment],
        candidates_by_unit_id: Dict[str, RetrievedCandidate],
        units_by_id: Dict[str, CoverageUnit],
        target_document_id: str,
        target_doc_role: str,
    ) -> RequirementCoverageResult:
        if not judgments:
            return RequirementCoverageResult(
                req_id=requirement.req_id,
                source_document_id=requirement.source_document_id,
                target_document_id=target_document_id,
                target_doc_role=target_doc_role,
                status=CoverageStatus.MISSING,
            )

        best_status = CoverageStatus.MISSING
        evidence_items: List[EvidenceItem] = []
        uncovered: List[str] = []
        conflicts: List[str] = []

        for j in judgments:
            status = _label_to_status(j.rule_adjusted_label)
            if _STATUS_RANK[status] > _STATUS_RANK[best_status]:
                best_status = status

            unit = units_by_id.get(j.unit_id)
            candidate = candidates_by_unit_id.get(j.unit_id)
            if unit is not None:
                evidence_items.append(
                    EvidenceItem(
                        unit_id=j.unit_id,
                        fragment_id=unit.fragment_id,
                        section_id=unit.section_id,
                        text=unit.text[:300],
                        retrieval_score=candidate.retrieval_score if candidate else 0.0,
                        judgment=j,
                    )
                )
            uncovered.extend(j.missing_aspects)
            conflicts.extend(j.conflict_aspects)

        # TODO: future — composite partial coverage across multiple fragments

        return RequirementCoverageResult(
            req_id=requirement.req_id,
            source_document_id=requirement.source_document_id,
            target_document_id=target_document_id,
            target_doc_role=target_doc_role,
            status=best_status,
            evidence=evidence_items,
            uncovered_aspects=list(dict.fromkeys(uncovered)),
            conflict_details=list(dict.fromkeys(conflicts)),
        )
