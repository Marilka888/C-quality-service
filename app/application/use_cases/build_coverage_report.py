"""
Stage 8: assemble the final CoverageAnalysisResult from per-requirement results.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from app.domain.c_quality_enums import Applicability, CoverageStatus
from app.domain.c_quality_models import (
    CoverageAnalysisResult,
    CoverageSummary,
    DocumentCoverageReport,
    PairJudgment,
    RequirementCoverageResult,
)


def _tally(results: List[RequirementCoverageResult]) -> CoverageSummary:
    s = CoverageSummary()
    for r in results:
        # NOT_APPLICABLE and OUT_OF_SCOPE rows are excluded from
        # total_requirements and all status buckets so coverage_rate
        # reflects only requirements that were actually checked. They
        # are tracked in not_applicable for informational display.
        if r.applicability != Applicability.APPLICABLE:
            s.not_applicable += 1
            continue
        s.total_requirements += 1
        if r.status == CoverageStatus.COVERED:
            s.covered += 1
        elif r.status == CoverageStatus.PARTIAL:
            s.partial += 1
        elif r.status == CoverageStatus.CONFLICT:
            s.conflict += 1
        else:
            s.missing += 1
    return s


class CoverageReportBuilder:
    def build(
        self,
        job_id: str,
        package_id: str,
        source_document_id: str,
        requirement_results: List[RequirementCoverageResult],
        pair_judgments: Optional[List[PairJudgment]] = None,
        warnings: Optional[List[str]] = None,
    ) -> CoverageAnalysisResult:
        target_doc_ids = list(
            dict.fromkeys(r.target_document_id for r in requirement_results)
        )

        # Per-document tallies
        per_doc: Dict[str, DocumentCoverageReport] = {}
        for r in requirement_results:
            if r.target_document_id not in per_doc:
                per_doc[r.target_document_id] = DocumentCoverageReport(
                    target_document_id=r.target_document_id,
                    target_doc_role=r.target_doc_role,
                )
            report = per_doc[r.target_document_id]
            # NOT_APPLICABLE and OUT_OF_SCOPE rows are excluded from
            # total_requirements and status buckets — same logic as _tally.
            if r.applicability != Applicability.APPLICABLE:
                report.not_applicable += 1
                continue
            report.total_requirements += 1
            if r.status == CoverageStatus.COVERED:
                report.covered += 1
            elif r.status == CoverageStatus.PARTIAL:
                report.partial += 1
            elif r.status == CoverageStatus.CONFLICT:
                report.conflict += 1
            else:
                report.missing += 1

        return CoverageAnalysisResult(
            job_id=job_id,
            package_id=package_id,
            source_document_id=source_document_id,
            target_document_ids=target_doc_ids,
            summary=_tally(requirement_results),
            document_reports=list(per_doc.values()),
            requirement_results=requirement_results,
            pair_judgments=pair_judgments,
            warnings=warnings or [],
        )
