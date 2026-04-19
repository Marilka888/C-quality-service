from __future__ import annotations

from typing import Dict, List

from app.core.config import ScoringConfig
from app.domain.entities import OrphanTestFinding, RequirementFinding, TestCase, TraceLink
from app.domain.enums import LinkStatus


class OrphanTestDetector:
    def __init__(self, scoring_config: ScoringConfig):
        self._scoring_config = scoring_config

    def detect(
        self,
        test_cases: List[TestCase],
        requirement_findings: List[RequirementFinding],
    ) -> List[OrphanTestFinding]:
        supported_statuses = {
            LinkStatus.ADEQUATE,
            LinkStatus.PARTIAL,
            LinkStatus.INADEQUATE,
            LinkStatus.CONFLICT,
        }
        support_threshold = self._scoring_config.thresholds.orphan_support_score_threshold
        supported_test_ids: set[str] = set()
        evidence_by_test_id: Dict[str, List[TraceLink]] = {}

        for finding in requirement_findings:
            for link in finding.evaluated_links:
                evidence_by_test_id.setdefault(link.test_case_id or "", []).append(link)
                if (
                    link.test_case_id
                    and link.link_status in supported_statuses
                    and link.retrieval_score >= support_threshold
                ):
                    supported_test_ids.add(link.test_case_id)

        orphans: List[OrphanTestFinding] = []
        for test_case in test_cases:
            if test_case.id in supported_test_ids:
                continue
            orphans.append(
                OrphanTestFinding(
                    test_id=test_case.id,
                    explanation="Test case was not selected as a meaningful requirement candidate above the orphan support threshold.",
                    similarity_evidence=evidence_by_test_id.get(test_case.id, []),
                )
            )
        return orphans
