"""
Stage 5: run the LLM judge over the top-K shortlist for one requirement.
"""
from __future__ import annotations

from typing import Dict, List

from app.core.logging import get_logger
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.llm.coverage_judge import CoverageJudge

logger = get_logger(__name__)


class PairJudgeService:
    def __init__(self, judge: CoverageJudge) -> None:
        self._judge = judge

    def judge_shortlist(
        self,
        requirement: RequirementUnit,
        shortlist: List[RetrievedCandidate],
        units_by_id: Dict[str, CoverageUnit],
    ) -> List[PairJudgment]:
        judgments: List[PairJudgment] = []
        for candidate in shortlist:
            unit = units_by_id.get(candidate.unit_id)
            if unit is None:
                logger.warning("unit_id=%s not found in index; skipping", candidate.unit_id)
                continue
            judgment = self._judge.judge(requirement, unit)
            judgments.append(judgment)
        return judgments
