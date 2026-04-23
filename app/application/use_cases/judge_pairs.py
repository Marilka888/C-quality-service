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
        # Collect existing units once so we can dispatch to a batch judge
        # when the backend supports it (~10× faster for cross-encoders).
        pairs: List[CoverageUnit] = []
        for candidate in shortlist:
            unit = units_by_id.get(candidate.unit_id)
            if unit is None:
                logger.warning("unit_id=%s not found in index; skipping", candidate.unit_id)
                continue
            pairs.append(unit)

        if not pairs:
            return []

        judge_batch = getattr(self._judge, "judge_batch", None)
        if callable(judge_batch):
            try:
                return list(judge_batch(requirement, pairs))
            except Exception as exc:
                # If the batch path errors (e.g. network blip), fall back
                # to per-pair — we'd rather be slow than lose the request.
                logger.warning(
                    "judge.judge_batch failed (%s); falling back to per-pair", exc,
                )

        return [self._judge.judge(requirement, u) for u in pairs]
