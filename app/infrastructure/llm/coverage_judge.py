"""Abstract interface for pair-level LLM judge."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit


class CoverageJudge(ABC):
    @abstractmethod
    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        """Return a PairJudgment for the (req, unit) pair."""
