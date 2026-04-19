from __future__ import annotations

from app.domain.entities import Requirement, TestCase
from app.domain.value_objects import JudgeOutput


class LLMJudge:
    def evaluate(self, requirement: Requirement, test_case: TestCase) -> JudgeOutput:
        raise NotImplementedError
