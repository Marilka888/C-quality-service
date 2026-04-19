from __future__ import annotations

from app.domain.entities import Requirement, TestCase
from app.domain.value_objects import JudgeOutput
from app.infrastructure.llm.base import LLMJudge


class DisabledLLMJudge(LLMJudge):
    def evaluate(self, requirement: Requirement, test_case: TestCase) -> JudgeOutput:
        return JudgeOutput(
            explanation="LLM judge is disabled; baseline rule-based assessment was used.",
            raw_payload={"mode": "disabled"},
        )
