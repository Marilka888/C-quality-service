from __future__ import annotations

import re
from typing import List, Sequence

from app.core.config import RuleConfig
from app.domain.entities import Requirement, TestCase
from app.domain.enums import RuleFlag
from app.domain.value_objects import RuleEvaluation

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
TIMEOUT_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(ms|мс|sec|s|сек|секунд[а-я]*|мин|minute[s]?)\b", re.IGNORECASE)
ATTEMPTS_RE = re.compile(r"\b(\d+)\s*(attempts?|попыт(?:ки|ок|ка))\b", re.IGNORECASE)
ROLE_RE = re.compile(r"\b(admin|administrator|user|operator|guest|администратор|пользователь|оператор|гость)\b", re.IGNORECASE)


def _normalize_numbers(values: Sequence[str]) -> List[str]:
    return sorted({value.replace(",", ".") for value in values})


class RuleBasedConflictDetector:
    def __init__(self, config: RuleConfig):
        self._config = config

    def evaluate(self, requirement: Requirement, test_case: TestCase) -> RuleEvaluation:
        requirement_text = requirement.text or ""
        test_text = " ".join(part for part in [test_case.text, test_case.expected_result or ""] if part)
        flags: List[RuleFlag] = []
        notes: List[str] = []
        strong_conflict = False

        req_numbers = _normalize_numbers(NUMBER_RE.findall(requirement_text))
        test_numbers = _normalize_numbers(NUMBER_RE.findall(test_text))
        if req_numbers and test_numbers and set(req_numbers) != set(test_numbers):
            shared_numbers = set(req_numbers).intersection(test_numbers)
            if not shared_numbers:
                flags.append(RuleFlag.NUMERIC_MISMATCH)
                notes.append(f"Different numeric constraints: requirement={req_numbers}, test={test_numbers}")
                strong_conflict = True
            else:
                flags.append(RuleFlag.NUMERIC_PARTIAL_MISMATCH)
                notes.append(
                    f"Partially different numeric constraints: requirement={req_numbers}, test={test_numbers}, shared={sorted(shared_numbers)}"
                )

        req_timeout = _normalize_numbers([match[0] for match in TIMEOUT_RE.findall(requirement_text)])
        test_timeout = _normalize_numbers([match[0] for match in TIMEOUT_RE.findall(test_text)])
        if req_timeout and test_timeout and req_timeout != test_timeout:
            flags.append(RuleFlag.TIMEOUT_MISMATCH)
            notes.append(f"Different timeout values: requirement={req_timeout}, test={test_timeout}")
            strong_conflict = True

        req_attempts = _normalize_numbers([match[0] for match in ATTEMPTS_RE.findall(requirement_text)])
        test_attempts = _normalize_numbers([match[0] for match in ATTEMPTS_RE.findall(test_text)])
        if req_attempts and test_attempts and req_attempts != test_attempts:
            flags.append(RuleFlag.ATTEMPT_MISMATCH)
            notes.append(f"Different attempt counts: requirement={req_attempts}, test={test_attempts}")
            strong_conflict = True

        req_roles = {match.lower() for match in ROLE_RE.findall(requirement_text)}
        test_roles = {match.lower() for match in ROLE_RE.findall(test_text)}
        if req_roles and test_roles and req_roles != test_roles:
            flags.append(RuleFlag.ROLE_MISMATCH)
            notes.append(f"Different roles mentioned: requirement={sorted(req_roles)}, test={sorted(test_roles)}")
            strong_conflict = True

        if self._has_direction_conflict(requirement_text, test_text):
            flags.append(RuleFlag.RANGE_DIRECTION_CONFLICT)
            notes.append("Range direction differs: one side says upper bound, another lower bound.")
            strong_conflict = True

        if self._config.require_expected_result_for_assertive_tests and self._expected_result_is_required(test_case):
            if not test_case.expected_result:
                flags.append(RuleFlag.EXPECTED_RESULT_MISSING)
                notes.append("Expected result is missing for an assertive or verificative test step.")

        return RuleEvaluation(
            flags=flags,
            has_strong_conflict=strong_conflict,
            explanation="; ".join(notes) if notes else None,
        )

    @staticmethod
    def _has_direction_conflict(requirement_text: str, test_text: str) -> bool:
        upper_markers = ("не более", "at most", "no more than", "max", "maximum")
        lower_markers = ("не менее", "at least", "no less than", "min", "minimum")
        requirement_upper = any(marker in requirement_text.lower() for marker in upper_markers)
        requirement_lower = any(marker in requirement_text.lower() for marker in lower_markers)
        test_upper = any(marker in test_text.lower() for marker in upper_markers)
        test_lower = any(marker in test_text.lower() for marker in lower_markers)
        return (requirement_upper and test_lower) or (requirement_lower and test_upper)

    @staticmethod
    def _expected_result_is_required(test_case: TestCase) -> bool:
        markers = ("должен", "must", "verify", "проверить", "ожидается", "ожидаемый")
        combined = f"{test_case.text} {test_case.expected_result or ''}".lower()
        return any(marker in combined for marker in markers)
