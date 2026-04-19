from __future__ import annotations

import re
from typing import Dict, List

from app.core.config import ScoringConfig
from app.core.text import tokenize_content
from app.domain.entities import CandidateTestCase, Requirement, RequirementFinding, TestCase, TraceLink
from app.domain.enums import LinkStatus, RuleFlag
from app.domain.value_objects import Evidence
from app.infrastructure.llm.base import LLMJudge
from app.infrastructure.rules.conflict_detector import RuleBasedConflictDetector

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


class RequirementCoverageEvaluator:
    def __init__(self, rules: RuleBasedConflictDetector, llm_judge: LLMJudge, scoring_config: ScoringConfig):
        self._rules = rules
        self._llm_judge = llm_judge
        self._scoring_config = scoring_config

    def evaluate(
        self,
        requirements: List[Requirement],
        test_cases_by_id: Dict[str, TestCase],
        candidate_map: Dict[str, List[CandidateTestCase]],
    ) -> List[RequirementFinding]:
        findings: List[RequirementFinding] = []

        for requirement in requirements:
            candidates = candidate_map.get(requirement.id, [])
            if not candidates:
                findings.append(
                    RequirementFinding(
                        requirement_id=requirement.id,
                        final_status=LinkStatus.MISSING,
                        explanation="No sufficiently relevant test case candidate was retrieved for the requirement.",
                        candidate_list=[],
                    )
                )
                continue

            links = [
                self._evaluate_candidate(requirement, test_cases_by_id[candidate.test_case_id], candidate)
                for candidate in candidates
                if candidate.test_case_id in test_cases_by_id
            ]
            links.sort(key=self._link_rank, reverse=True)
            best = links[0]
            findings.append(
                RequirementFinding(
                    requirement_id=requirement.id,
                    selected_best_match=best,
                    evaluated_links=links,
                    final_status=best.link_status,
                    explanation=best.explanation,
                    evidence=best.evidence,
                    candidate_list=candidates,
                    rule_flags=best.rule_flags,
                    judge_output=best.judge_output,
                )
            )
        return findings

    def _evaluate_candidate(self, requirement: Requirement, test_case: TestCase, candidate: CandidateTestCase) -> TraceLink:
        rule_evaluation = self._rules.evaluate(requirement, test_case)
        judge_output = self._llm_judge.evaluate(requirement, test_case)
        evidence = self._build_evidence(requirement, test_case)
        status = self._determine_status(requirement, test_case, candidate, rule_evaluation.flags, rule_evaluation.has_strong_conflict)
        explanation_parts = []
        if rule_evaluation.explanation:
            explanation_parts.append(rule_evaluation.explanation)
        if judge_output.explanation:
            explanation_parts.append(judge_output.explanation)
        if not explanation_parts:
            explanation_parts.append(self._default_explanation(status, candidate.retrieval_score))

        return TraceLink(
            requirement_id=requirement.id,
            test_case_id=test_case.id,
            retrieval_score=candidate.retrieval_score,
            rule_flags=rule_evaluation.flags,
            judge_score=judge_output.semantic_alignment_score,
            link_status=status,
            evidence=evidence,
            explanation=" ".join(explanation_parts),
            judge_output=judge_output,
        )

    def _determine_status(
        self,
        requirement: Requirement,
        test_case: TestCase,
        candidate: CandidateTestCase,
        flags: List[RuleFlag],
        has_strong_conflict: bool,
    ) -> LinkStatus:
        if has_strong_conflict:
            return LinkStatus.CONFLICT

        thresholds = self._scoring_config.thresholds
        requirement_tokens = tokenize_content(requirement.text)
        test_tokens = tokenize_content(f"{test_case.text} {test_case.expected_result or ''}")
        overlap = requirement_tokens & test_tokens
        overlap_ratio = len(overlap) / len(requirement_tokens) if requirement_tokens else 0.0

        if RuleFlag.NUMERIC_PARTIAL_MISMATCH in flags and overlap_ratio >= thresholds.minimal_overlap_threshold:
            return LinkStatus.PARTIAL

        if RuleFlag.EXPECTED_RESULT_MISSING in flags and overlap_ratio >= thresholds.minimal_overlap_threshold:
            return LinkStatus.PARTIAL

        if candidate.retrieval_score < thresholds.inadequate_score_threshold or overlap_ratio < thresholds.minimal_overlap_threshold:
            return LinkStatus.INADEQUATE

        req_numbers = set(NUMBER_RE.findall(requirement.text))
        test_numbers = set(NUMBER_RE.findall(f"{test_case.text} {test_case.expected_result or ''}"))
        if req_numbers and not req_numbers.issubset(test_numbers):
            return LinkStatus.PARTIAL

        if overlap_ratio < thresholds.adequate_overlap_threshold:
            return LinkStatus.PARTIAL

        return LinkStatus.ADEQUATE

    @staticmethod
    def _build_evidence(requirement: Requirement, test_case: TestCase) -> Evidence:
        requirement_tokens = tokenize_content(requirement.text)
        test_tokens = tokenize_content(f"{test_case.text} {test_case.expected_result or ''}")
        return Evidence(
            matched_keywords=sorted(requirement_tokens & test_tokens),
            requirement_numbers=NUMBER_RE.findall(requirement.text),
            test_numbers=NUMBER_RE.findall(f"{test_case.text} {test_case.expected_result or ''}"),
            section_hint=f"{requirement.section} -> {test_case.section}",
        )

    @staticmethod
    def _default_explanation(status: LinkStatus, retrieval_score: float) -> str:
        return f"Baseline assessment produced status {status.value} with retrieval score {retrieval_score:.2f}."

    @staticmethod
    def _link_rank(link: TraceLink) -> tuple:
        status_rank = {
            LinkStatus.ADEQUATE: 5,
            LinkStatus.CONFLICT: 4,
            LinkStatus.PARTIAL: 3,
            LinkStatus.INADEQUATE: 2,
            LinkStatus.MISSING: 1,
        }
        return status_rank.get(link.link_status, 0), link.retrieval_score
