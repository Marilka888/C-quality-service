from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.domain.enums import FindingType, LinkStatus, RuleFlag
from app.domain.value_objects import Evidence, JudgeOutput


class Requirement(BaseModel):
    id: str
    text: str
    section: str
    source_doc_id: str
    page: Optional[int | str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    id: str
    text: str
    expected_result: Optional[str] = None
    section: str
    source_doc_id: str
    page: Optional[int | str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateTestCase(BaseModel):
    test_case_id: str
    retrieval_score: float
    lexical_score: float = 0.0
    section_score: float = 0.0
    metadata_score: float = 0.0
    embedding_score: float = 0.0


class TraceLink(BaseModel):
    requirement_id: str
    test_case_id: Optional[str] = None
    retrieval_score: float = 0.0
    rule_flags: List[RuleFlag] = Field(default_factory=list)
    judge_score: Optional[float] = None
    link_status: LinkStatus = LinkStatus.MISSING
    evidence: Evidence = Field(default_factory=Evidence)
    explanation: str = ""
    judge_output: Optional[JudgeOutput] = None


class RequirementFinding(BaseModel):
    type: FindingType = FindingType.REQUIREMENT
    requirement_id: str
    selected_best_match: Optional[TraceLink] = None
    evaluated_links: List[TraceLink] = Field(default_factory=list)
    final_status: LinkStatus
    explanation: str
    evidence: Evidence = Field(default_factory=Evidence)
    candidate_list: List[CandidateTestCase] = Field(default_factory=list)
    rule_flags: List[RuleFlag] = Field(default_factory=list)
    judge_output: Optional[JudgeOutput] = None


class OrphanTestFinding(BaseModel):
    type: FindingType = FindingType.ORPHAN_TEST
    test_id: str
    explanation: str
    similarity_evidence: List[TraceLink] = Field(default_factory=list)


class SummaryMetrics(BaseModel):
    total_requirements: int
    adequate_count: int
    partial_count: int
    inadequate_count: int
    missing_count: int
    conflict_count: int
    orphan_test_count: int


class AggregatedMetrics(BaseModel):
    adequate_coverage_rate: float
    partial_coverage_rate: float
    missing_rate: float
    conflict_rate: float
    orphan_test_rate: float
    score_c: float


class TraceabilityReport(BaseModel):
    summary: SummaryMetrics
    aggregated_metrics: AggregatedMetrics
    accepted_links: List[TraceLink] = Field(default_factory=list)
    conflicting_links: List[TraceLink] = Field(default_factory=list)
    uncovered_requirements: List[str] = Field(default_factory=list)
    orphan_test_cases: List[OrphanTestFinding] = Field(default_factory=list)
    detailed_findings: List[RequirementFinding] = Field(default_factory=list)
