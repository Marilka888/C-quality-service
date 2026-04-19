from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.domain.c_quality_enums import (
    CoverageStatus,
    CoverageUnitType,
    LLMLabel,
    Modality,
    RequirementType,
)


class Constraint(BaseModel):
    """A single measurable constraint extracted from text."""

    kind: str  # e.g. "retention_period", "response_time", "generic"
    operator: str  # ">=", "<=", "=", ">", "<", "!="
    value: float
    unit: Optional[str] = None  # normalised unit string, e.g. "days", "sec", "ms"

    def __hash__(self) -> int:
        return hash((self.kind, self.operator, self.value, self.unit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constraint):
            return False
        return (
            self.kind == other.kind
            and self.operator == other.operator
            and self.value == other.value
            and self.unit == other.unit
        )


class RequirementUnit(BaseModel):
    req_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_document_id: str
    source_section_id: Optional[str] = None
    source_fragment_id: Optional[str] = None
    text: str
    normalized_text: str
    requirement_type: RequirementType = RequirementType.OTHER
    modality: Modality = Modality.UNKNOWN
    entities: List[str] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CoverageUnit(BaseModel):
    unit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_document_id: str
    target_doc_role: str
    section_id: Optional[str] = None
    fragment_id: Optional[str] = None
    unit_type: CoverageUnitType = CoverageUnitType.PARAGRAPH
    text: str
    normalized_text: str
    entities: List[str] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedCandidate(BaseModel):
    req_id: str
    unit_id: str
    target_document_id: str
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    constraint_overlap_score: float = 0.0
    section_prior_score: float = 0.0
    retrieval_score: float = 0.0


class PairJudgment(BaseModel):
    req_id: str
    unit_id: str
    target_document_id: str
    llm_label: LLMLabel = LLMLabel.IRRELEVANT
    llm_confidence: float = 0.0
    rule_adjusted_label: LLMLabel = LLMLabel.IRRELEVANT
    matched_aspects: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)
    conflict_aspects: List[str] = Field(default_factory=list)
    explanation: str = ""


class EvidenceItem(BaseModel):
    unit_id: str
    fragment_id: Optional[str] = None
    section_id: Optional[str] = None
    text: str
    retrieval_score: float
    judgment: PairJudgment


class RequirementCoverageResult(BaseModel):
    req_id: str
    source_document_id: str
    target_document_id: str
    target_doc_role: str
    status: CoverageStatus = CoverageStatus.MISSING
    evidence: List[EvidenceItem] = Field(default_factory=list)
    uncovered_aspects: List[str] = Field(default_factory=list)
    conflict_details: List[str] = Field(default_factory=list)


class DocumentCoverageReport(BaseModel):
    target_document_id: str
    target_doc_role: str
    total_requirements: int = 0
    covered: int = 0
    partial: int = 0
    missing: int = 0
    conflict: int = 0

    @property
    def coverage_rate(self) -> float:
        if self.total_requirements == 0:
            return 0.0
        return (self.covered + self.partial * 0.5) / self.total_requirements


class CoverageSummary(BaseModel):
    total_requirements: int = 0
    covered: int = 0
    partial: int = 0
    missing: int = 0
    conflict: int = 0

    @property
    def coverage_rate(self) -> float:
        if self.total_requirements == 0:
            return 0.0
        return (self.covered + self.partial * 0.5) / self.total_requirements


class CoverageAnalysisResult(BaseModel):
    job_id: str
    package_id: str
    source_document_id: str
    target_document_ids: List[str]
    summary: CoverageSummary
    document_reports: List[DocumentCoverageReport]
    requirement_results: List[RequirementCoverageResult]
    pair_judgments: Optional[List[PairJudgment]] = None
    warnings: List[str] = Field(default_factory=list)
