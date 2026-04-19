from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.domain.enums import LinkStatus, RuleFlag


class Evidence(BaseModel):
    matched_keywords: List[str] = Field(default_factory=list)
    requirement_numbers: List[str] = Field(default_factory=list)
    test_numbers: List[str] = Field(default_factory=list)
    section_hint: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    relevance_score: Optional[float] = None
    semantic_alignment_score: Optional[float] = None
    constraint_alignment_score: Optional[float] = None
    expected_result_alignment_score: Optional[float] = None
    verification_sufficiency_score: Optional[float] = None
    recommended_status: Optional[LinkStatus] = None
    explanation: Optional[str] = None
    raw_payload: Optional[Dict] = None


class RuleEvaluation(BaseModel):
    flags: List[RuleFlag] = Field(default_factory=list)
    has_strong_conflict: bool = False
    explanation: Optional[str] = None
