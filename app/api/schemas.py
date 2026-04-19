from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RequirementInput(BaseModel):
    id: str
    text: str
    section: str
    source_doc_id: str
    page: Optional[int | str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestCaseInput(BaseModel):
    id: str
    text: str
    expected_result: Optional[str] = None
    section: str
    source_doc_id: str
    page: Optional[int | str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluateTraceabilityRequest(BaseModel):
    requirements: List[RequirementInput]
    test_cases: List[TestCaseInput]
    top_k: int = Field(default=5, ge=1, le=20)
    min_retrieval_score: float = Field(default=0.2, ge=0.0, le=1.0)
    use_llm: bool = False
    use_embeddings: bool = False


class EvaluateTraceabilityResponse(BaseModel):
    report: Dict[str, Any]
