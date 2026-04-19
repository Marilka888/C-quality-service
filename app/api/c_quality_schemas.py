"""
Request / response DTOs for the /coverage/analyze endpoint.

The PreparedArtifact schema is intentionally minimal — only stable fields are
required; any extra fields from prepare-service pass through as metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# PreparedArtifact sub-schemas (mirrors prepare-service contract)
# ---------------------------------------------------------------------------


class FragmentArtifact(BaseModel):
    fragment_id: Optional[str] = None
    text: str
    kind: Optional[str] = None          # paragraph / list_item / test_step / …
    section_id: Optional[str] = None
    page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SectionArtifact(BaseModel):
    section_id: str
    title: str
    level: Optional[int] = None


class RequirementCandidateArtifact(BaseModel):
    req_id: Optional[str] = None
    text: str
    section_id: Optional[str] = None
    fragment_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PreparedArtifact(BaseModel):
    document_id: str
    package_id: Optional[str] = None
    doc_role: str                        # "tz" | "pmi" | "pz" | …
    sections: List[SectionArtifact] = Field(default_factory=list)
    fragments: List[FragmentArtifact] = Field(default_factory=list)
    requirement_candidates: Optional[List[RequirementCandidateArtifact]] = None
    # TODO: use sentences[] for finer granularity when prepare-service exposes them
    sentences: Optional[List[Dict[str, Any]]] = None


class DocumentInput(BaseModel):
    document_id: str
    doc_role: str
    prepared_artifact: PreparedArtifact


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------


class CoverageOptions(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)
    enable_llm_judge: bool = False
    enable_rule_verification: bool = True
    min_retrieval_score: float = Field(default=0.05, ge=0.0, le=1.0)
    # "auto" | "sections" | "candidates" | "fragments"
    requirement_extraction: str = "auto"


class CoverageAnalysisRequest(BaseModel):
    job_id: Optional[str] = None
    package_id: str
    source_doc_role: str = "tz"
    target_doc_roles: List[str] = Field(default_factory=lambda: ["pmi", "pz"])
    documents: List[DocumentInput]
    options: CoverageOptions = Field(default_factory=CoverageOptions)


class CoverageAnalysisResponse(BaseModel):
    result: Dict[str, Any]
