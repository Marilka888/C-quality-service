"""
Request / response DTOs for the /coverage/analyze endpoint.

The PreparedArtifact schema is intentionally minimal — only stable fields are
required; any extra fields from prepare-service pass through as metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Document roles (B3): strictly enumerated at the API boundary so that any
# unknown role from a misconfigured caller produces a clean 422 instead of
# being silently ignored downstream. Internal code paths additionally treat
# any unrecognised string as "unknown" with a warning, but that fallback
# never fires when callers go through the FastAPI request schema below.
# ---------------------------------------------------------------------------

DocRole = Literal["tz", "pmi", "pz", "unknown"]
ALLOWED_DOC_ROLES: tuple[str, ...] = ("tz", "pmi", "pz", "unknown")


def _normalize_role(value: Any) -> Any:
    """Lowercase / strip incoming role strings before Literal validation.

    Pydantic Literal is case-sensitive, so callers that historically sent
    "TZ" or " pmi " would otherwise be rejected. Lowercasing is a small
    backward-compat concession; the allow-list itself is still strict.
    """
    if isinstance(value, str):
        return value.strip().lower()
    return value


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
    doc_role: DocRole                    # B3: strict enum, was free-form str
    sections: List[SectionArtifact] = Field(default_factory=list)
    fragments: List[FragmentArtifact] = Field(default_factory=list)
    requirement_candidates: Optional[List[RequirementCandidateArtifact]] = None
    sentences: Optional[List[Dict[str, Any]]] = None

    @field_validator("doc_role", mode="before")
    @classmethod
    def _normalize_doc_role(cls, v: Any) -> Any:
        return _normalize_role(v)


class DocumentInput(BaseModel):
    document_id: str
    doc_role: DocRole                    # B3: strict enum, was free-form str
    prepared_artifact: PreparedArtifact

    @field_validator("doc_role", mode="before")
    @classmethod
    def _normalize_doc_role(cls, v: Any) -> Any:
        return _normalize_role(v)


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
    # B3: strict enum on source/target roles too. Both go through the same
    # lowercase-trim normalisation as DocumentInput.doc_role.
    source_doc_role: DocRole = "tz"
    target_doc_roles: List[DocRole] = Field(default_factory=lambda: ["pmi", "pz"])
    documents: List[DocumentInput]
    options: CoverageOptions = Field(default_factory=CoverageOptions)

    @field_validator("source_doc_role", mode="before")
    @classmethod
    def _normalize_source(cls, v: Any) -> Any:
        return _normalize_role(v)

    @field_validator("target_doc_roles", mode="before")
    @classmethod
    def _normalize_targets(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_normalize_role(x) for x in v]
        return v


class CoverageAnalysisResponse(BaseModel):
    result: Dict[str, Any]
