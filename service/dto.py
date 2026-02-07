from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

DocType = Literal["TZ", "PZ", "PMI", "OTHER"]

class Chunk(BaseModel):
    chunk_id: str
    section_path: Optional[str] = None
    text: str
    order: Optional[int] = None
    page: Optional[int] = None

class Document(BaseModel):
    doc_id: str
    doc_type: DocType
    chunks: List[Chunk]

class CheckCConfig(BaseModel):
    top_k: int = 10
    emb_threshold_ok: float = 0.72
    emb_threshold_partial: float = 0.62
    use_llm: bool = False
    llm_only_if_score_ge: float = 0.70

class CheckCRequest(BaseModel):
    package_id: str
    documents: List[Document]
    config: CheckCConfig = Field(default_factory=CheckCConfig)

class Candidate(BaseModel):
    chunk_id: str
    doc_type: DocType
    score_bm25: float = 0.0
    score_emb: float = 0.0
    score_final: float = 0.0
    text: Optional[str] = None

class Match(BaseModel):
    tz_req_key: str
    tz_text: str
    target_doc_type: DocType
    status: Literal["OK", "PARTIAL", "MISSING", "CONTRADICTION"]
    matched_chunk_id: Optional[str] = None
    score_final: float = 0.0
    rationale: str = ""
    top_candidates: List[Candidate] = Field(default_factory=list)

class Defect(BaseModel):
    defect_type: Literal["C1_MISSING", "C2_EXTRA", "C3_CONTRADICTION"]
    severity: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    tz_req_key: Optional[str] = None
    tz_text: Optional[str] = None
    target_doc_type: Optional[DocType] = None
    other_chunk_id: Optional[str] = None
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

class Stats(BaseModel):
    tz_requirements: int = 0
    missing_in_pz: int = 0
    missing_in_pmi: int = 0
    extra_in_pz: int = 0
    extra_in_pmi: int = 0
    contradictions: int = 0

class CheckCResponse(BaseModel):
    package_id: str
    stats: Stats
    matches: List[Match] = Field(default_factory=list)
    defects: List[Defect] = Field(default_factory=list)
