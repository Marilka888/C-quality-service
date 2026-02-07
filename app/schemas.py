from typing import List, Literal, Optional
from pydantic import BaseModel

class Candidate(BaseModel):
    id: str
    text: str
    meta: dict = {}

class JudgePair(BaseModel):
    tz: Candidate
    pmi: Optional[Candidate] = None

class JudgeRequest(BaseModel):
    packageId: str
    # Backward compatible mode (explicit pairs)
    pairs: Optional[List[JudgePair]] = None

    # Pipeline mode: provide all TZ requirements and all PMI tests.
    tz_requirements: Optional[List[Candidate]] = None
    pmi_tests: Optional[List[Candidate]] = None

    # Retrieval settings
    top_k: int = 5
    # NOTE: early versions used `min_score`, later drafts used `min_similarity`.
    # Accept both to keep clients compatible.
    min_score: float = 0.35
    min_similarity: Optional[float] = None

class JudgeDecision(BaseModel):
    tzId: str
    pmiId: Optional[str]
    verdict: Literal["COVERED", "MISSING", "CONFLICT", "EXTRA"]
    explanation: str

class JudgeResponse(BaseModel):
    packageId: str
    decisions: List[JudgeDecision]
