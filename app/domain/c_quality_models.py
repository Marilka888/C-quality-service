from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.domain.c_quality_enums import (
    Applicability,
    CoverageRequirementLevel,
    CoverageStatus,
    CoverageUnitType,
    EvidenceStrength,
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
    # ── PR-K explainability (additive) ──────────────────────────────
    # Discrete bin of retrieval_score, see EvidenceStrength.
    evidence_strength: EvidenceStrength = EvidenceStrength.NO_EVIDENCE
    # Short human-readable reason for the score: which component drove
    # the result. Filled by CandidateRetriever.
    score_reason: Optional[str] = None
    # Did AdaptiveCandidateSelector send this candidate to the LLM?
    selected_for_llm: bool = False
    # Reranker telemetry. reranker_used==True means the cross-encoder
    # was applied to this shortlist (per ConditionalReranker rules);
    # reranker_score is the raw reranker output before re-sorting.
    reranker_used: bool = False
    reranker_score: Optional[float] = None
    # Filled after the LLM judge runs (so a single CandidateEvidence
    # carries the full retrieval-and-judging trace).
    judge_label: Optional[str] = None
    judge_confidence: Optional[float] = None
    grounding_passed: Optional[bool] = None
    verifier_actions: List[str] = Field(default_factory=list)


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
    # BUG-3: phrases the LLM claimed it pulled from the evidence. Each entry
    # MUST be a substring of the corresponding CoverageUnit text — that is
    # the grounding contract enforced in the response-parser. Empty for
    # IRRELEVANT, for the disabled judge, and for legacy payloads.
    cited_phrases: List[str] = Field(default_factory=list)
    # BUG-3 / BUG-9: True when the judgment cannot be trusted on its own —
    # set by the response-parser when `cited_phrases` aren't grounded in
    # the evidence, or by the pipeline when retrieval scores are below
    # CoverageRetrievalConfig.evidence_floor. Aggregator propagates this
    # to the RequirementCoverageResult so the orchestrator / UI can dim
    # the row instead of rendering it as an authoritative verdict.
    low_confidence: bool = False
    # PR-K P0: separate "grounding failed" (cited_phrases don't substring-match
    # evidence — LLM hallucination) from "retrieval below floor" (retrieval
    # score < evidence_floor — retrieval-quality issue, not a grounding bug).
    # The aggregator only treats `grounding_failed=True` as ungrounded; a
    # below-floor judgment with proper citations still counts as grounded
    # and can produce COVERED — just with low_confidence flag set on the row.
    # Real-package symptom (Polyakov 0.20::sent1): "Время восстановления..."
    # got conf=1.0, fully grounded, but retrieval=0.44 < floor=0.5 → old
    # code marked low_confidence which the aggregator interpreted as
    # "ungrounded" → MISSING_LOW_GROUNDING. Real verdict: COVERED.
    grounding_failed: bool = False
    # PR-K: deterministic verifier audit trail. Each entry is a short
    # tag describing what the rule-based PairVerifier did with this
    # judgment (e.g. "conflict_confirmed_numeric",
    # "demote_covered_constraints_missing", "no_op_kept_label"). The
    # aggregator inspects these tags to decide whether a CONFLICT
    # verdict was actually confirmed by a deterministic rule (gate
    # against LLM-only conflicts that retrieval-quality couldn't justify).
    verifier_actions: List[str] = Field(default_factory=list)


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
    # BUG-3 / BUG-9: True when the verdict cannot be trusted authoritatively
    # — either the LLM judgement was not grounded in retrieved evidence
    # (BUG-3) or the maximum evidence retrieval score sat below
    # CoverageRetrievalConfig.evidence_floor (BUG-9). The orchestrator and
    # the UI should render such rows in a "low-confidence" style and the
    # status itself should be treated as MISSING-equivalent for grade /
    # criticalCount calculations.
    low_confidence: bool = False
    # BUG-12: source-side context for the requirement so the orchestrator /
    # UI can render the requirement card with proper locator instead of
    # the opaque `req_id` hash. Populated from the RequirementUnit at
    # aggregation time; legacy artifacts produced before this contract
    # leave them as empty strings.
    req_text: Optional[str] = None
    req_section_title: Optional[str] = None
    req_section_id: Optional[str] = None
    req_number: Optional[str] = None
    # PR-F follow-up: authoritative human-readable rationale chosen at
    # aggregation time. Mirrors the explanation of the judgment that
    # determined `status`; falls back to a default Russian sentence for
    # MISSING rows when no judgment carries useful text. Decoupling this
    # from per-judgment explanations means the UI never has to guess
    # which evidence's judgment to render.
    rationale: Optional[str] = None
    # ── PR-K aggregation diagnostics (additive) ─────────────────────
    # Sub-status code that refines `status` for UI rendering and
    # downstream rollups. Examples:
    #   "MISSING_NO_EVIDENCE"     — retrieval found nothing strong
    #                               enough to even ask the LLM.
    #   "MISSING_LOW_GROUNDING"   — LLM said COVERED but cited phrases
    #                               weren't substring-grounded.
    #   "MISSING_LOW_CONFIDENCE"  — judge confidence below threshold.
    #   "OPTIONAL_NOT_FOUND"      — REQUIRED-equivalent missing but
    #                               applicability is OPTIONAL.
    #   "COVERED"                 — clean grounded covered.
    #   "PARTIAL"                 — partial aspects covered.
    #   "CONFLICT_VERIFIED"       — explicit numeric/aspect contradiction.
    # Old readers see the canonical CoverageStatus on the wire; new
    # readers can use `status_subcode` to render finer-grained badges.
    status_subcode: Optional[str] = None
    # The unit that drove the final verdict (winning judgment).
    winning_candidate_id: Optional[str] = None
    # Final confidence for the row (post-aggregation), in [0, 1].
    final_confidence: float = 0.0
    # Plain-English aggregator reason. Filled even on COVERED ("strong
    # evidence + grounded + LLM confident") so the UI can show a
    # tooltip with the decision logic.
    aggregation_reason: Optional[str] = None
    # PR-K: tighter applicability signal — REQUIRED / OPTIONAL /
    # NOT_APPLICABLE. Distinct from `applicability` (which is binary).
    coverage_requirement_level: CoverageRequirementLevel = CoverageRequirementLevel.REQUIRED
    # PR-K full evidence/retrieval/judge/verifier trace. Top-level
    # presence is gated by CoverageConfig.debug.enabled — when
    # disabled, this is None to keep the wire compact. UI uses this
    # to render the "why" panel.
    evidence_trace: Optional[Dict[str, Any]] = None
    # ── Type-aware refactor ─────────────────────────────────────────
    # Functional class of the requirement (FUNCTIONAL / SECURITY /
    # PERFORMANCE / DELIVERY_REQUIREMENT / …). Drives applicability
    # and severity. Defaults to OTHER when no rule fires.
    requirement_type: RequirementType = RequirementType.OTHER
    # Whether this row should be checked at all in the target document.
    # OUT_OF_SCOPE = the requirement type itself isn't a coverage
    # requirement (delivery/process). NOT_APPLICABLE = it's a coverage
    # requirement but doesn't fit this target (e.g. ARCHITECTURE in PMI).
    # Both flags exclude the row from criticalCount and from the C-axis
    # sub-score in the package grade.
    applicability: Applicability = Applicability.APPLICABLE
    # Severity of this row when status != COVERED. "low" / "medium" /
    # "high". Computed from (requirement_type, target_role, status,
    # applicability) — see applicability.severity_for. The orchestrator
    # uses this to pick the per-row priority in CRequirement.
    severity: str = "low"
    # Whether this row contributes to package-level criticalCount.
    # Only APPLICABLE rows in CONFLICT / safety-relevant MISSING.
    should_affect_critical: bool = False
    # Whether this row contributes to the C-axis sub-score in the
    # package grade. Excludes OUT_OF_SCOPE / NOT_APPLICABLE.
    should_affect_grade: bool = True


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
