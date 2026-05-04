from __future__ import annotations

from enum import Enum


class CoverageStatus(str, Enum):
    COVERED = "COVERED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"


class RequirementType(str, Enum):
    """Functional class of a requirement.

    Drives type-aware retrieval (which target document is the natural
    home), CONFLICT validation (same-aspect check), applicability
    (whether the requirement should be checked at all in target X) and
    severity (how MISSING/CONFLICT contributes to criticalCount).

    Legacy values (PERFORMANCE / SECURITY / LOGGING / STORAGE / INTERFACE
    / FUNCTIONAL / OTHER) are preserved for backward compat with existing
    artifacts. New values describe the broader taxonomy needed to stop
    treating delivery / process / economic requirements as functional
    coverage.
    """
    # ── Legacy / functional axes (kept) ─────────────────────────────
    PERFORMANCE = "performance"
    SECURITY = "security"
    LOGGING = "logging"
    STORAGE = "storage"
    INTERFACE = "interface"
    FUNCTIONAL = "functional"
    # ── Extended taxonomy (PR-G+ refactor) ──────────────────────────
    RELIABILITY = "reliability"
    DATA_IO = "data_io"
    ARCHITECTURE_IMPLEMENTATION = "architecture_implementation"
    DOCUMENTATION_REQUIREMENT = "documentation_requirement"
    DELIVERY_REQUIREMENT = "delivery_requirement"
    PROCESS_REQUIREMENT = "process_requirement"
    ENVIRONMENT_REQUIREMENT = "environment_requirement"
    ECONOMIC_OR_NEED = "economic_or_need"
    OTHER = "other"


class Applicability(str, Enum):
    """Whether a requirement should be checked for coverage in a given
    target document. APPLICABLE — yes, surface MISSING / CONFLICT
    normally. NOT_APPLICABLE — the requirement type doesn't fit this
    target (e.g. ARCHITECTURE in PMI). OUT_OF_SCOPE — the requirement
    isn't a coverage requirement at all (delivery, process, …).

    Non-APPLICABLE rows must NOT inflate criticalCount or pull the
    package grade down; the orchestrator dims them in the UI."""
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class CoverageRequirementLevel(str, Enum):
    """Stronger applicability signal that drives both the LLM-call gate
    AND the should_affect_critical aggregation flag.

    Difference from Applicability:
      * Applicability is binary-ish (yes / no / not-our-business).
      * Level distinguishes "must find coverage" (REQUIRED) from
        "nice to find but not critical" (OPTIONAL). Aggregator uses
        OPTIONAL_NOT_FOUND status (via debug field) when REQUIRED would
        otherwise have produced critical MISSING.

    Mapping is in `app.application.use_cases.applicability.
    coverage_requirement_level_for(req_type, target_role)`.
    """
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceStrength(str, Enum):
    """Discrete bin of retrieval_score for a single candidate.

    Used by AdaptiveCandidateSelector to decide how many candidates to
    send to the LLM and by EvidenceBasedCoverageAggregator to refuse
    confident verdicts on weak retrieval. Default thresholds in
    `CoverageRetrievalConfig.evidence_strength_*`.

    NO_EVIDENCE means the score is so low that even calling the LLM
    is wasteful — the row is downgraded to MISSING_NO_EVIDENCE
    (REQUIRED) or OPTIONAL_NOT_FOUND (OPTIONAL) directly.
    """
    STRONG = "STRONG"
    MEDIUM = "MEDIUM"
    WEAK = "WEAK"
    NO_EVIDENCE = "NO_EVIDENCE"


class CoverageUnitType(str, Enum):
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TEST_STEP = "test_step"
    EXPECTED_RESULT = "expected_result"
    PRECONDITION = "precondition"
    TABLE_ROW_TEXT = "table_row_text"


class Modality(str, Enum):
    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"
    MAY = "may"
    UNKNOWN = "unknown"


class LLMLabel(str, Enum):
    COVERED = "COVERED"
    PARTIAL = "PARTIAL"
    CONFLICT = "CONFLICT"
    IRRELEVANT = "IRRELEVANT"
