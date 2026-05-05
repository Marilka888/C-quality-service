"""
Type-aware routing matrix for C-quality.

Two pure functions:

  applicability_for(req_type, target_role)   → Applicability
  severity_for(req_type, target_role, status, applicability) → "low"|"medium"|"high"

`applicability_for` answers: should this requirement type be checked at
all in this target document? DELIVERY_REQUIREMENT in PMI is
OUT_OF_SCOPE — the LMS-submission rule isn't a coverage requirement.
ARCHITECTURE_IMPLEMENTATION in PMI is NOT_APPLICABLE — architecture
lives in PZ; PMI may carry auxiliary mentions but coverage is judged in
PZ. Both flags downgrade the row's contribution to criticalCount.

`severity_for` is the per-row priority that drives the package-level
critical / grade rollups. SECURITY and PERFORMANCE missing in PMI are
high; functional requirements are high in PMI / medium in PZ when the
PMI side covers them; documentation requirements are medium; delivery /
process / economic requirements are low (or zero, if non-applicable).

These rules are deliberately compact and keyword-free — they take only
the typed RequirementType and the target's doc role. No hardcoded
matches against specific package texts or filenames.
"""
from __future__ import annotations

from app.domain.c_quality_enums import (
    Applicability,
    CoverageRequirementLevel,
    CoverageStatus,
    EvidenceStrength,
    RequirementType,
)


_OUT_OF_SCOPE_TYPES = frozenset({
    RequirementType.DELIVERY_REQUIREMENT,
    RequirementType.PROCESS_REQUIREMENT,
})

_PZ_ONLY_TYPES = frozenset({
    RequirementType.ARCHITECTURE_IMPLEMENTATION,
    RequirementType.ECONOMIC_OR_NEED,
})

# Documentation meta-requirements (GOST sections, document composition):
# only checked in PMI (where the TZ may specify what the PMI must
# contain); PZ and other target roles are not the natural coverage home
# for documentation structure requirements. A-quality handles those.
_PMI_ONLY_TYPES = frozenset({
    RequirementType.DOCUMENTATION_REQUIREMENT,
})


def applicability_for(
    req_type: RequirementType,
    target_role: str,
) -> Applicability:
    role = (target_role or "").strip().lower()

    if req_type in _OUT_OF_SCOPE_TYPES:
        return Applicability.OUT_OF_SCOPE

    if req_type in _PZ_ONLY_TYPES:
        return Applicability.APPLICABLE if role == "pz" else Applicability.NOT_APPLICABLE

    if req_type in _PMI_ONLY_TYPES:
        return Applicability.APPLICABLE if role == "pmi" else Applicability.NOT_APPLICABLE

    return Applicability.APPLICABLE


def should_affect_critical(
    req_type: RequirementType,
    applicability: Applicability,
    status: CoverageStatus,
    target_role: str = "",
) -> bool:
    """A row contributes to package criticalCount only when it is
    APPLICABLE, in a "bad" status (CONFLICT / MISSING), and of a type
    that genuinely matters for safety / correctness. DOCUMENTATION /
    INTERFACE / ENVIRONMENT MISSING are warnings, not critical.

    `target_role` is optional for backwards compatibility — when supplied,
    PZ-demoted spec types (functional/data_io/performance/storage/logging)
    contribute only as warnings even when MISSING (the PZ doesn't restate
    the TZ spec, so functional MISSING in PZ is a documentation pattern,
    not a correctness gap).
    """
    if applicability != Applicability.APPLICABLE:
        return False
    if status == CoverageStatus.COVERED:
        return False
    if status == CoverageStatus.CONFLICT:
        # Real CONFLICT (after same-aspect validation) is always critical.
        return True
    role = (target_role or "").strip().lower()
    if role == "pz" and req_type in _PZ_DEMOTED_TO_OPTIONAL:
        return False
    # MISSING / PARTIAL: critical only for safety-relevant types.
    return req_type in {
        RequirementType.FUNCTIONAL,
        RequirementType.SECURITY,
        RequirementType.PERFORMANCE,
        RequirementType.RELIABILITY,
        RequirementType.DATA_IO,
        RequirementType.ARCHITECTURE_IMPLEMENTATION,
    } and status == CoverageStatus.MISSING


def should_affect_grade(
    req_type: RequirementType,
    applicability: Applicability,
) -> bool:
    """Whether the row's status contributes to the C-axis sub-score in
    the package grade. Out-of-scope / non-applicable rows are excluded
    so the grade reflects only checks that were actually meaningful."""
    return applicability == Applicability.APPLICABLE


# ── PR-K: REQUIRED / OPTIONAL / NOT_APPLICABLE routing ───────────────
#
# `applicability_for` already returns OUT_OF_SCOPE / NOT_APPLICABLE /
# APPLICABLE. PR-K adds a finer split: an APPLICABLE row may be
# REQUIRED (must find coverage; missing = critical) or OPTIONAL
# (nice to find but missing is acceptable). The aggregator uses this
# to choose between MISSING (REQUIRED) and OPTIONAL_NOT_FOUND
# (OPTIONAL) sub-statuses.

# REQUIRED requirement-types: any APPLICABLE pair must find coverage.
_REQUIRED_TYPES = frozenset({
    RequirementType.FUNCTIONAL,
    RequirementType.SECURITY,
    RequirementType.PERFORMANCE,
    RequirementType.RELIABILITY,
    RequirementType.DATA_IO,
    RequirementType.ARCHITECTURE_IMPLEMENTATION,
    RequirementType.STORAGE,
    RequirementType.LOGGING,
})

# OPTIONAL requirement-types: APPLICABLE but missing is not critical.
_OPTIONAL_TYPES = frozenset({
    RequirementType.INTERFACE,
    RequirementType.DOCUMENTATION_REQUIREMENT,
    RequirementType.ENVIRONMENT_REQUIREMENT,
    RequirementType.OTHER,
})


# PZ (пояснительная записка) describes IMPLEMENTATION, not the
# functional spec — that lives in TZ. Real ВКР-class PZ docs frequently
# omit verbatim restatement of functional / data_io / interface
# requirements; instead they describe how the implementation realises
# them (architecture, components, data flow). Treating those types as
# REQUIRED-in-PZ produces a flood of false-MISSING criticals (Polyakov:
# 19/20 functional reqs marked MISSING in PZ even though the system
# evidently works — the PZ just doesn't restate the spec).
#
# Pragmatic fix: in PZ, demote spec-class types to OPTIONAL. They are
# still checked; if the PZ does happen to describe them they get
# COVERED/PARTIAL credit, but their absence stops triggering criticalCount.
# ARCHITECTURE_IMPLEMENTATION / RELIABILITY / SECURITY remain REQUIRED
# in PZ — those genuinely belong in a design document.
_PZ_DEMOTED_TO_OPTIONAL = frozenset({
    RequirementType.FUNCTIONAL,
    RequirementType.DATA_IO,
    RequirementType.PERFORMANCE,
    RequirementType.STORAGE,
    RequirementType.LOGGING,
})


def coverage_requirement_level_for(
    req_type: RequirementType,
    target_role: str,
) -> CoverageRequirementLevel:
    """REQUIRED / OPTIONAL / NOT_APPLICABLE for a (type, target) pair.

    REQUIRED — coverage must be found; missing = critical.
    OPTIONAL — coverage is nice-to-have; missing = warning, not critical.
    NOT_APPLICABLE — should not check at all (delivery, process, …).
    """
    appl = applicability_for(req_type, target_role)
    if appl != Applicability.APPLICABLE:
        return CoverageRequirementLevel.NOT_APPLICABLE
    role = (target_role or "").strip().lower()
    if role == "pz" and req_type in _PZ_DEMOTED_TO_OPTIONAL:
        return CoverageRequirementLevel.OPTIONAL
    if req_type in _REQUIRED_TYPES:
        return CoverageRequirementLevel.REQUIRED
    return CoverageRequirementLevel.OPTIONAL


def evidence_strength_from_score(
    retrieval_score: float,
    strong: float = 0.45,
    medium: float = 0.25,
    weak: float = 0.12,
) -> EvidenceStrength:
    """Discretise a retrieval_score into one of four bins.

    Defaults match `CoverageRetrievalConfig.evidence_strength_*`. Pass
    config-derived values when calling from the pipeline so a researcher
    can retune without touching code.
    """
    s = float(retrieval_score or 0.0)
    if s >= strong:
        return EvidenceStrength.STRONG
    if s >= medium:
        return EvidenceStrength.MEDIUM
    if s >= weak:
        return EvidenceStrength.WEAK
    return EvidenceStrength.NO_EVIDENCE


def severity_for(
    req_type: RequirementType,
    target_role: str,
    status: CoverageStatus,
    applicability: Applicability,
) -> str:
    """Return "low" / "medium" / "high". Used as the docback `priority`
    on CRequirement so the orchestrator's status banner and the UI's
    sort order match the type-aware semantics."""
    if applicability != Applicability.APPLICABLE:
        return "low"
    if status == CoverageStatus.COVERED:
        return "low"
    if status == CoverageStatus.CONFLICT:
        return "high"

    role = (target_role or "").strip().lower()

    # MISSING / PARTIAL severity matrix.
    if req_type == RequirementType.SECURITY:
        return "high"
    if req_type == RequirementType.PERFORMANCE:
        return "high" if role == "pmi" else "medium"
    if req_type == RequirementType.FUNCTIONAL:
        return "high" if role == "pmi" else "medium"
    if req_type == RequirementType.RELIABILITY:
        return "high" if role == "pmi" else "medium"
    if req_type == RequirementType.ARCHITECTURE_IMPLEMENTATION:
        return "high" if role == "pz" else "low"
    if req_type == RequirementType.DATA_IO:
        return "medium"
    if req_type == RequirementType.DOCUMENTATION_REQUIREMENT:
        return "medium"
    if req_type == RequirementType.INTERFACE:
        return "medium"
    if req_type == RequirementType.ENVIRONMENT_REQUIREMENT:
        return "low"
    return "low"
