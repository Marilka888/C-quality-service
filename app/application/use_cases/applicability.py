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
    CoverageStatus,
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


def applicability_for(
    req_type: RequirementType,
    target_role: str,
) -> Applicability:
    role = (target_role or "").strip().lower()

    if req_type in _OUT_OF_SCOPE_TYPES:
        return Applicability.OUT_OF_SCOPE

    if req_type in _PZ_ONLY_TYPES:
        return Applicability.APPLICABLE if role == "pz" else Applicability.NOT_APPLICABLE

    return Applicability.APPLICABLE


def should_affect_critical(
    req_type: RequirementType,
    applicability: Applicability,
    status: CoverageStatus,
) -> bool:
    """A row contributes to package criticalCount only when it is
    APPLICABLE, in a "bad" status (CONFLICT / MISSING), and of a type
    that genuinely matters for safety / correctness. DOCUMENTATION /
    INTERFACE / ENVIRONMENT MISSING are warnings, not critical.
    """
    if applicability != Applicability.APPLICABLE:
        return False
    if status == CoverageStatus.COVERED:
        return False
    if status == CoverageStatus.CONFLICT:
        # Real CONFLICT (after same-aspect validation) is always critical.
        return True
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
