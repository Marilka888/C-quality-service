"""
Stage 6: rule-based verifier applied after LLM judgment.

Rules (in priority order):
1. Numeric constraint conflict → override to CONFLICT
2. Negation/modality contradiction → override to CONFLICT
3. Req has constraints but unit has none → downgrade COVERED → PARTIAL
4. Entity mismatch (no shared entities) on "COVERED" → downgrade to PARTIAL

The verifier never upgrades a label; it can only keep or downgrade.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.core.logging import get_logger
from app.domain.c_quality_enums import LLMLabel, Modality
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)

logger = get_logger(__name__)

_VALUE_TOLERANCE = 1e-6

# Strict prohibition markers only — excludes quantifiers like "не более / не менее"
_PROHIBITION_RE = re.compile(
    r"\b(не должен|не должна|не должны|запрещено|недопустимо|не допускается|не разрешается|"
    r"not allowed|forbidden|prohibited)\b",
    re.I,
)


def _same_unit_class(u1: Optional[str], u2: Optional[str]) -> bool:
    """Are two units in the same equivalence class?"""
    _time_units = {"days", "hours", "min", "sec", "ms"}
    _size_units = {"kb", "mb", "gb", "tb"}
    _rate_units = {"rps", "rpm"}
    for cls in (_time_units, _size_units, _rate_units):
        if u1 in cls and u2 in cls:
            return True
    return u1 == u2


def _values_conflict(rc: Constraint, uc: Constraint) -> Optional[str]:
    """
    Return a conflict description if the two constraints are incompatible,
    None otherwise.

    Guards (in order):
    1. Must be in the same unit class (e.g. both time, both size).
    2. Unitless constraints (unit=None on both sides) are skipped — they are
       likely document numbers, years, section references, or IP fragments
       extracted by the constraint parser; comparing them cross-document is
       pure noise.
    3. If both constraints carry an explicit, specific kind and the kinds differ,
       they belong to different aspects and must not be compared
       (e.g. retention_period vs response_time both measured in seconds).
    4. Same value → no conflict.
    """
    if not _same_unit_class(rc.unit, uc.unit):
        return None  # different measurement class

    # Skip unitless pairs — too noisy (years, counts, section numbers, …)
    if rc.unit is None and uc.unit is None:
        return None

    # Different named aspects in the same unit class → not comparable
    if rc.kind and uc.kind and rc.kind != uc.kind:
        return None

    if abs(rc.value - uc.value) < _VALUE_TOLERANCE:
        return None  # same value → no conflict

    return (
        f"req[{rc.kind}]: {rc.operator}{rc.value} {rc.unit or ''} | "
        f"unit[{uc.kind}]: {uc.operator}{uc.value} {uc.unit or ''}"
    )


def _find_numeric_conflict(
    req_constraints: List[Constraint],
    unit_constraints: List[Constraint],
) -> List[str]:
    conflicts: List[str] = []
    for rc in req_constraints:
        for uc in unit_constraints:
            desc = _values_conflict(rc, uc)
            if desc:
                conflicts.append(desc)
    return conflicts


def _negation_contradiction(req: RequirementUnit, unit: CoverageUnit) -> bool:
    """
    True only when a requirement with explicit MUST_NOT modality is covered by
    a unit that makes a positive assertion (or vice-versa).
    Uses strict prohibition markers to avoid false positives from quantifiers
    like "не более" / "не менее".
    """
    req_prohibited = req.modality == Modality.MUST_NOT or bool(_PROHIBITION_RE.search(req.normalized_text))
    unit_prohibited = unit.modality == Modality.MUST_NOT if hasattr(unit, "modality") else bool(_PROHIBITION_RE.search(unit.normalized_text))
    # Only flag when one side is explicitly prohibitive and the other isn't
    return req_prohibited != unit_prohibited


def _entity_overlap(req: RequirementUnit, unit: CoverageUnit) -> float:
    if not req.entities or not unit.entities:
        return 0.0
    req_set = {e.lower() for e in req.entities}
    unit_set = {e.lower() for e in unit.entities}
    return len(req_set & unit_set) / len(req_set | unit_set)


class PairVerifier:
    """Applies rule-based adjustments to a PairJudgment."""

    def verify(
        self,
        judgment: PairJudgment,
        req: RequirementUnit,
        unit: CoverageUnit,
    ) -> PairJudgment:
        if judgment.llm_label == LLMLabel.IRRELEVANT:
            judgment.rule_adjusted_label = LLMLabel.IRRELEVANT
            return judgment

        conflict_details: List[str] = list(judgment.conflict_aspects)

        # Rule 1: numeric constraint conflict
        numeric_conflicts = _find_numeric_conflict(req.constraints, unit.constraints)
        if numeric_conflicts:
            conflict_details.extend(numeric_conflicts)
            judgment.rule_adjusted_label = LLMLabel.CONFLICT
            judgment.conflict_aspects = conflict_details
            judgment.explanation += f" [rule] Numeric conflict: {'; '.join(numeric_conflicts)}"
            logger.debug("Rule: numeric conflict for req=%s unit=%s", req.req_id[:8], unit.unit_id[:8])
            return judgment

        # Rule 2: negation/modality contradiction
        # Guard: only fire when llm_confidence >= 0.25.
        # At lower confidence the pair shares only superficial vocabulary — req and unit
        # are likely about *different* topics that happen to share a few tokens, so a
        # modality mismatch (one prohibits, the other permits) is not a real conflict.
        if _negation_contradiction(req, unit) and judgment.llm_confidence >= 0.25:
            judgment.rule_adjusted_label = LLMLabel.CONFLICT
            msg = "[rule] Negation contradiction between requirement and coverage unit"
            judgment.conflict_aspects = conflict_details + [msg]
            judgment.explanation += f" {msg}"
            return judgment

        # Rule 3: COVERED but req has constraints and unit has none → PARTIAL
        if (
            judgment.llm_label == LLMLabel.COVERED
            and req.constraints
            and not unit.constraints
        ):
            judgment.rule_adjusted_label = LLMLabel.PARTIAL
            judgment.missing_aspects = list(judgment.missing_aspects) + [
                f"Numeric constraint not verified: {c.value} {c.unit or ''}"
                for c in req.constraints
            ]
            judgment.explanation += " [rule] Required numeric constraints absent in coverage unit → PARTIAL"
            return judgment

        # Rule 4: COVERED but very low entity overlap when both have many entities → PARTIAL
        # Only applies when both sides have 3+ entities (single-entity docs produce noisy overlap)
        if (
            judgment.llm_label == LLMLabel.COVERED
            and len(req.entities) >= 3
            and len(unit.entities) >= 3
        ):
            overlap = _entity_overlap(req, unit)
            if overlap < 0.1:
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                judgment.explanation += f" [rule] Low entity overlap ({overlap:.2f}) with many entities → PARTIAL"
                return judgment

        # No adjustment needed — carry the LLM label forward
        judgment.rule_adjusted_label = judgment.llm_label
        return judgment
