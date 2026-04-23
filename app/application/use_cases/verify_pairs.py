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


_TIME_UNITS = {"days", "hours", "min", "sec", "ms"}
_SIZE_UNITS = {"kb", "mb", "gb", "tb"}
_RATE_UNITS = {"rps", "rpm"}

# Factors that convert a given unit into the canonical unit of its class.
# Canonicals: seconds for time, bytes for size, requests-per-second for rate.
# A value in unit U can be compared to a value in unit V (same class) by
# multiplying each by its factor, then comparing numerically.
_UNIT_TO_CANONICAL: dict = {
    # time → seconds
    "ms":   0.001,
    "sec":  1.0,
    "min":  60.0,
    "hours": 3600.0,
    "days": 86400.0,
    # size → bytes
    "kb": 1024.0,
    "mb": 1024.0 ** 2,
    "gb": 1024.0 ** 3,
    "tb": 1024.0 ** 4,
    # rate → rps
    "rps": 1.0,
    "rpm": 1.0 / 60.0,
}


def _canonical_value(value: float, unit: Optional[str]) -> Optional[float]:
    """
    Convert `value` from `unit` to the canonical unit of its class.

    Returns None if the unit has no canonical mapping (e.g. "%", None) —
    callers should then fall back to strict equality by unit string.
    """
    if not unit:
        return None
    factor = _UNIT_TO_CANONICAL.get(unit)
    if factor is None:
        return None
    return value * factor


def _same_unit_class(u1: Optional[str], u2: Optional[str]) -> bool:
    """Are two units in the same equivalence class?"""
    for cls in (_TIME_UNITS, _SIZE_UNITS, _RATE_UNITS):
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

    # Compare in a common unit when possible: "90 дней" vs "3 месяца" (if a
    # months→days converter existed) or "2 сек" vs "2000 мс" must NOT fire
    # a conflict. Fall back to strict value equality when either unit lies
    # outside the known canonical table.
    rc_canon = _canonical_value(rc.value, rc.unit)
    uc_canon = _canonical_value(uc.value, uc.unit)
    if rc_canon is not None and uc_canon is not None:
        # Tolerance: 0.1% of the larger value — handles rounding in
        # conversions like 60*60*24 days without false conflicts.
        tol = max(abs(rc_canon), abs(uc_canon)) * 1e-3
        if abs(rc_canon - uc_canon) <= max(tol, _VALUE_TOLERANCE):
            return None
        return (
            f"req[{rc.kind}]: {rc.operator}{rc.value} {rc.unit or ''} "
            f"(~{rc_canon:g} canon) | "
            f"unit[{uc.kind}]: {uc.operator}{uc.value} {uc.unit or ''} "
            f"(~{uc_canon:g} canon)"
        )

    # No canonical conversion available — require exact match on raw value.
    if abs(rc.value - uc.value) < _VALUE_TOLERANCE:
        return None

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
    # Compare entities by lemmatised word-bag so "Модуль аутентификации"
    # and "модулем аутентификации" (same concept, different surface form)
    # count as a match.
    from app.core.lemmatize import lemma

    def _key(entity: str) -> frozenset:
        return frozenset(lemma(w) for w in entity.lower().split() if w)

    req_set = {_key(e) for e in req.entities}
    unit_set = {_key(e) for e in unit.entities}
    # Remove trivial empty keys that can appear for degenerate inputs
    req_set.discard(frozenset())
    unit_set.discard(frozenset())
    if not req_set or not unit_set:
        return 0.0
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

        # Rule 1: numeric constraint conflict.
        #
        # Guard: require some topical-relatedness signal before we claim a
        # conflict. A req and a unit that happen to share `3 сек` as a
        # number, with zero entity overlap and almost no lexical overlap,
        # are simply two unrelated sentences — not a contradiction. Raising
        # CONFLICT on them was the biggest single source of false positives
        # in manual review (e.g. pkg_0005 #7, pkg_0008 #7).
        numeric_conflicts = _find_numeric_conflict(req.constraints, unit.constraints)
        if numeric_conflicts:
            from app.core.text import tokenize_content

            ent_overlap = _entity_overlap(req, unit)
            req_tokens = tokenize_content(req.normalized_text)
            unit_tokens = tokenize_content(unit.normalized_text)
            shared_tokens = req_tokens & unit_tokens
            # If both sides have a constraint with the same declared kind
            # (e.g. both `retention_period`), we already know they are
            # talking about the same aspect — that IS the topic link.
            shared_constraint_kinds = (
                {c.kind for c in req.constraints if c.kind and c.kind != "generic"}
                & {c.kind for c in unit.constraints if c.kind and c.kind != "generic"}
            )
            # Topical signal: same constraint kind OR some entity overlap
            # OR enough shared content tokens OR LLM confidence.
            has_topic_link = (
                bool(shared_constraint_kinds)
                or ent_overlap >= 0.15
                or len(shared_tokens) >= 2
                or (judgment.llm_confidence or 0) >= 0.4
            )

            # v3-review regression: the BGE judge confidently says COVERED on
            # pairs like "ОС Windows 7/10/11" vs "Windows 10" (10 is in the
            # allowed set). The numeric_conflicts rule then blindly fires
            # CONFLICT because 7 ≠ 10. The judge already aggregated the
            # semantic picture — if it is confidently COVERED, trust it and
            # suppress the rule override. 4/4 false CONFLICTs in manual
            # review came from this path.
            judge_strongly_says_covered = (
                judgment.llm_label == LLMLabel.COVERED
                and (judgment.llm_confidence or 0) >= 0.7
            )
            if judge_strongly_says_covered:
                judgment.explanation += (
                    f" [rule] Numeric value mismatch suppressed — judge is "
                    f"confidently COVERED (conf={judgment.llm_confidence:.2f}): "
                    f"{'; '.join(numeric_conflicts)}"
                )
                logger.debug(
                    "Rule: numeric mismatch SUPPRESSED (judge confident COVERED) "
                    "for req=%s unit=%s",
                    req.req_id[:8], unit.unit_id[:8],
                )
                # fall through to subsequent rules — skip the CONFLICT branch
                has_topic_link = False
            if has_topic_link:
                conflict_details.extend(numeric_conflicts)
                judgment.rule_adjusted_label = LLMLabel.CONFLICT
                judgment.conflict_aspects = conflict_details
                judgment.explanation += (
                    f" [rule] Numeric conflict (ent_ov={ent_overlap:.2f}, "
                    f"shared_tokens={len(shared_tokens)}): {'; '.join(numeric_conflicts)}"
                )
                logger.debug(
                    "Rule: numeric conflict for req=%s unit=%s (ent=%.2f, shared=%d)",
                    req.req_id[:8], unit.unit_id[:8], ent_overlap, len(shared_tokens),
                )
                return judgment
            else:
                judgment.explanation += (
                    f" [rule] Numeric value mismatch suppressed — no topical link "
                    f"(ent_ov={ent_overlap:.2f}, shared_tokens={len(shared_tokens)}, "
                    f"conf={judgment.llm_confidence:.2f}): {'; '.join(numeric_conflicts)}"
                )
                logger.debug(
                    "Rule: numeric mismatch SUPPRESSED for req=%s unit=%s (no topic)",
                    req.req_id[:8], unit.unit_id[:8],
                )

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
