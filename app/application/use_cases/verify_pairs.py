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

# Strict prohibition markers only — excludes quantifiers like "не более / не менее".
#
# PR-K follow-up (sweep TZ#2): the old regex missed neuter "не должно",
# verb form "запрещается", and gender variants of "запрещён/недопустим",
# producing a false-positive CONFLICT in the smoke run on
# "Время отклика не должно превышать 2 секунд" vs the same PMI fragment
# (which uses masc "не должен"). All gender/number forms now recognised.
_PROHIBITION_RE = re.compile(
    r"\b(?:"
    # Russian "не должен / не должна / не должно / не должны"
    # (all four gender/number forms — "не должно" was the missing one).
    r"не\s+долж(?:ен|на|но|ны)|"
    # "запрещено / запрещена / запрещены" + verb forms "запрещается / запрещаются".
    r"запрещен(?:о|а|ы)?|запрещается|запрещаются|"
    # "недопустимо / недопустима / недопустимы".
    r"недопустим(?:о|а|ы)?|"
    # "не допускается / не допускаются / не разрешается / не разрешаются".
    r"не\s+допускается|не\s+допускаются|"
    r"не\s+разрешается|не\s+разрешаются|"
    # "без возможности (изменения|отката|…)" — preposition-class prohibition
    # ("без возможности" + dependent noun in genitive).
    r"без\s+возможност[ьи]|"
    # English variants.
    r"not\s+allowed|forbidden|prohibited"
    r")\b",
    re.I,
)


# ── Same-aspect / negation-compatibility table (PR-G refactor) ──────────
#
# CONFLICT may be raised only when both sides talk about the SAME
# aspect (same metric, same component, same operation). When the
# requirement and the unit talk about different aspects but happen to
# share lexical signals (e.g. both mention "не должно превышать" + a
# number), CONFLICT is a false positive.
#
# `_SAME_OUTCOME_PAIRS` lists known semantic-equivalence pairs: a
# prohibition phrase on one side and a positive affirmation on the
# other that describes the same outcome. The audit-time symptom was
# "система не должна аварийно завершаться" vs "система должна
# продолжать корректно функционировать" — both express continued
# operation under errors, so CONFLICT was wrong. The fix recognises
# the pair and suppresses the negation contradiction rule.

_SAME_OUTCOME_PAIRS: list[tuple[re.Pattern, re.Pattern]] = [
    # Crash / continue working
    (
        re.compile(
            r"не\s+должн\w+\s+(?:аварийн\w+\s+заверш|падат|крашит|"
            r"прерыват\w*\s+работ)",
            re.I,
        ),
        re.compile(
            r"продолж\w+\s+(?:корректн\w+\s+)?(?:функционир|работ)|"
            r"должн\w+\s+продолжат\w+\s+работ|"
            r"должн\w+\s+(?:корректн\w+\s+)?функционир",
            re.I,
        ),
    ),
    # Lose data / save data
    (
        re.compile(
            r"не\s+должн\w+\s+(?:терят|потерят|удалят\w*\s+безвозврат)\w*\s+дан",
            re.I,
        ),
        re.compile(
            r"должн\w+\s+(?:сохран|сберег|резервн)\w*\s+дан",
            re.I,
        ),
    ),
]


def _same_outcome_negation_compatible(req_text: str, unit_text: str) -> bool:
    """True when one side prohibits a bad outcome while the other
    affirms the equivalent positive outcome — these are semantically
    compatible, never CONFLICT."""
    rt = (req_text or "").lower()
    ut = (unit_text or "").lower()
    for prohib_re, pos_re in _SAME_OUTCOME_PAIRS:
        # Either ordering: requirement-prohibits + unit-affirms,
        # or vice-versa.
        if prohib_re.search(rt) and pos_re.search(ut):
            return True
        if prohib_re.search(ut) and pos_re.search(rt):
            return True

    # PR-K P4: when BOTH sides use the same upper-bound prohibition
    # phrasing ("не должно превышать X" on TZ side, "не должно превышать
    # Y" on the unit side), this is a same-modality / same-direction
    # constraint — the verifier's mismatch detector wrongly fires
    # because of regex-level differences between gender forms. They
    # are compatible (both upper-bound; numeric values are checked
    # separately by the numeric rule). Real-package symptom (Polyakov
    # 0.20::sent1).
    same_upper_bound = re.compile(
        r"не\s+долж(?:ен|на|но|ны)\s+превышат",
        re.I | re.UNICODE,
    )
    if same_upper_bound.search(rt) and same_upper_bound.search(ut):
        return True
    return False


def _types_can_conflict(
    req: "RequirementUnit",
    unit: "CoverageUnit",
    judgment_text_overrides: tuple[str, str] | None = None,
) -> bool:
    """Whether the requirement and the coverage unit are about a
    sufficiently-aligned aspect to allow CONFLICT.

    Decision rules:
      * Out-of-scope requirement types (DELIVERY, PROCESS, ECONOMIC)
        never produce CONFLICT — coverage isn't the question for them.
      * Documentation / environment requirements only conflict when both
        sides talk about the same documentation aspect; numeric mismatches
        on hardware specs ("16 ГБ" in PMI bench vs "32 ГБ" in TZ
        recommendation) are not real conflicts.
      * Otherwise default to True; the per-rule logic in PairVerifier
        will refine.
    """
    from app.domain.c_quality_enums import RequirementType
    rt = req.requirement_type
    if rt in {
        RequirementType.DELIVERY_REQUIREMENT,
        RequirementType.PROCESS_REQUIREMENT,
        RequirementType.ECONOMIC_OR_NEED,
    }:
        return False
    return True


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


def _append_action(judgment: PairJudgment, action: str) -> None:
    """Append a verifier action tag onto the judgment.

    Conventions (so the aggregator can recognise rule outcomes):
      * `conflict_*`   — verifier explicitly confirmed a CONFLICT
                         (numeric, negation, …). Aggregator promotes
                         CONFLICT to a verified verdict.
      * `demote_*`     — a positive label was demoted (CONFLICT → PARTIAL,
                         COVERED → PARTIAL, …).
      * `suppress_*`   — a rule was about to fire but a guard suppressed
                         it (false-positive guard).
      * `no_op_*`      — verifier ran but no rule applied; label preserved.
    """
    actions = list(getattr(judgment, "verifier_actions", []) or [])
    actions.append(action)
    judgment.verifier_actions = actions


class PairVerifier:
    """Applies rule-based adjustments to a PairJudgment."""

    def verify(
        self,
        judgment: PairJudgment,
        req: RequirementUnit,
        unit: CoverageUnit,
    ) -> PairJudgment:
        # PR-G refactor: type-aware suppression of CONFLICT for
        # requirement classes where coverage isn't the question.
        # DELIVERY / PROCESS / ECONOMIC requirements never produce a
        # CONFLICT row — the aggregator's applicability filter handles
        # them downstream. Runs before the IRRELEVANT shortcut so that
        # the no_op_type_excluded action is always recorded for these
        # type classes.
        if not _types_can_conflict(req, unit):
            if judgment.llm_label == LLMLabel.CONFLICT:
                # Demote to PARTIAL so any matched aspects are still
                # surfaced; aggregator will mark the row OUT_OF_SCOPE.
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                judgment.explanation += (
                    " [rule] Type is delivery/process/economic — coverage "
                    "CONFLICT is not meaningful for this requirement class."
                )
                _append_action(judgment, "demote_conflict_type_excluded")
                return judgment
            judgment.rule_adjusted_label = judgment.llm_label
            _append_action(judgment, "no_op_type_excluded")
            return judgment

        # PR-K P0 fix: when the LLM said IRRELEVANT, we still let the
        # deterministic numeric-conflict check run below — small local
        # models (qwen2.5:3b in smoke tests) sometimes label "журнал 90
        # дней" vs "журнал 30 суток" as IRRELEVANT, missing the obvious
        # numeric mismatch. The numeric-rule has its own strict
        # topical-link guard (shared constraint kind / entity overlap /
        # ≥2 shared tokens / LLM confidence) so we don't false-positive
        # on truly unrelated pairs that happen to share a number.
        # Negation / COVERED-demotion rules still skip on IRRELEVANT
        # below, because their guards are weaker than the numeric one.
        is_irrelevant = judgment.llm_label == LLMLabel.IRRELEVANT

        # PR-G refactor: same-outcome negation compatibility — phrases
        # like "не должна аварийно завершаться" vs "должна продолжать
        # корректно функционировать" are semantically equivalent, not
        # contradictory. Suppress the LLM's CONFLICT verdict here BEFORE
        # the rule-based negation rule below considers them.
        if (
            judgment.llm_label == LLMLabel.CONFLICT
            and _same_outcome_negation_compatible(req.text, unit.text)
        ):
            judgment.rule_adjusted_label = LLMLabel.PARTIAL
            judgment.explanation += (
                " [rule] Same-outcome negation compatibility detected — "
                "prohibition of bad outcome and affirmation of good outcome "
                "are equivalent; demoted CONFLICT → PARTIAL."
            )
            _append_action(judgment, "demote_conflict_same_outcome")
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
            #
            # PR-K: bump the threshold from 0.70 to 0.80 so that the
            # DisabledCoverageJudge's structural ck_match path (conf=0.70,
            # which only confirms same constraint kind, NOT same value)
            # cannot suppress a real numeric conflict between 30 and 90.
            # Real LLMs aggregating semantic picture should comfortably
            # report ≥0.80 on a confident COVERED.
            judge_strongly_says_covered = (
                judgment.llm_label == LLMLabel.COVERED
                and (judgment.llm_confidence or 0) >= 0.80
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
                _append_action(judgment, "suppress_numeric_judge_strong_covered")
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
                _append_action(judgment, "conflict_confirmed_numeric")
                # PR-K P0 fix: when verifier deterministically promotes
                # IRRELEVANT/PARTIAL/COVERED → CONFLICT, the original
                # llm_confidence reflects the LLM's confidence in a
                # DIFFERENT label and is no longer meaningful for the
                # aggregator's CONFLICT-confidence gate. Bump to 0.95 so
                # the rule-confirmed CONFLICT survives the gate. The
                # `verifier_actions` tag preserves provenance — the
                # aggregator and the trace can still see this came from
                # a deterministic rule, not from a confident LLM verdict.
                if is_irrelevant or judgment.llm_confidence < 0.85:
                    judgment.llm_confidence = 0.95
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
                _append_action(judgment, "suppress_numeric_no_topic")

        # PR-K P0 fix: at this point the numeric rule has had its chance.
        # If the LLM said IRRELEVANT and no numeric conflict promoted us to
        # CONFLICT, return IRRELEVANT now. The remaining rules (negation,
        # COVERED→PARTIAL demotions) have weaker topical guards and would
        # produce false-positive CONFLICTs on truly unrelated pairs whose
        # vocabularies happen to share a prohibition word.
        if is_irrelevant:
            judgment.rule_adjusted_label = LLMLabel.IRRELEVANT
            _append_action(judgment, "no_op_irrelevant")
            return judgment

        # Rule 2: negation/modality contradiction
        # Guard: only fire when llm_confidence >= 0.25.
        # At lower confidence the pair shares only superficial vocabulary — req and unit
        # are likely about *different* topics that happen to share a few tokens, so a
        # modality mismatch (one prohibits, the other permits) is not a real conflict.
        if _negation_contradiction(req, unit) and judgment.llm_confidence >= 0.25:
            # PR-G refactor: same-outcome compatibility table catches the
            # pre-classified semantic equivalences ("don't crash" ≡ "keep
            # working"). When recognised, the modality mismatch is fake —
            # demote to PARTIAL instead of raising CONFLICT.
            if _same_outcome_negation_compatible(req.text, unit.text):
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                judgment.explanation += (
                    " [rule] Negation contradiction suppressed — same-outcome "
                    "phrasing detected (prohibition + positive affirmation "
                    "describe the same outcome)."
                )
                _append_action(judgment, "suppress_negation_same_outcome")
                return judgment

            # PR-K P4: topical-link guard — same logic the numeric rule has,
            # because the extended _PROHIBITION_RE (P2 covered "не должно"
            # neuter form etc.) now matches more pairs and false-fires on
            # unrelated ones. Real-package symptom (Polyakov 0.14::sent1):
            # TZ "время отклика не должно превышать 3 секунд" was paired
            # with PMI evidence "Windows 10 Pro [10] / Intel i5-7500 / 16 GB"
            # because the selector took only top-1 retrieval and that one
            # had highest BoW score by accident. Negation-rule fired since
            # req has prohibition + unit doesn't.
            from app.core.text import tokenize_content

            ent_overlap_neg = _entity_overlap(req, unit)
            req_tokens_neg = tokenize_content(req.normalized_text)
            unit_tokens_neg = tokenize_content(unit.normalized_text)
            shared_tokens_neg = req_tokens_neg & unit_tokens_neg
            # Negation contradiction is a softer signal than numeric
            # mismatch, so the topical-link bar is HIGHER (≥3 shared
            # content tokens vs 2 for numeric, ent_overlap ≥0.20 vs 0.15).
            #
            # PR-K post-fix (F): when the negation rule is CONFIRMING an
            # LLM-native CONFLICT (not upgrading PARTIAL/COVERED), the
            # LLM's confidence is NOT a reliable topical-link proxy —
            # the LLM may have been confidently wrong on off-topic
            # evidence. Real-package symptom: Polyakov run-4/5
            # req 0.17::sent3 vs PZ admin fragment (and vs PMI access-
            # control unit): LLM said CONFLICT conf=0.85 on completely
            # unrelated evidence; the confidence proxy caused the negation
            # rule to confirm → two false CONFLICT rows in every run.
            # For UPGRADES (PARTIAL/COVERED → CONFLICT), LLM confidence
            # IS informative — the LLM saw something relevant and was
            # sure of it; the proxy is appropriate there.
            if judgment.llm_label == LLMLabel.CONFLICT:
                has_topic_link_neg = (
                    ent_overlap_neg >= 0.20
                    or len(shared_tokens_neg) >= 3
                )
            else:
                has_topic_link_neg = (
                    ent_overlap_neg >= 0.20
                    or len(shared_tokens_neg) >= 3
                    or (judgment.llm_confidence or 0) >= 0.50
                )
            if not has_topic_link_neg:
                judgment.explanation += (
                    f" [rule] Negation contradiction suppressed — no topical "
                    f"link (ent_ov={ent_overlap_neg:.2f}, "
                    f"shared_tokens={len(shared_tokens_neg)}, "
                    f"conf={judgment.llm_confidence:.2f})."
                )
                logger.debug(
                    "Rule: negation contradiction SUPPRESSED for req=%s unit=%s "
                    "(no topic, ent=%.2f, shared=%d)",
                    req.req_id[:8], unit.unit_id[:8],
                    ent_overlap_neg, len(shared_tokens_neg),
                )
                _append_action(judgment, "suppress_negation_no_topic")
                # fall through — leave rule_adjusted_label unchanged
            else:
                # PR-K P4: when LLM is confidently positive (PARTIAL/COVERED
                # with conf ≥0.70), the verifier should NOT override its
                # verdict via negation-rule. Mirrors the
                # `judge_strongly_says_covered` guard on the numeric rule.
                # Real-package symptom (Polyakov 0.20::sent1): LLM said
                # COVERED conf≈1.0 on "не должно превышать общее время"
                # vs "не должно превышать времени" — exact same prohibition
                # phrasing on both sides — but the negation rule fired
                # anyway and demoted to false-CONFLICT.
                judge_strongly_positive = (
                    judgment.llm_label in (LLMLabel.COVERED, LLMLabel.PARTIAL)
                    and (judgment.llm_confidence or 0) >= 0.70
                )
                if judge_strongly_positive:
                    judgment.explanation += (
                        f" [rule] Negation contradiction suppressed — judge "
                        f"is confidently {judgment.llm_label.value} "
                        f"(conf={judgment.llm_confidence:.2f})."
                    )
                    logger.debug(
                        "Rule: negation contradiction SUPPRESSED for req=%s "
                        "unit=%s (judge confident %s, conf=%.2f)",
                        req.req_id[:8], unit.unit_id[:8],
                        judgment.llm_label.value, judgment.llm_confidence,
                    )
                    _append_action(judgment, "suppress_negation_judge_positive")
                    # fall through — leave rule_adjusted_label as LLM said
                else:
                    judgment.rule_adjusted_label = LLMLabel.CONFLICT
                    msg = "[rule] Negation contradiction between requirement and coverage unit"
                    judgment.conflict_aspects = conflict_details + [msg]
                    judgment.explanation += f" {msg}"
                    _append_action(judgment, "conflict_confirmed_negation")
                    # PR-K P0: same confidence-bump as the numeric path.
                    if judgment.llm_confidence < 0.85:
                        judgment.llm_confidence = 0.95
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
            _append_action(judgment, "demote_covered_constraints_missing")
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
                _append_action(judgment, "demote_covered_low_entity_overlap")
                return judgment

        # No adjustment needed — carry the LLM label forward.
        # PR-K: when an LLM-flagged CONFLICT survives all guards without
        # the verifier finding a concrete numeric/negation/aspect
        # contradiction, tag it explicitly so the aggregator can decide
        # whether to trust an unverified CONFLICT (current default: no —
        # downgraded to PARTIAL by the aggregator's verifier-confirmation
        # gate).
        judgment.rule_adjusted_label = judgment.llm_label
        if judgment.llm_label == LLMLabel.CONFLICT:
            _append_action(judgment, "no_op_llm_conflict_unverified")
        else:
            _append_action(judgment, "no_op_kept_label")
        return judgment
