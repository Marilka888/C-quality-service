"""
Stage 7: aggregate per (requirement, target_document) judgments into a
RequirementCoverageResult.

PR-K refactor: replace the old "max(rank(label))" strategy with an
evidence-based decision tree:

  * COVERED is accepted only when the winning judgment has llm_confidence
    >= covered_confidence_threshold AND grounding_passed AND its
    candidate's retrieval_score >= medium_retrieval_threshold.
  * CONFLICT requires conflict_confidence_threshold + grounding +
    medium retrieval + verifier confirmation (`verifier_actions`
    contains a "conflict_confirmed_*" entry).
  * Sub-statuses surface the reason the row didn't reach a confident
    verdict: MISSING_NO_EVIDENCE / MISSING_LOW_GROUNDING /
    MISSING_LOW_CONFIDENCE / OPTIONAL_NOT_FOUND.
  * Selection trace, retrieval breakdown and judge / verifier metadata
    are bundled into evidence_trace when CoverageDebugConfig.enabled.

Priority rank (CONFLICT > COVERED > PARTIAL > MISSING) is preserved on
the wire so legacy readers keep working.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.application.use_cases.applicability import (
    applicability_for,
    severity_for,
    should_affect_critical,
    should_affect_grade,
)
from app.core.config import CoverageAggregatorConfig, CoverageDebugConfig
from app.domain.c_quality_enums import (
    Applicability,
    CoverageRequirementLevel,
    CoverageStatus,
    EvidenceStrength,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    CoverageUnit,
    EvidenceItem,
    PairJudgment,
    RequirementCoverageResult,
    RequirementUnit,
    RetrievedCandidate,
)


def _label_to_status(label: LLMLabel) -> CoverageStatus:
    return {
        LLMLabel.COVERED: CoverageStatus.COVERED,
        LLMLabel.PARTIAL: CoverageStatus.PARTIAL,
        LLMLabel.CONFLICT: CoverageStatus.CONFLICT,
        LLMLabel.IRRELEVANT: CoverageStatus.MISSING,
        LLMLabel.NOT_JUDGED: CoverageStatus.UNKNOWN,
    }[label]


_STATUS_RANK = {
    CoverageStatus.CONFLICT: 4,
    CoverageStatus.COVERED: 3,
    CoverageStatus.PARTIAL: 2,
    CoverageStatus.MISSING: 1,
}


# ── Sub-status codes (PR-K) ───────────────────────────────────────────
SUBCODE_MISSING_NO_EVIDENCE = "MISSING_NO_EVIDENCE"
SUBCODE_MISSING_LOW_GROUNDING = "MISSING_LOW_GROUNDING"
SUBCODE_MISSING_LOW_CONFIDENCE = "MISSING_LOW_CONFIDENCE"
SUBCODE_OPTIONAL_NOT_FOUND = "OPTIONAL_NOT_FOUND"
SUBCODE_NOT_APPLICABLE = "NOT_APPLICABLE"
SUBCODE_OUT_OF_SCOPE = "OUT_OF_SCOPE"
SUBCODE_COVERED = "COVERED"
SUBCODE_PARTIAL = "PARTIAL"
SUBCODE_CONFLICT_VERIFIED = "CONFLICT_VERIFIED"
SUBCODE_CONFLICT_UNVERIFIED = "CONFLICT_UNVERIFIED"
# Polyakov-regression (2026-05-10): runtime LLM-failure rows. The
# aggregator surfaces these as CoverageStatus.UNKNOWN (not MISSING)
# so an Ollama timeout doesn't morph into a CRITICAL package status.
SUBCODE_UNKNOWN_LLM_UNAVAILABLE = "UNKNOWN_LLM_UNAVAILABLE"
# Polyakov-regression Step 6 (2026-05-11): when a row would be
# COVERED but the judge itself listed uncovered aspects, surface as
# PARTIAL with this subcode. Internally consistent: never claim full
# coverage when something is still uncovered.
SUBCODE_PARTIAL_DOWNGRADED_FROM_COVERED = "PARTIAL_DOWNGRADED_FROM_COVERED"


# Polyakov-regression Step 9: tag-shaped noise that leaks into
# uncovered_aspects from per-rule verifier flags. These are
# extraction-internal labels, not real missing requirements aspects.
# Drop them from the surface list so the reviewer sees only honest
# domain phrases.
_ASPECT_NOISE_TAG_RE = re.compile(
    r"^(?:"
    r"specific_object_match|verb_match|verb_object_coverage|"
    r"sufficient_lexical_density|object_phrase_match|"
    r"low_entity_overlap|near_zero_entity_overlap|"
    r"shared_substantive_token"
    r")$",
    re.IGNORECASE,
)
# Latin-transliteration noise («регISTRATION» when LLM started typing
# the cyrillic word in english). Pattern: a word containing BOTH
# Cyrillic letters AND uppercase Latin letters in the same token.
_MIXED_SCRIPT_NOISE_RE = re.compile(
    r"[А-Яа-яЁё][A-Z]|[A-Z][А-Яа-яЁё]",
    re.UNICODE,
)


def _is_mixed_script_noise(s: str) -> bool:
    """True when the string contains a Cyrillic→uppercase-Latin (or
    reverse) immediate transition — a signature of LLM mid-word
    transliteration glitch («регISTRATION», «парольPASSWORD»). Such
    tokens are extraction artefacts and must be dropped from the
    surface aspect list."""
    return bool(_MIXED_SCRIPT_NOISE_RE.search(s or ""))


def _normalize_uncovered_aspects(raws: List[str]) -> List[str]:
    """Polyakov-regression Step 9: dedup uncovered_aspects with light
    normalization. The LLM judge frequently emits:
      * progressive substrings («загрузка» / «загрузка файлов» /
        «загрузка файлов в и из системы»);
      * latin-transliteration noise («регISTRATION»);
      * extraction-internal tags («specific_object_match»,
        «verb_match», «sufficient_lexical_density») leaked from
        verifier per-rule flags.

    Strategy:
      1. Strip whitespace + trailing punctuation; lowercase for compare.
      2. Drop empty / pure-noise tags via `_ASPECT_NOISE_TAG_RE`.
      3. Drop strict substrings: if phrase A is a token-substring of
         phrase B (B carries strictly more information), keep only B.
      4. Preserve insertion order on the kept set.
    """
    if not raws:
        return []
    cleaned: List[Tuple[str, str]] = []  # (lower_norm, original)
    seen: Set[str] = set()
    for raw in raws:
        if not raw:
            continue
        s = re.sub(r"^[\s\-—–.,;:()«»\"']+|[\s\-—–.,;:()«»\"']+$", "", str(raw))
        if not s or len(s) < 3:
            continue
        if _ASPECT_NOISE_TAG_RE.match(s):
            continue
        if _is_mixed_script_noise(s):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((key, s))
    if not cleaned:
        return []
    # Drop strict substrings: phrase A whose lowercase form is a
    # contiguous substring of another phrase B's lowercase form.
    keep: List[Tuple[str, str]] = []
    for i, (key_i, orig_i) in enumerate(cleaned):
        is_substring_of_other = False
        for j, (key_j, _) in enumerate(cleaned):
            if i == j:
                continue
            if key_i != key_j and key_i in key_j and len(key_j) > len(key_i):
                is_substring_of_other = True
                break
        if not is_substring_of_other:
            keep.append((key_i, orig_i))
    return [orig for _, orig in keep]


# Default Russian rationales for MISSING when no judgment yields a usable
# explanation. Keeps the UI from rendering an empty rationale box next to
# a "Не покрыто" badge — reviewers want to know WHY the row is missing.
_DEFAULT_RATIONALE_MISSING_LOWCONF = (
    "Не найдено достаточно сильных совпадений в целевом документе: "
    "максимальная оценка retrieval ниже порога доверия. Возможно, "
    "соответствующее покрытие отсутствует или сформулировано вне "
    "ожидаемой терминологии."
)
_DEFAULT_RATIONALE_MISSING = (
    "Не найдено соответствующих фрагментов в целевом документе. "
    "Требование, вероятно, не покрыто."
)
_DEFAULT_RATIONALE_NOT_APPLICABLE = (
    "Требование не подлежит проверке покрытия в данном целевом документе "
    "(тип требования не соответствует роли документа)."
)
_DEFAULT_RATIONALE_OUT_OF_SCOPE = (
    "Требование не относится к области покрытия (поставка / процесс / "
    "потребности) и не требует проверки в целевых документах."
)
_DEFAULT_RATIONALE_OPTIONAL = (
    "Не найдено покрытия для опционального требования. "
    "Отсутствие не считается критичным."
)
_DEFAULT_RATIONALE_UNKNOWN = (
    "LLM-судья не смог вынести вердикт по этой паре (таймаут / ошибка "
    "соединения / некорректный ответ). Покрытие не оценено — статус "
    "UNKNOWN. Строка не учитывается ни в criticalCount, ни в C-grade. "
    "Перезапустите C-quality после восстановления LLM-сервиса."
)


def _pick_rationale(
    judgments: List[PairJudgment],
    best_status: CoverageStatus,
    low_confidence: bool,
    applicability: Applicability,
    coverage_requirement_level: CoverageRequirementLevel,
) -> str:
    """Choose the rationale string that best explains `best_status`.

    Priority:
      1. Explanation of a judgment whose `rule_adjusted_label` maps to
         `best_status` (the "winning" judgment).
      2. Explanation of any judgment that has non-empty text — caller-
         friendly fallback when no winner has a useful message.
      3. Default Russian text — varies by applicability / level / status.
    """
    target_label = {
        CoverageStatus.COVERED: LLMLabel.COVERED,
        CoverageStatus.PARTIAL: LLMLabel.PARTIAL,
        CoverageStatus.CONFLICT: LLMLabel.CONFLICT,
        CoverageStatus.MISSING: LLMLabel.IRRELEVANT,
    }.get(best_status)

    for j in judgments:
        if (
            target_label is not None
            and j.rule_adjusted_label == target_label
            and j.explanation
        ):
            return j.explanation

    for j in judgments:
        if j.explanation:
            return j.explanation

    if applicability == Applicability.OUT_OF_SCOPE:
        return _DEFAULT_RATIONALE_OUT_OF_SCOPE
    if applicability == Applicability.NOT_APPLICABLE:
        return _DEFAULT_RATIONALE_NOT_APPLICABLE

    if best_status == CoverageStatus.MISSING:
        # low_confidence (= retrieval was below the floor) takes priority
        # over the OPTIONAL fallback: the reviewer needs to know the row
        # is dim because retrieval was poor, not because the requirement
        # is optional.
        if low_confidence:
            return _DEFAULT_RATIONALE_MISSING_LOWCONF
        if coverage_requirement_level == CoverageRequirementLevel.OPTIONAL:
            return _DEFAULT_RATIONALE_OPTIONAL
        return _DEFAULT_RATIONALE_MISSING
    if best_status == CoverageStatus.UNKNOWN:
        return _DEFAULT_RATIONALE_UNKNOWN
    return ""


def _verifier_confirmed_conflict(judgment: PairJudgment) -> bool:
    """A judgment carries a verifier-confirmed CONFLICT when its
    `verifier_actions` contains an entry that begins with "conflict_"
    (e.g. "conflict_confirmed_numeric", "conflict_confirmed_aspect").
    Falls back to True when no `verifier_actions` field is set, so
    that legacy judgments produced before PR-K still produce CONFLICT
    rows when the rule-adjusted label says so. New judgments always
    populate the field, so the fallback is safe."""
    actions = getattr(judgment, "verifier_actions", None)
    if not actions:
        return True  # legacy / not populated → trust label
    for a in actions:
        a = (a or "").strip().lower()
        if a.startswith("conflict_"):
            return True
    return False


def _build_evidence_trace(
    requirement: RequirementUnit,
    candidates_by_unit_id: Dict[str, RetrievedCandidate],
    selection_result: Optional[Any],
    judgments: List[PairJudgment],
    decision_log: List[str],
    debug_cfg: CoverageDebugConfig,
) -> Dict[str, Any]:
    """Assemble the evidence_trace dict for the UI / debug reader.

    Capped at `debug_cfg.max_candidates`. When `include_discarded` is
    False, only candidates that were either selected for LLM or actually
    judged appear in the trace. The decision log lists the aggregator's
    reasoning steps so a researcher can see why a row landed where it did.
    """
    trace: Dict[str, Any] = {
        "requirement": {
            "req_id": requirement.req_id,
            "type": (requirement.requirement_type or RequirementType.OTHER).value,
            "constraints": [
                {"kind": c.kind, "operator": c.operator, "value": c.value, "unit": c.unit}
                for c in (requirement.constraints or [])
            ],
        },
        "selection": None,
        "candidates": [],
        "decision_log": list(decision_log),
    }
    if selection_result is not None:
        trace["selection"] = {
            "selected_k": getattr(selection_result, "selected_k", 0),
            "skip_llm": bool(getattr(selection_result, "skip_llm", False)),
            "skip_reason": getattr(selection_result, "skip_reason", "") or "",
            "selection_reason": getattr(selection_result, "selection_reason", "") or "",
        }

    judgments_by_unit = {j.unit_id: j for j in judgments}
    seen_unit_ids: List[str] = []
    # Order: selected first (so the "main" evidence is on top), then any
    # judged-but-not-selected (shouldn't happen in PR-K but defensive),
    # then discarded if include_discarded.
    if selection_result is not None:
        for c in getattr(selection_result, "selected", []) or []:
            seen_unit_ids.append(c.unit_id)
        if debug_cfg.include_discarded:
            for c in getattr(selection_result, "discarded", []) or []:
                if c.unit_id not in seen_unit_ids:
                    seen_unit_ids.append(c.unit_id)
    else:
        # No selection_result was passed in — fall back to candidates_by_unit_id.
        seen_unit_ids = list(candidates_by_unit_id.keys())

    cap = max(1, int(debug_cfg.max_candidates or 5))
    for unit_id in seen_unit_ids[:cap]:
        c = candidates_by_unit_id.get(unit_id)
        if c is None:
            continue
        j = judgments_by_unit.get(unit_id)
        item: Dict[str, Any] = {
            "unit_id": unit_id,
            "retrieval_score": c.retrieval_score,
            "lexical_score": c.lexical_score,
            "semantic_score": c.semantic_score,
            "constraint_overlap_score": c.constraint_overlap_score,
            "section_prior_score": c.section_prior_score,
            "score_reason": c.score_reason or "",
            "evidence_strength": (
                c.evidence_strength.value
                if c.evidence_strength
                else EvidenceStrength.NO_EVIDENCE.value
            ),
            "selected_for_llm": bool(c.selected_for_llm),
            "reranker_used": bool(c.reranker_used),
            "reranker_score": c.reranker_score,
        }
        if j is not None:
            item["judge_label"] = j.rule_adjusted_label.value
            item["judge_confidence"] = float(j.llm_confidence or 0.0)
            item["llm_label"] = j.llm_label.value
            item["matched_aspects"] = list(j.matched_aspects or [])
            item["missing_aspects"] = list(j.missing_aspects or [])
            item["conflict_aspects"] = list(j.conflict_aspects or [])
            item["cited_phrases"] = list(j.cited_phrases or [])
            item["verifier_actions"] = list(getattr(j, "verifier_actions", []) or [])
            # PR-K P0: grounding_passed reflects ONLY whether cited_phrases
            # were substring-matched in evidence (true grounding gate).
            # `low_confidence` separately can be set by below-evidence-floor
            # retrieval — that's a retrieval-quality flag, not a grounding flag.
            item["grounding_passed"] = not bool(getattr(j, "grounding_failed", False))
            item["below_evidence_floor"] = bool(
                getattr(j, "low_confidence", False)
                and not getattr(j, "grounding_failed", False)
            )
        trace["candidates"].append(item)

    return trace


class CoverageAggregator:
    def aggregate(
        self,
        requirement: RequirementUnit,
        judgments: List[PairJudgment],
        candidates_by_unit_id: Dict[str, RetrievedCandidate],
        units_by_id: Dict[str, CoverageUnit],
        target_document_id: str,
        target_doc_role: str,
        # ── PR-K additive parameters ────────────────────────────────
        selection_result: Optional[Any] = None,
        coverage_requirement_level: Optional[CoverageRequirementLevel] = None,
        debug_cfg: Optional[CoverageDebugConfig] = None,
        aggregator_cfg: Optional[CoverageAggregatorConfig] = None,
    ) -> RequirementCoverageResult:
        # Defaults so legacy callers don't have to pass these.
        debug_cfg = debug_cfg or CoverageDebugConfig()
        aggregator_cfg = aggregator_cfg or CoverageAggregatorConfig()

        # P0 #9 (Хамроев) — applicability is the FIRST decision in
        # aggregate(). NOT_APPLICABLE / OUT_OF_SCOPE rows must never be
        # overwritten by MISSING_NO_EVIDENCE just because the upstream
        # pipeline produced an empty shortlist. Putting the
        # applicability check before any snapshot / branch-B logic
        # makes the ordering explicit and pins it against future
        # regression: even a caller that bypasses the pipeline gate
        # (run_coverage_analysis._handle_one_requirement) and invokes
        # aggregate() directly with empty judgments still gets the
        # correct OUT_OF_SCOPE row.
        req_type = requirement.requirement_type or RequirementType.OTHER
        applicability = applicability_for(req_type, target_doc_role)

        # BUG-12: snapshot the requirement's source-side context once so
        # both the empty-shortlist branch and the regular branch return
        # the same locator fields.
        req_text = requirement.text
        meta = requirement.metadata or {}
        req_section_title = meta.get("section_title") or meta.get("sectionTitle") or None
        req_section_id = requirement.source_section_id
        req_number = (
            meta.get("number")
            or meta.get("sectionNumber")
            or meta.get("source_number")
            or None
        )

        if coverage_requirement_level is None:
            # Derive from applicability if caller didn't pass it (legacy path).
            from app.application.use_cases.applicability import (
                coverage_requirement_level_for,
            )
            coverage_requirement_level = coverage_requirement_level_for(
                req_type, target_doc_role,
            )

        decision_log: List[str] = []

        # ── Branch A: NOT_APPLICABLE / OUT_OF_SCOPE row ──────────────
        if applicability != Applicability.APPLICABLE:
            status = CoverageStatus.MISSING
            subcode = (
                SUBCODE_NOT_APPLICABLE
                if applicability == Applicability.NOT_APPLICABLE
                else SUBCODE_OUT_OF_SCOPE
            )
            decision_log.append(
                f"applicability={applicability.value}; row excluded from "
                f"coverage check in target role '{target_doc_role}'."
            )
            agg_reason = (
                f"Requirement type {req_type.value} is "
                f"{applicability.value} for target role '{target_doc_role}'; "
                f"no LLM call performed."
            )
            return RequirementCoverageResult(
                req_id=requirement.req_id,
                source_document_id=requirement.source_document_id,
                target_document_id=target_document_id,
                target_doc_role=target_doc_role,
                status=status,
                req_text=req_text,
                req_section_title=req_section_title,
                req_section_id=req_section_id,
                req_number=req_number,
                rationale=_pick_rationale(
                    [], status, False, applicability, coverage_requirement_level,
                ),
                requirement_type=req_type,
                applicability=applicability,
                severity=severity_for(req_type, target_doc_role, status, applicability, subcode),
                should_affect_critical=should_affect_critical(
                    req_type, applicability, status, target_doc_role,
                ),
                should_affect_grade=should_affect_grade(req_type, applicability),
                status_subcode=subcode,
                winning_candidate_id=None,
                final_confidence=0.0,
                aggregation_reason=agg_reason,
                coverage_requirement_level=coverage_requirement_level,
                evidence_trace=(
                    _build_evidence_trace(
                        requirement, candidates_by_unit_id, selection_result,
                        [], decision_log, debug_cfg,
                    )
                    if debug_cfg.enabled
                    else None
                ),
            )

        # ── Branch B: no judgments produced (skip_llm or empty shortlist) ─
        if not judgments:
            status = CoverageStatus.MISSING
            # Distinguish OPTIONAL_NOT_FOUND from a genuine no-evidence row.
            if coverage_requirement_level == CoverageRequirementLevel.OPTIONAL:
                subcode = SUBCODE_OPTIONAL_NOT_FOUND
                decision_log.append(
                    "Optional requirement: no LLM call performed, no candidate "
                    "selected; scored as OPTIONAL_NOT_FOUND."
                )
            else:
                subcode = SUBCODE_MISSING_NO_EVIDENCE
                decision_log.append(
                    "No judgments produced (empty shortlist or LLM skipped "
                    "by selector); marked MISSING_NO_EVIDENCE."
                )

            skip_reason = (
                getattr(selection_result, "skip_reason", "") if selection_result else ""
            )
            sel_reason = (
                getattr(selection_result, "selection_reason", "") if selection_result else ""
            )
            agg_reason = (
                f"No judgments to aggregate: {sel_reason or skip_reason or 'empty shortlist'}."
            )

            return RequirementCoverageResult(
                req_id=requirement.req_id,
                source_document_id=requirement.source_document_id,
                target_document_id=target_document_id,
                target_doc_role=target_doc_role,
                status=status,
                req_text=req_text,
                req_section_title=req_section_title,
                req_section_id=req_section_id,
                req_number=req_number,
                rationale=_pick_rationale(
                    [], status, False, applicability, coverage_requirement_level,
                ),
                requirement_type=req_type,
                applicability=applicability,
                severity=severity_for(req_type, target_doc_role, status, applicability, subcode),
                should_affect_critical=should_affect_critical(
                    req_type,
                    applicability,
                    # OPTIONAL_NOT_FOUND must not contribute to criticalCount.
                    status if coverage_requirement_level == CoverageRequirementLevel.REQUIRED
                    else CoverageStatus.COVERED,
                    target_doc_role,
                ),
                should_affect_grade=should_affect_grade(req_type, applicability),
                status_subcode=subcode,
                winning_candidate_id=None,
                final_confidence=0.0,
                aggregation_reason=agg_reason,
                coverage_requirement_level=coverage_requirement_level,
                evidence_trace=(
                    _build_evidence_trace(
                        requirement, candidates_by_unit_id, selection_result,
                        [], decision_log, debug_cfg,
                    )
                    if debug_cfg.enabled
                    else None
                ),
            )

        # ── Branch B': all judgments are runtime-failure sentinels ──
        # Polyakov-regression (2026-05-10): when the LLM judge
        # backend errored at request time (timeout / connection /
        # HTTP / parse-exhausted / unexpected exception), the wrappers
        # return `make_unknown_judgment(...)` — a sentinel
        # PairJudgment with llm_label=NOT_JUDGED. If EVERY judgment
        # for this requirement is such a sentinel, the pair was simply
        # never assessed; surfacing it as MISSING (the old behaviour)
        # converts an infrastructure failure into a documentation
        # defect on the report and inflates criticalCount. UNKNOWN is
        # the explicit "we couldn't judge this pair" status; it does
        # not contribute to criticalCount and is excluded from the
        # C-grade denominator (should_affect_grade=False below).
        # NOTE: when only SOME judgments are sentinels, Branch C
        # below filters them out and aggregates the real ones — the
        # sentinel doesn't poison the verdict.
        from app.infrastructure.llm.coverage_judge import is_unknown_judgment
        unknown_judgments = [j for j in judgments if is_unknown_judgment(j)]
        if unknown_judgments and len(unknown_judgments) == len(judgments):
            status = CoverageStatus.UNKNOWN
            # First sentinel's reason tag tells the reviewer WHY
            # (timeout / HTTP / …); use the first explanation as the
            # row-level rationale so the UI tooltip is specific.
            first_sentinel = unknown_judgments[0]
            decision_log.append(
                f"All {len(judgments)} judgment(s) are LLM-unavailable "
                f"sentinels; row reported as UNKNOWN."
            )
            agg_reason = (
                f"All {len(judgments)} candidate pair(s) were not judged "
                f"due to LLM backend unavailability "
                f"({(first_sentinel.verifier_actions or ['unknown'])[0]}). "
                f"Status UNKNOWN: row excluded from criticalCount and "
                f"C-grade denominator."
            )
            return RequirementCoverageResult(
                req_id=requirement.req_id,
                source_document_id=requirement.source_document_id,
                target_document_id=target_document_id,
                target_doc_role=target_doc_role,
                status=status,
                req_text=req_text,
                req_section_title=req_section_title,
                req_section_id=req_section_id,
                req_number=req_number,
                rationale=(
                    first_sentinel.explanation
                    or _DEFAULT_RATIONALE_UNKNOWN
                ),
                requirement_type=req_type,
                applicability=applicability,
                # UNKNOWN is "we don't know" — surface as low priority
                # so it doesn't dominate the UI sort order.
                severity="low",
                # Infrastructure failure must NOT inflate criticalCount.
                should_affect_critical=False,
                # And must NOT be in the C-grade assessable denominator —
                # otherwise an Ollama timeout drags grade down on a
                # perfectly-fine package.
                should_affect_grade=False,
                status_subcode=SUBCODE_UNKNOWN_LLM_UNAVAILABLE,
                # Polyakov-regression Step 4: all judgments were
                # sentinels — the count IS the row's pair count.
                unjudged_pair_count=len(judgments),
                winning_candidate_id=None,
                final_confidence=0.0,
                aggregation_reason=agg_reason,
                coverage_requirement_level=coverage_requirement_level,
                evidence_trace=(
                    _build_evidence_trace(
                        requirement, candidates_by_unit_id, selection_result,
                        judgments, decision_log, debug_cfg,
                    )
                    if debug_cfg.enabled
                    else None
                ),
            )

        # If SOME judgments are sentinels, drop them — Branch C
        # aggregates only judgments that actually carry a verdict.
        # Polyakov-regression Step 4 (2026-05-11): track the count of
        # filtered sentinels so the row's visibility flag
        # (`unjudged_pair_count`) reflects partial-shortlist runs.
        unjudged_pair_count_for_row = len(unknown_judgments)
        if unknown_judgments:
            judgments = [j for j in judgments if not is_unknown_judgment(j)]
            decision_log.append(
                f"Filtered {len(unknown_judgments)} LLM-unavailable "
                f"sentinel(s) before aggregation; {len(judgments)} "
                f"real judgment(s) remain."
            )

        # ── Branch C: judgments exist — evidence-based decision ──────
        any_low_conf = any(getattr(j, "low_confidence", False) for j in judgments)
        # P0 #10: row-level grounding flag mirrors the per-judgment
        # grounding_failed signal. True when ANY judgment that could
        # have driven the verdict had its citations rejected by the
        # substring-grounding gate (LLM hallucinated its quotes). The
        # docback mapper exposes this as a UI badge separate from
        # low_confidence (which also covers below-evidence-floor
        # retrieval, a retrieval-quality flag rather than LLM honesty).
        any_grounding_failed = any(
            getattr(j, "grounding_failed", False) for j in judgments
        )

        # Build EvidenceItem list (used by the wire payload).
        evidence_items: List[EvidenceItem] = []
        uncovered: List[str] = []
        conflicts: List[str] = []
        for j in judgments:
            unit = units_by_id.get(j.unit_id)
            candidate = candidates_by_unit_id.get(j.unit_id)
            if unit is not None:
                # Polyakov-regression: the wire-truncation budget for
                # evidence text was a hard `[:300]` cut that landed
                # mid-word («жизненн» → user-visible report). The UI
                # «Требования» tab renders this string verbatim, so
                # mid-word cuts read as garbled. Use the same
                # sentence-boundary truncator that build_coverage_units
                # applies to SECTION_WINDOW units, and lift the budget
                # to 600 chars so a typical PMI restatement paragraph
                # fits without truncation at all.
                from app.application.use_cases.build_coverage_units import (
                    _truncate_at_sentence_boundary,
                )
                evidence_items.append(
                    EvidenceItem(
                        unit_id=j.unit_id,
                        fragment_id=unit.fragment_id,
                        section_id=unit.section_id,
                        text=_truncate_at_sentence_boundary(unit.text, 600),
                        retrieval_score=candidate.retrieval_score if candidate else 0.0,
                        judgment=j,
                    )
                )
            uncovered.extend(j.missing_aspects)
            conflicts.extend(j.conflict_aspects)

        # Decision: scan judgments by rank, but accept the verdict only
        # when confidence + grounding + retrieval_score thresholds line up.
        cov_thr = aggregator_cfg.covered_confidence_threshold
        cnf_thr = aggregator_cfg.conflict_confidence_threshold
        par_thr = aggregator_cfg.partial_confidence_threshold
        med_thr = aggregator_cfg.medium_retrieval_threshold

        # Polyakov-regression (2026-05-10): per-type medium-retrieval
        # floor relaxation for SECURITY / PERFORMANCE / RELIABILITY.
        # These types tend to use specialised vocabulary («атаки типа
        # Внедрение кода», «время отклика», «отказоустойчивость») that
        # rarely shares lexical mass with the surrounding PMI/PZ
        # narrative, so retrieval scores cap at ~0.20-0.30 even when
        # the LLM judge correctly identifies a partial-coverage
        # relationship (Polyakov 0.14::sent1: judge PARTIAL with
        # retrieval=0.28; 0.18::sent2: judge PARTIAL conf 0.7 with
        # retrieval=0.20). Both got rejected by the 0.30 floor and
        # surfaced as MISSING_LOW_CONFIDENCE, inflating criticalCount
        # for what the LLM read as legitimate partial coverage. Lower
        # the floor to 0.20 for these specialised types — the LLM
        # confidence + grounding gates still apply, so we're not
        # admitting hallucinations, just relaxing the lex-density
        # gate that hurts narrow-domain requirements.
        _RELAXED_FLOOR_TYPES = {
            RequirementType.SECURITY,
            RequirementType.PERFORMANCE,
            RequirementType.RELIABILITY,
        }
        if req_type in _RELAXED_FLOOR_TYPES:
            relaxed_med_thr = min(med_thr, 0.20)
            if relaxed_med_thr < med_thr:
                decision_log.append(
                    f"medium_retrieval_threshold relaxed for type "
                    f"{req_type.value}: {med_thr:.2f} → {relaxed_med_thr:.2f} "
                    f"(specialised-vocabulary type cap)."
                )
                med_thr = relaxed_med_thr

        # Sort judgments by status priority (CONFLICT > COVERED > PARTIAL >
        # MISSING) so we evaluate the strongest claim first.
        ordered = sorted(
            judgments,
            key=lambda j: (
                _STATUS_RANK[_label_to_status(j.rule_adjusted_label)],
                float(j.llm_confidence or 0.0),
            ),
            reverse=True,
        )

        chosen_status: CoverageStatus = CoverageStatus.MISSING
        chosen_subcode: str = SUBCODE_MISSING_NO_EVIDENCE
        winning: Optional[PairJudgment] = None
        winning_score: float = 0.0
        agg_reason: str = ""

        # PR-F follow-up: when a strong COVERED already exists, demote
        # weaker CONFLICTs (same logic as legacy aggregator). Computed
        # once here so the decision tree below can consult it.
        # Strong-COVERED detection: ignore both grounding-failed and
        # below-floor judgments to be safe — strong COVERED should mean
        # actually-trustworthy COVERED.
        strong_covered_conf = max(
            (
                j.llm_confidence or 0.0
                for j in judgments
                if j.rule_adjusted_label == LLMLabel.COVERED
                and not getattr(j, "low_confidence", False)
                and not getattr(j, "grounding_failed", False)
            ),
            default=0.0,
        )
        has_strong_covered = strong_covered_conf >= cov_thr

        for j in ordered:
            label = j.rule_adjusted_label
            status = _label_to_status(label)
            cand = candidates_by_unit_id.get(j.unit_id)
            r_score = cand.retrieval_score if cand else 0.0
            conf = float(j.llm_confidence or 0.0)
            # PR-K P0: split grounding from below-floor semantics.
            # `grounding_failed` is the LLM-hallucination signal (cited
            # phrases not in evidence) — true grounding violation.
            # `low_confidence` may also mean retrieval was below floor —
            # that's a quality signal but does NOT make the judgment
            # ungrounded for the aggregator gate. The row will still
            # carry low_confidence=True for UI dimming, but COVERED
            # remains accepted when conf + retrieval pass.
            grounded = not bool(getattr(j, "grounding_failed", False))

            if status == CoverageStatus.CONFLICT:
                # Demote weak CONFLICT in presence of strong COVERED.
                if has_strong_covered and conf < strong_covered_conf:
                    decision_log.append(
                        f"CONFLICT pair unit={j.unit_id[:12]} demoted to PARTIAL: "
                        f"weaker than dominant COVERED (conf {conf:.2f} < "
                        f"{strong_covered_conf:.2f})."
                    )
                    continue  # try the next-rank judgment
                if conf < cnf_thr:
                    decision_log.append(
                        f"CONFLICT pair unit={j.unit_id[:12]} rejected: "
                        f"conf {conf:.2f} < {cnf_thr:.2f}."
                    )
                    continue
                if not grounded:
                    decision_log.append(
                        f"CONFLICT pair unit={j.unit_id[:12]} rejected: "
                        f"grounding failed."
                    )
                    continue
                if r_score < med_thr:
                    decision_log.append(
                        f"CONFLICT pair unit={j.unit_id[:12]} rejected: "
                        f"retrieval_score {r_score:.2f} < {med_thr:.2f}."
                    )
                    continue
                if not _verifier_confirmed_conflict(j):
                    # An unconfirmed CONFLICT means the LLM reported a
                    # contradiction but no deterministic rule could validate
                    # it. This is NOT sufficient evidence of partial
                    # coverage — the pair is often off-topic (e.g. LLM
                    # hallucinated a conflict on unrelated evidence). Setting
                    # PARTIAL here was the primary source of PARTIAL
                    # inflation in real packages (Поляков runs). Instead,
                    # record CONFLICT_UNVERIFIED as the MISSING subcode so
                    # reviewers can see why the row is flagged, and continue
                    # scanning for a genuine PARTIAL/COVERED verdict from
                    # other candidates.
                    decision_log.append(
                        f"CONFLICT pair unit={j.unit_id[:12]} not confirmed by verifier "
                        f"(verifier_actions={list(getattr(j, 'verifier_actions', []) or [])}); "
                        f"treating as MISSING(CONFLICT_UNVERIFIED), scanning for better verdict."
                    )
                    if chosen_status == CoverageStatus.MISSING:
                        chosen_subcode = SUBCODE_CONFLICT_UNVERIFIED
                        winning = j
                        winning_score = conf
                        agg_reason = (
                            f"CONFLICT not confirmed by verifier; row treated as "
                            f"MISSING (CONFLICT_UNVERIFIED). Pair may be off-topic "
                            f"(conf={conf:.2f}, retrieval={r_score:.2f}, "
                            f"verifier_actions={list(getattr(j, 'verifier_actions', []) or [])})."
                        )
                    continue
                # Confirmed CONFLICT.
                chosen_status = CoverageStatus.CONFLICT
                chosen_subcode = SUBCODE_CONFLICT_VERIFIED
                winning = j
                winning_score = conf
                agg_reason = (
                    f"CONFLICT verified: judge conf {conf:.2f}, "
                    f"retrieval {r_score:.2f}, verifier confirmed via "
                    f"{list(getattr(j, 'verifier_actions', []) or [])}."
                )
                decision_log.append(
                    f"CONFLICT accepted from unit={j.unit_id[:12]}: "
                    f"conf {conf:.2f} ≥ {cnf_thr:.2f}, retrieval {r_score:.2f} ≥ "
                    f"{med_thr:.2f}, grounded, verifier confirmed."
                )
                break

            if status == CoverageStatus.COVERED:
                if conf < cov_thr:
                    decision_log.append(
                        f"COVERED pair unit={j.unit_id[:12]} rejected: "
                        f"conf {conf:.2f} < {cov_thr:.2f}."
                    )
                    if chosen_status == CoverageStatus.MISSING:
                        chosen_subcode = SUBCODE_MISSING_LOW_CONFIDENCE
                        winning = j
                        winning_score = conf
                        agg_reason = (
                            f"COVERED rejected: judge confidence {conf:.2f} below "
                            f"threshold {cov_thr:.2f}; reported as MISSING_LOW_CONFIDENCE."
                        )
                    continue
                if not grounded:
                    decision_log.append(
                        f"COVERED pair unit={j.unit_id[:12]} rejected: "
                        f"grounding failed."
                    )
                    if chosen_status == CoverageStatus.MISSING:
                        chosen_subcode = SUBCODE_MISSING_LOW_GROUNDING
                        winning = j
                        winning_score = conf
                        agg_reason = (
                            f"COVERED rejected: cited phrases not grounded in "
                            f"retrieved evidence; reported as MISSING_LOW_GROUNDING."
                        )
                    continue
                if r_score < med_thr:
                    decision_log.append(
                        f"COVERED pair unit={j.unit_id[:12]} rejected: "
                        f"retrieval_score {r_score:.2f} < {med_thr:.2f}."
                    )
                    if chosen_status == CoverageStatus.MISSING:
                        chosen_subcode = SUBCODE_MISSING_LOW_CONFIDENCE
                        winning = j
                        winning_score = conf
                        agg_reason = (
                            f"COVERED rejected: retrieval_score {r_score:.2f} below "
                            f"medium threshold {med_thr:.2f}; reported as "
                            f"MISSING_LOW_CONFIDENCE."
                        )
                    continue
                # Confirmed COVERED.
                chosen_status = CoverageStatus.COVERED
                chosen_subcode = SUBCODE_COVERED
                winning = j
                winning_score = conf
                agg_reason = (
                    f"COVERED accepted: judge conf {conf:.2f} ≥ {cov_thr:.2f}, "
                    f"retrieval {r_score:.2f} ≥ {med_thr:.2f}, grounded."
                )
                decision_log.append(
                    f"COVERED accepted from unit={j.unit_id[:12]}: "
                    f"conf {conf:.2f}, retrieval {r_score:.2f}, grounded."
                )
                break

            if status == CoverageStatus.PARTIAL:
                # PARTIAL has a softer bar — accept when confidence is at
                # least par_thr AND retrieval is at least medium AND
                # grounding holds. Otherwise demote to MISSING_*.
                if conf < par_thr:
                    if chosen_status == CoverageStatus.MISSING:
                        chosen_subcode = SUBCODE_MISSING_LOW_CONFIDENCE
                        winning = j
                        winning_score = conf
                        agg_reason = (
                            f"PARTIAL rejected: judge confidence {conf:.2f} below "
                            f"partial threshold {par_thr:.2f}; reported as "
                            f"MISSING_LOW_CONFIDENCE."
                        )
                    decision_log.append(
                        f"PARTIAL pair unit={j.unit_id[:12]} rejected: "
                        f"conf {conf:.2f} < par_thr {par_thr:.2f}."
                    )
                    continue
                if not grounded:
                    if chosen_status == CoverageStatus.MISSING:
                        chosen_subcode = SUBCODE_MISSING_LOW_GROUNDING
                        winning = j
                        winning_score = conf
                        agg_reason = (
                            "PARTIAL rejected: grounding failed; reported as "
                            "MISSING_LOW_GROUNDING."
                        )
                    decision_log.append(
                        f"PARTIAL pair unit={j.unit_id[:12]} rejected: "
                        f"grounding failed."
                    )
                    continue
                if r_score < med_thr:
                    if chosen_status == CoverageStatus.MISSING:
                        chosen_subcode = SUBCODE_MISSING_LOW_CONFIDENCE
                        winning = j
                        winning_score = conf
                        agg_reason = (
                            f"PARTIAL rejected: retrieval_score {r_score:.2f} below "
                            f"medium threshold {med_thr:.2f}; reported as "
                            f"MISSING_LOW_CONFIDENCE."
                        )
                    decision_log.append(
                        f"PARTIAL pair unit={j.unit_id[:12]} rejected: "
                        f"retrieval_score {r_score:.2f} < {med_thr:.2f}."
                    )
                    continue
                chosen_status = CoverageStatus.PARTIAL
                chosen_subcode = SUBCODE_PARTIAL
                winning = j
                winning_score = conf
                agg_reason = (
                    f"PARTIAL accepted: judge conf {conf:.2f}, "
                    f"retrieval {r_score:.2f} ≥ {med_thr:.2f}, grounded."
                )
                decision_log.append(
                    f"PARTIAL accepted from unit={j.unit_id[:12]}: "
                    f"conf {conf:.2f}, retrieval {r_score:.2f}."
                )
                break

            # IRRELEVANT/MISSING — keep MISSING and continue scanning,
            # but capture confidence so final_confidence reflects the
            # highest-conf irrelevant judgment when nothing better exists.
            if winning is None:
                winning = j
                winning_score = conf

        # Fallthrough: if we never found a positive verdict, the chosen
        # status stays MISSING and the subcode reflects whichever
        # rejection reason we last recorded (or MISSING_NO_EVIDENCE if
        # no judgment was even worth considering).
        if chosen_status == CoverageStatus.MISSING and chosen_subcode == SUBCODE_MISSING_NO_EVIDENCE:
            # If at least one judgment was IRRELEVANT, leave MISSING_NO_EVIDENCE.
            # If all judgments were rejected for grounding/conf reasons, the
            # subcode was already updated above.
            agg_reason = agg_reason or (
                "All judgments produced IRRELEVANT or were rejected; row marked MISSING."
            )
            decision_log.append(agg_reason)

        # OPTIONAL downgrade: only use OPTIONAL_NOT_FOUND when the row
        # is genuinely missing evidence (subcode MISSING_NO_EVIDENCE).
        # When MISSING is caused by a rejected positive verdict
        # (MISSING_LOW_GROUNDING / MISSING_LOW_CONFIDENCE) the more
        # informative subcode is the rejection reason — the reviewer
        # needs to see WHY the verdict was rejected, not just that the
        # row is optional.
        if (
            chosen_status == CoverageStatus.MISSING
            and coverage_requirement_level == CoverageRequirementLevel.OPTIONAL
            and chosen_subcode == SUBCODE_MISSING_NO_EVIDENCE
        ):
            chosen_subcode = SUBCODE_OPTIONAL_NOT_FOUND
            decision_log.append(
                "Requirement level is OPTIONAL; subcode upgraded to OPTIONAL_NOT_FOUND."
            )

        # Final wire status: MISSING for all subcodes that downgraded a
        # would-be positive verdict, COVERED / PARTIAL / CONFLICT for the
        # accepted ones. (chosen_status already encodes this.)

        # Per-judgment low_confidence already considered. Mirror to row.
        # PARTIAL acceptance via "demoted CONFLICT" already filled
        # winning/agg_reason; if final is PARTIAL but conflicts non-empty,
        # drop them to keep the UI consistent.
        final_conflicts = (
            list(dict.fromkeys(conflicts)) if chosen_status == CoverageStatus.CONFLICT else []
        )

        # Polyakov-regression Step 9: dedup + denoise uncovered_aspects
        # before they reach the wire. Removes substrings, latin-mix
        # artefacts («регISTRATION»), and verifier-internal tags
        # («specific_object_match», «verb_match», …).
        normalized_uncovered = _normalize_uncovered_aspects(uncovered)

        # Polyakov-regression Step 6: downgrade COVERED → PARTIAL when
        # the judge itself listed uncovered aspects. The current LLM
        # frequently writes «фрагмент полностью покрывает …» while
        # simultaneously emitting `uncovered_aspects` — the row was
        # internally inconsistent: «full coverage» + «here's what's
        # missing». Force-downgrade so the reviewer sees PARTIAL with
        # the explicit list of remaining gaps. Real-package examples
        # from the Polyakov May-11 run all flipped here:
        #   * 0.11::sent1 «Регистрация, авторизация и аутентификация»
        #     COVERED with 4 aspects → PARTIAL.
        #   * 0.15::sent1 «понятный интерфейс…» COVERED with 4
        #     aspects → PARTIAL.
        #   * 0.17::sent2/sent4, 0.20::sent1 — same pattern.
        if chosen_status == CoverageStatus.COVERED and normalized_uncovered:
            decision_log.append(
                f"Step-6 downgrade COVERED → PARTIAL: judge declared "
                f"full coverage but listed {len(normalized_uncovered)} "
                f"uncovered aspect(s) after dedup."
            )
            downgrade_note = (
                f" Downgraded COVERED → PARTIAL: judge's uncovered_aspects "
                f"is non-empty after dedup ({len(normalized_uncovered)} "
                f"aspect(s) remain): "
                f"{', '.join(repr(a) for a in normalized_uncovered[:3])}."
            )
            agg_reason = (agg_reason or "") + downgrade_note
            chosen_status = CoverageStatus.PARTIAL
            chosen_subcode = SUBCODE_PARTIAL_DOWNGRADED_FROM_COVERED

        return RequirementCoverageResult(
            req_id=requirement.req_id,
            source_document_id=requirement.source_document_id,
            target_document_id=target_document_id,
            target_doc_role=target_doc_role,
            status=chosen_status,
            evidence=evidence_items,
            uncovered_aspects=normalized_uncovered,
            unjudged_pair_count=unjudged_pair_count_for_row,
            conflict_details=final_conflicts,
            low_confidence=any_low_conf,
            grounding_failed=any_grounding_failed,
            req_text=req_text,
            req_section_title=req_section_title,
            req_section_id=req_section_id,
            req_number=req_number,
            rationale=_pick_rationale(
                judgments, chosen_status, any_low_conf,
                applicability, coverage_requirement_level,
            ),
            requirement_type=req_type,
            applicability=applicability,
            severity=severity_for(
                req_type, target_doc_role, chosen_status, applicability, chosen_subcode,
            ),
            should_affect_critical=should_affect_critical(
                req_type,
                applicability,
                # OPTIONAL_NOT_FOUND never contributes to criticalCount.
                chosen_status if chosen_subcode != SUBCODE_OPTIONAL_NOT_FOUND
                else CoverageStatus.COVERED,
                target_doc_role,
            ),
            should_affect_grade=should_affect_grade(req_type, applicability),
            status_subcode=chosen_subcode,
            winning_candidate_id=(winning.unit_id if winning is not None else None),
            final_confidence=round(float(winning_score or 0.0), 4),
            aggregation_reason=agg_reason or None,
            coverage_requirement_level=coverage_requirement_level,
            evidence_trace=(
                _build_evidence_trace(
                    requirement, candidates_by_unit_id, selection_result,
                    judgments, decision_log, debug_cfg,
                )
                if debug_cfg.enabled
                else None
            ),
        )
