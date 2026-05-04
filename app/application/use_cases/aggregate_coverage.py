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

from typing import Any, Dict, List, Optional

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

        req_type = requirement.requirement_type or RequirementType.OTHER
        applicability = applicability_for(req_type, target_doc_role)
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
                severity=severity_for(req_type, target_doc_role, status, applicability),
                should_affect_critical=should_affect_critical(req_type, applicability, status),
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
                severity=severity_for(req_type, target_doc_role, status, applicability),
                should_affect_critical=should_affect_critical(
                    req_type,
                    applicability,
                    # OPTIONAL_NOT_FOUND must not contribute to criticalCount.
                    status if coverage_requirement_level == CoverageRequirementLevel.REQUIRED
                    else CoverageStatus.COVERED,
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

        # ── Branch C: judgments exist — evidence-based decision ──────
        any_low_conf = any(getattr(j, "low_confidence", False) for j in judgments)

        # Build EvidenceItem list (used by the wire payload).
        evidence_items: List[EvidenceItem] = []
        uncovered: List[str] = []
        conflicts: List[str] = []
        for j in judgments:
            unit = units_by_id.get(j.unit_id)
            candidate = candidates_by_unit_id.get(j.unit_id)
            if unit is not None:
                evidence_items.append(
                    EvidenceItem(
                        unit_id=j.unit_id,
                        fragment_id=unit.fragment_id,
                        section_id=unit.section_id,
                        text=unit.text[:300],
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
        med_thr = aggregator_cfg.medium_retrieval_threshold

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
                    decision_log.append(
                        f"CONFLICT pair unit={j.unit_id[:12]} rejected: "
                        f"verifier did not confirm (verifier_actions={list(getattr(j, 'verifier_actions', []) or [])})."
                    )
                    chosen_status = CoverageStatus.PARTIAL
                    chosen_subcode = SUBCODE_PARTIAL
                    winning = j
                    winning_score = conf
                    agg_reason = (
                        f"CONFLICT downgraded to PARTIAL: verifier did not confirm "
                        f"the contradiction (conf={conf:.2f}, retrieval={r_score:.2f})."
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
                # PARTIAL has a softer bar — accept when retrieval is at
                # least medium AND grounding holds. Otherwise demote to
                # MISSING_LOW_GROUNDING / MISSING_LOW_CONFIDENCE.
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

        return RequirementCoverageResult(
            req_id=requirement.req_id,
            source_document_id=requirement.source_document_id,
            target_document_id=target_document_id,
            target_doc_role=target_doc_role,
            status=chosen_status,
            evidence=evidence_items,
            uncovered_aspects=list(dict.fromkeys(uncovered)),
            conflict_details=final_conflicts,
            low_confidence=any_low_conf,
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
            severity=severity_for(req_type, target_doc_role, chosen_status, applicability),
            should_affect_critical=should_affect_critical(
                req_type,
                applicability,
                # OPTIONAL_NOT_FOUND never contributes to criticalCount.
                chosen_status if chosen_subcode != SUBCODE_OPTIONAL_NOT_FOUND
                else CoverageStatus.COVERED,
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
