"""
Stage 7: aggregate per (requirement, target_document) judgments into a
RequirementCoverageResult.

Priority: CONFLICT > COVERED > PARTIAL > MISSING
"""
from __future__ import annotations

from typing import Dict, List

from app.application.use_cases.applicability import (
    applicability_for,
    severity_for,
    should_affect_critical,
    should_affect_grade,
)
from app.domain.c_quality_enums import (
    Applicability,
    CoverageStatus,
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


def _pick_rationale(
    judgments: List[PairJudgment],
    best_status: CoverageStatus,
    low_confidence: bool,
) -> str:
    """Choose the rationale string that best explains `best_status`.

    Priority:
      1. Explanation of a judgment whose `rule_adjusted_label` maps to
         `best_status` (the "winning" judgment).
      2. Explanation of any judgment that has non-empty text — caller-
         friendly fallback when no winner has a useful message (LLM
         returned an empty explanation for a confident verdict).
      3. Default Russian text for MISSING — different copy when
         low_confidence so the UI can hint at retrieval-quality issues.
      4. Empty string for non-MISSING statuses with no usable
         explanation (should not happen in practice — COVERED/PARTIAL/
         CONFLICT always carry rationale text from LLM or rule verifier).
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

    if best_status == CoverageStatus.MISSING:
        return (
            _DEFAULT_RATIONALE_MISSING_LOWCONF
            if low_confidence
            else _DEFAULT_RATIONALE_MISSING
        )
    return ""


class CoverageAggregator:
    def aggregate(
        self,
        requirement: RequirementUnit,
        judgments: List[PairJudgment],
        candidates_by_unit_id: Dict[str, RetrievedCandidate],
        units_by_id: Dict[str, CoverageUnit],
        target_document_id: str,
        target_doc_role: str,
    ) -> RequirementCoverageResult:
        # BUG-12: snapshot the requirement's source-side context once so
        # both the empty-shortlist branch and the regular branch return
        # the same locator fields.
        req_text = requirement.text
        # `metadata.section_title` is populated by RequirementBuilder in
        # the sections-driven and model-driven paths (and forwarded from
        # prepare-service candidates' metadata in the candidates path).
        meta = requirement.metadata or {}
        req_section_title = meta.get("section_title") or meta.get("sectionTitle") or None
        req_section_id = requirement.source_section_id
        # PR-F follow-up to BUG-12: docback's prepared_builder forwards the
        # canonical structural number under "sectionNumber". Fall back to
        # other historical spellings so older payloads still work.
        req_number = (
            meta.get("number")
            or meta.get("sectionNumber")
            or meta.get("source_number")
            or None
        )

        # PR-G refactor: type-aware applicability / severity. Computed
        # once per (requirement, target) — the values are stable across
        # the empty-shortlist branch and the regular branch below.
        req_type = requirement.requirement_type or RequirementType.OTHER
        applicability = applicability_for(req_type, target_doc_role)

        if not judgments:
            empty_status = CoverageStatus.MISSING
            return RequirementCoverageResult(
                req_id=requirement.req_id,
                source_document_id=requirement.source_document_id,
                target_document_id=target_document_id,
                target_doc_role=target_doc_role,
                status=empty_status,
                req_text=req_text,
                req_section_title=req_section_title,
                req_section_id=req_section_id,
                req_number=req_number,
                rationale=_pick_rationale([], empty_status, low_confidence=False),
                requirement_type=req_type,
                applicability=applicability,
                severity=severity_for(req_type, target_doc_role, empty_status, applicability),
                should_affect_critical=should_affect_critical(req_type, applicability, empty_status),
                should_affect_grade=should_affect_grade(req_type, applicability),
            )

        best_status = CoverageStatus.MISSING
        evidence_items: List[EvidenceItem] = []
        uncovered: List[str] = []
        conflicts: List[str] = []

        # Pre-scan: is there any high-confidence COVERED judgment in the
        # shortlist? If yes, we suppress CONFLICT when it comes from a
        # less-confident pair. Rationale: the rule verifier's numeric
        # conflict signal is noisy on compound requirements where one
        # PMI unit cleanly covers the aspect and another happens to share
        # a different number. In the demo report on 4 packages this was
        # the dominant source of CONFLICT false-positives (pkg_0008: 18).
        strong_covered = max(
            (j.llm_confidence or 0.0
             for j in judgments
             if j.rule_adjusted_label == LLMLabel.COVERED),
            default=0.0,
        )
        has_strong_covered = strong_covered >= 0.8

        for j in judgments:
            status = _label_to_status(j.rule_adjusted_label)
            # Demote CONFLICT → PARTIAL when a strong COVERED already
            # exists for this requirement and this particular conflict
            # judgment is weak (conf < strong_covered). The judge on the
            # winning pair already resolved the ambiguity.
            if (
                status == CoverageStatus.CONFLICT
                and has_strong_covered
                and (j.llm_confidence or 0.0) < strong_covered
            ):
                status = CoverageStatus.PARTIAL

            if _STATUS_RANK[status] > _STATUS_RANK[best_status]:
                best_status = status

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

        # TODO: future — composite partial coverage across multiple fragments

        # BUG-3: propagate per-judgment low_confidence into the result. Any
        # ungrounded judgment that contributed to `best_status` taints the
        # whole row — the orchestrator treats it as MISSING-equivalent for
        # grade calculations and the UI dims the row.
        any_low_conf = any(getattr(j, "low_confidence", False) for j in judgments)

        # PR-F follow-up to BUG-3: when best_status is not CONFLICT, drop
        # any conflict_details collected from judgments that were demoted
        # (CONFLICT→PARTIAL by the strong-COVERED suppression rule above,
        # or never reached CONFLICT in the first place). Otherwise the UI
        # shows a row whose status reads PARTIAL but whose conflictDetails
        # array still lists numeric/aspect contradictions, which is
        # contradictory and confuses reviewers.
        final_conflicts = list(dict.fromkeys(conflicts)) if best_status == CoverageStatus.CONFLICT else []

        return RequirementCoverageResult(
            req_id=requirement.req_id,
            source_document_id=requirement.source_document_id,
            target_document_id=target_document_id,
            target_doc_role=target_doc_role,
            status=best_status,
            evidence=evidence_items,
            uncovered_aspects=list(dict.fromkeys(uncovered)),
            conflict_details=final_conflicts,
            low_confidence=any_low_conf,
            req_text=req_text,
            req_section_title=req_section_title,
            req_section_id=req_section_id,
            req_number=req_number,
            rationale=_pick_rationale(judgments, best_status, any_low_conf),
            requirement_type=req_type,
            applicability=applicability,
            severity=severity_for(req_type, target_doc_role, best_status, applicability),
            should_affect_critical=should_affect_critical(req_type, applicability, best_status),
            should_affect_grade=should_affect_grade(req_type, applicability),
        )
