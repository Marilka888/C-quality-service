"""
PR-C regression tests:
  BUG-3  — LLM-judge response must be grounded in evidence text.
  BUG-9  — verdicts produced from below-floor retrieval are flagged.
  BUG-14 — duplicate (req_id, target_document_id) results are collapsed.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.aggregate_coverage import CoverageAggregator
from app.domain.c_quality_enums import CoverageStatus, LLMLabel, RequirementType, Modality
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementCoverageResult,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.llm.ollama_coverage_judge import _parse_response


# ── BUG-3: grounding gate ──────────────────────────────────────────────


def _req(req_id: str = "r1") -> RequirementUnit:
    return RequirementUnit(
        req_id=req_id,
        source_document_id="doc-tz",
        text="Система должна обеспечивать аутентификацию через Keycloak.",
        normalized_text="система должна обеспечивать аутентификацию через keycloak",
    )


def _unit(unit_id: str = "u1", text: str = "Аутентификация через Keycloak реализована.") -> CoverageUnit:
    return CoverageUnit(
        unit_id=unit_id,
        target_document_id="doc-pz",
        target_doc_role="pz",
        text=text,
        normalized_text=text.lower(),
    )


def test_grounded_covered_kept_as_covered():
    """LLM cited a phrase that is a substring of the evidence — verdict survives."""
    raw = {
        "label": "COVERED",
        "confidence": 0.9,
        "matched_aspects": ["аутентификация"],
        "missing_aspects": [],
        "conflict_aspects": [],
        "cited_phrases": ["Аутентификация через Keycloak"],
        "explanation": "Фрагмент описывает реализацию аутентификации.",
    }
    j = _parse_response(
        raw, "r1", "u1", "doc-pz",
        evidence_text="Аутентификация через Keycloak реализована.",
    )
    assert j.llm_label == LLMLabel.COVERED
    assert j.low_confidence is False
    assert j.cited_phrases == ["Аутентификация через Keycloak"]
    assert j.llm_confidence == pytest.approx(0.9)


def test_grounded_match_is_case_and_whitespace_insensitive():
    raw = {
        "label": "PARTIAL",
        "confidence": 0.7,
        "matched_aspects": ["a"],
        "missing_aspects": [],
        "conflict_aspects": [],
        # Different case + extra whitespace from the evidence — still grounded.
        "cited_phrases": ["АУТЕНТИФИКАЦИЯ   через  Keycloak"],
        "explanation": "x",
    }
    j = _parse_response(
        raw, "r1", "u1", "doc-pz",
        evidence_text="Аутентификация через Keycloak реализована.",
    )
    assert j.llm_label == LLMLabel.PARTIAL
    assert j.low_confidence is False


def test_ungrounded_conflict_is_demoted_to_irrelevant():
    """Reproduces the audit-time hallucination: CONFLICT rationale mentions
    'период защиты' but the evidence has no such phrase. Must be demoted to
    IRRELEVANT with low_confidence=True and confidence capped."""
    raw = {
        "label": "CONFLICT",
        "confidence": 0.8,
        "matched_aspects": [],
        "missing_aspects": [],
        "conflict_aspects": ["период защиты"],
        "cited_phrases": ["три дня до начала периода защиты"],
        "explanation": "Фрагмент противоречит требованию по периоду защиты.",
    }
    evidence = (
        "Испытания проводятся по порядку, описанному в разделе 3. "
        "Программный интерфейс должен быть представлен в виде REST API."
    )
    j = _parse_response(raw, "r1", "u1", "doc-pmi", evidence_text=evidence)
    assert j.llm_label == LLMLabel.IRRELEVANT, "ungrounded CONFLICT must be demoted"
    assert j.low_confidence is True
    assert j.cited_phrases == []
    assert j.conflict_aspects == []
    assert j.llm_confidence <= 0.3
    assert "[ungrounded]" in j.explanation


def test_ungrounded_covered_is_demoted():
    raw = {
        "label": "COVERED",
        "confidence": 0.95,
        "matched_aspects": ["аутентификация"],
        "missing_aspects": [],
        "conflict_aspects": [],
        # Phrase NOT in evidence.
        "cited_phrases": ["биометрический сканер сетчатки"],
        "explanation": "Полное соответствие.",
    }
    j = _parse_response(
        raw, "r1", "u1", "doc-pz",
        evidence_text="Аутентификация через Keycloak реализована.",
    )
    assert j.llm_label == LLMLabel.IRRELEVANT
    assert j.low_confidence is True


def test_irrelevant_skips_grounding_check():
    """IRRELEVANT verdicts pass through without grounding — they don't claim
    coverage so there's nothing to ground."""
    raw = {
        "label": "IRRELEVANT",
        "confidence": 0.6,
        "matched_aspects": [],
        "missing_aspects": [],
        "conflict_aspects": [],
        "cited_phrases": [],
        "explanation": "Не относится к требованию.",
    }
    j = _parse_response(raw, "r1", "u1", "doc-pz", evidence_text="что угодно")
    assert j.llm_label == LLMLabel.IRRELEVANT
    assert j.low_confidence is False


def test_no_evidence_text_skips_grounding_for_legacy_callers():
    """Tests / legacy callers that don't pass evidence_text must keep working
    — grounding is a no-op when no evidence is supplied."""
    raw = {
        "label": "COVERED",
        "confidence": 0.9,
        "matched_aspects": ["x"],
        "missing_aspects": [],
        "conflict_aspects": [],
        "cited_phrases": ["arbitrary phrase"],
        "explanation": "x",
    }
    j = _parse_response(raw, "r1", "u1", "doc-pz")  # no evidence_text
    assert j.llm_label == LLMLabel.COVERED
    assert j.low_confidence is False


def test_empty_cited_phrases_with_non_irrelevant_label_is_demoted():
    """LLM said COVERED but didn't supply ANY cited_phrase → can't be grounded
    → demote. Specifically catches lazy / misformatted LLM output."""
    raw = {
        "label": "COVERED",
        "confidence": 0.9,
        "matched_aspects": ["x"],
        "missing_aspects": [],
        "conflict_aspects": [],
        "cited_phrases": [],
        "explanation": "x",
    }
    j = _parse_response(
        raw, "r1", "u1", "doc-pz",
        evidence_text="Аутентификация через Keycloak реализована.",
    )
    assert j.llm_label == LLMLabel.IRRELEVANT
    assert j.low_confidence is True


# ── BUG-3 / BUG-9: low_confidence propagation in aggregator ─────────────


def test_aggregator_propagates_low_confidence_from_judgment():
    req = _req()
    unit = _unit()
    j = PairJudgment(
        req_id=req.req_id,
        unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        llm_label=LLMLabel.COVERED,
        rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=0.3,
        low_confidence=True,
    )
    cand = RetrievedCandidate(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id, retrieval_score=0.4,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j],
        candidates_by_unit_id={unit.unit_id: cand},
        units_by_id={unit.unit_id: unit},
        target_document_id=unit.target_document_id, target_doc_role="pz",
    )
    assert res.low_confidence is True
    # Status itself still reflects the (possibly-untrustworthy) verdict —
    # downstream consumers decide how to render low-confidence rows.
    assert res.status == CoverageStatus.COVERED


def test_aggregator_low_confidence_false_when_all_judgments_grounded():
    req = _req()
    unit = _unit()
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        llm_label=LLMLabel.COVERED, rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=0.9, low_confidence=False,
    )
    cand = RetrievedCandidate(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id, retrieval_score=0.85,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j],
        candidates_by_unit_id={unit.unit_id: cand},
        units_by_id={unit.unit_id: unit},
        target_document_id=unit.target_document_id, target_doc_role="pz",
    )
    assert res.low_confidence is False


# ── BUG-12: requirement source-side context propagation ────────────────


def test_aggregator_propagates_requirement_text_and_section():
    """RequirementCoverageResult must carry req_text, req_section_title,
    req_section_id and req_number so the orchestrator can render the
    requirement card with proper locator (BUG-12)."""
    req = RequirementUnit(
        req_id="r1",
        source_document_id="doc-tz",
        source_section_id="4.1",
        text="Система должна обеспечивать аутентификацию через Keycloak.",
        normalized_text="система должна обеспечивать аутентификацию через keycloak",
        metadata={"section_title": "Требования к функциональным характеристикам",
                  "number": "4.1"},
    )
    unit = _unit()
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        llm_label=LLMLabel.COVERED, rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=0.9,
    )
    cand = RetrievedCandidate(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id, retrieval_score=0.85,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j],
        candidates_by_unit_id={unit.unit_id: cand},
        units_by_id={unit.unit_id: unit},
        target_document_id=unit.target_document_id, target_doc_role="pz",
    )
    assert res.req_text == req.text
    assert res.req_section_title == "Требования к функциональным характеристикам"
    assert res.req_section_id == "4.1"
    assert res.req_number == "4.1"


def test_aggregator_empty_shortlist_still_carries_requirement_context():
    """Even when no judgments exist (empty shortlist), the resulting
    MISSING row must carry the requirement locator so the UI can show
    "Section X: this requirement was not addressed" instead of an
    opaque req_id."""
    req = RequirementUnit(
        req_id="r2",
        source_document_id="doc-tz",
        source_section_id="4.2",
        text="Система должна сохранять журналы 90 дней.",
        normalized_text="система должна сохранять журналы 90 дней",
        metadata={"section_title": "Требования к надёжности"},
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[], candidates_by_unit_id={},
        units_by_id={}, target_document_id="doc-pmi", target_doc_role="pmi",
    )
    assert res.status == CoverageStatus.MISSING
    assert res.req_text == req.text
    assert res.req_section_title == "Требования к надёжности"
    assert res.req_section_id == "4.2"


# ── PR-F: section_number propagation via metadata.sectionNumber ─────────


def test_aggregator_reads_section_number_from_docback_metadata():
    """docback's prepared_builder.go writes the structural number under
    metadata.sectionNumber (PR-F BUG-12 follow-up). aggregate_coverage
    must pick it up so RequirementCoverageResult.req_number is populated."""
    req = RequirementUnit(
        req_id="r3",
        source_document_id="doc-tz",
        source_section_id="4.1",
        text="x", normalized_text="x",
        metadata={"sectionNumber": "4.1.2", "section_title": "X"},
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[], candidates_by_unit_id={},
        units_by_id={}, target_document_id="doc-pmi", target_doc_role="pmi",
    )
    assert res.req_number == "4.1.2"


def test_aggregator_section_number_falls_back_to_explicit_number():
    """sections-driven path stores number under `number`. Aggregator
    accepts both `number` and `sectionNumber`."""
    req = RequirementUnit(
        req_id="r4",
        source_document_id="doc-tz",
        source_section_id="5",
        text="x", normalized_text="x",
        metadata={"number": "5", "section_title": "Y"},
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[], candidates_by_unit_id={},
        units_by_id={}, target_document_id="doc-pmi", target_doc_role="pmi",
    )
    assert res.req_number == "5"


# ── PR-F: conflict_details cleared when best_status != CONFLICT ─────────


def test_aggregator_clears_conflict_details_on_demote():
    """Strong-COVERED suppression demotes weak CONFLICT to PARTIAL. The
    judgment's conflict_aspects must NOT bleed into the result's
    conflict_details — otherwise the UI shows a row labelled PARTIAL
    with conflictDetails populated, which is contradictory."""
    req = _req()
    unit_covered = _unit("u-cov", "Covered text fully matching the requirement.")
    unit_weak_conflict = _unit("u-conf", "Tangentially related text mentioning a different number.")

    j_covered = PairJudgment(
        req_id=req.req_id, unit_id=unit_covered.unit_id,
        target_document_id=unit_covered.target_document_id,
        llm_label=LLMLabel.COVERED, rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=0.95,
    )
    j_weak_conflict = PairJudgment(
        req_id=req.req_id, unit_id=unit_weak_conflict.unit_id,
        target_document_id=unit_weak_conflict.target_document_id,
        llm_label=LLMLabel.CONFLICT, rule_adjusted_label=LLMLabel.CONFLICT,
        llm_confidence=0.30,
        conflict_aspects=["spurious_aspect"],
    )
    cands = {
        unit_covered.unit_id: RetrievedCandidate(
            req_id=req.req_id, unit_id=unit_covered.unit_id,
            target_document_id=unit_covered.target_document_id, retrieval_score=0.85,
        ),
        unit_weak_conflict.unit_id: RetrievedCandidate(
            req_id=req.req_id, unit_id=unit_weak_conflict.unit_id,
            target_document_id=unit_weak_conflict.target_document_id, retrieval_score=0.55,
        ),
    }
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j_covered, j_weak_conflict],
        candidates_by_unit_id=cands,
        units_by_id={unit_covered.unit_id: unit_covered, unit_weak_conflict.unit_id: unit_weak_conflict},
        target_document_id=unit_covered.target_document_id, target_doc_role="pz",
    )
    # Weak CONFLICT was demoted by the strong-COVERED rule. Best status =
    # COVERED, so conflict_details must be empty even though the demoted
    # judgment listed `spurious_aspect`.
    assert res.status == CoverageStatus.COVERED
    assert res.conflict_details == [], (
        f"conflict_details must be cleared when best_status != CONFLICT, got {res.conflict_details}"
    )


def test_aggregator_picks_rationale_from_winning_judgment():
    """Best status COVERED → rationale must come from the COVERED judgment,
    not from a tangentially-related IRRELEVANT one earlier in the list."""
    req = _req()
    unit_irrelevant = _unit("u-irr", "Tangential text.")
    unit_covered = _unit("u-cov", "Direct match for the requirement.")
    j_irrelevant = PairJudgment(
        req_id=req.req_id, unit_id=unit_irrelevant.unit_id,
        target_document_id=unit_irrelevant.target_document_id,
        llm_label=LLMLabel.IRRELEVANT, rule_adjusted_label=LLMLabel.IRRELEVANT,
        explanation="[disabled-judge] Insufficient overlap → IRRELEVANT",
    )
    j_covered = PairJudgment(
        req_id=req.req_id, unit_id=unit_covered.unit_id,
        target_document_id=unit_covered.target_document_id,
        llm_label=LLMLabel.COVERED, rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=0.9,
        explanation="LLM: Фрагмент полностью покрывает требование.",
    )
    cands = {
        unit_irrelevant.unit_id: RetrievedCandidate(
            req_id=req.req_id, unit_id=unit_irrelevant.unit_id,
            target_document_id=unit_irrelevant.target_document_id, retrieval_score=0.50,
        ),
        unit_covered.unit_id: RetrievedCandidate(
            req_id=req.req_id, unit_id=unit_covered.unit_id,
            target_document_id=unit_covered.target_document_id, retrieval_score=0.85,
        ),
    }
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j_irrelevant, j_covered],
        candidates_by_unit_id=cands,
        units_by_id={unit_irrelevant.unit_id: unit_irrelevant, unit_covered.unit_id: unit_covered},
        target_document_id=unit_covered.target_document_id, target_doc_role="pz",
    )
    assert res.status == CoverageStatus.COVERED
    assert res.rationale == "LLM: Фрагмент полностью покрывает требование."


def test_aggregator_default_rationale_for_missing_with_empty_explanations():
    """When status=MISSING and every judgment has an empty explanation,
    the result must carry a default Russian sentence — UI never shows
    a bare 'Не покрыто' badge with no context."""
    req = _req()
    unit = _unit()
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        llm_label=LLMLabel.IRRELEVANT, rule_adjusted_label=LLMLabel.IRRELEVANT,
        explanation="",  # empty — reproduces the audit-time symptom
    )
    cand = RetrievedCandidate(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id, retrieval_score=0.55,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j],
        candidates_by_unit_id={unit.unit_id: cand},
        units_by_id={unit.unit_id: unit},
        target_document_id=unit.target_document_id, target_doc_role="pz",
    )
    assert res.status == CoverageStatus.MISSING
    assert res.rationale, "MISSING result must carry a non-empty rationale"
    assert "не найдено" in res.rationale.lower(), (
        f"default rationale should explain absence, got: {res.rationale!r}"
    )


def test_aggregator_default_rationale_low_confidence_variant():
    """When MISSING is caused by below-floor retrieval, the default
    rationale text must reference the retrieval-quality issue so the
    reviewer knows it's not a hard absence."""
    req = _req()
    unit = _unit()
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        llm_label=LLMLabel.IRRELEVANT, rule_adjusted_label=LLMLabel.IRRELEVANT,
        explanation="",
        low_confidence=True,
    )
    cand = RetrievedCandidate(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id, retrieval_score=0.30,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j],
        candidates_by_unit_id={unit.unit_id: cand},
        units_by_id={unit.unit_id: unit},
        target_document_id=unit.target_document_id, target_doc_role="pz",
    )
    assert res.status == CoverageStatus.MISSING
    assert res.low_confidence is True
    assert res.rationale and "retrieval" in res.rationale.lower()


def test_aggregator_keeps_conflict_details_when_status_is_conflict():
    """Sanity: a real CONFLICT must keep its conflict_details."""
    req = _req()
    unit = _unit()
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        llm_label=LLMLabel.CONFLICT, rule_adjusted_label=LLMLabel.CONFLICT,
        llm_confidence=0.85,
        conflict_aspects=["response_time mismatch"],
    )
    cand = RetrievedCandidate(
        req_id=req.req_id, unit_id=unit.unit_id,
        target_document_id=unit.target_document_id, retrieval_score=0.85,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[j],
        candidates_by_unit_id={unit.unit_id: cand},
        units_by_id={unit.unit_id: unit},
        target_document_id=unit.target_document_id, target_doc_role="pz",
    )
    assert res.status == CoverageStatus.CONFLICT
    assert res.conflict_details == ["response_time mismatch"]


# ── BUG-14: dedup ───────────────────────────────────────────────────────


def test_pipeline_dedup_collapses_duplicate_pairs(monkeypatch):
    """Build a request that flows through the full pipeline with the
    DisabledCoverageJudge and verify that if duplicate results are produced
    (e.g. via a hand-crafted RequirementBuilder.build that emits two
    requirements with the same req_id), the orchestrator collapses them."""
    from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
    from app.application.use_cases.build_requirements import RequirementBuilder

    pipeline = CoverageAnalysisPipeline()

    # Force two requirements with the SAME req_id to leak through the
    # builder. Real RequirementBuilder.build prevents this, but we simulate
    # a regression by patching it.
    duplicate_req_id = "dup::req::1"
    def _fake_build(self, artifact):
        return [
            RequirementUnit(
                req_id=duplicate_req_id,
                source_document_id=artifact.get("document_id", "doc-tz"),
                text="Хранить журнал 90 дней.",
                normalized_text="хранить журнал 90 дней",
            ),
            RequirementUnit(
                req_id=duplicate_req_id,
                source_document_id=artifact.get("document_id", "doc-tz"),
                text="Хранить журнал 90 дней.",
                normalized_text="хранить журнал 90 дней",
            ),
        ]
    monkeypatch.setattr(RequirementBuilder, "build", _fake_build)

    request = {
        "job_id": "test-dedup",
        "package_id": "pkg-1",
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi"],
        "documents": [
            {
                "document_id": "doc-tz",
                "doc_role": "tz",
                "prepared_artifact": {
                    "document_id": "doc-tz",
                    "doc_role": "tz",
                    "fragments": [],
                    "sections": [],
                },
            },
            {
                "document_id": "doc-pmi",
                "doc_role": "pmi",
                "prepared_artifact": {
                    "document_id": "doc-pmi",
                    "doc_role": "pmi",
                    "fragments": [
                        {"fragment_id": "f1", "section_id": "3.1",
                         "text": "Проверяется хранение журнала 90 дней."}
                    ],
                    "sections": [{"section_id": "3.1", "title": "Порядок"}],
                },
            },
        ],
        "options": {
            "enable_llm_judge": False,
            "min_retrieval_score": 0.0,
        },
    }

    result = pipeline.run(request)

    pmi_results = [r for r in result.requirement_results if r.target_document_id == "doc-pmi"]
    same_id_results = [r for r in pmi_results if r.req_id == duplicate_req_id]
    assert len(same_id_results) == 1, (
        f"dedup must collapse duplicate (req_id, target) to one result, got "
        f"{len(same_id_results)}: {[r.status for r in same_id_results]}"
    )
    assert any("DUPLICATE_PAIRS" in w for w in result.warnings)


# ── BUG-9: evidence floor flag set in pipeline ──────────────────────────


def test_pipeline_marks_low_confidence_when_below_evidence_floor(monkeypatch):
    """A shortlist whose max retrieval_score sits below evidence_floor must
    produce results with low_confidence=True regardless of the judge label.
    We mock CandidateRetriever to return a candidate with a low score."""
    from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
    from app.application.use_cases.retrieve_candidates import CandidateRetriever

    pipeline = CoverageAnalysisPipeline()

    def _fake_retrieve(self, req, units):
        if not units:
            return []
        u = units[0]
        return [
            RetrievedCandidate(
                req_id=req.req_id,
                unit_id=u.unit_id,
                target_document_id=u.target_document_id,
                # Below default evidence_floor=0.5
                retrieval_score=0.3,
                lexical_score=0.3,
                semantic_score=0.3,
            )
        ]
    monkeypatch.setattr(CandidateRetriever, "retrieve", _fake_retrieve)

    request = {
        "job_id": "test-floor",
        "package_id": "pkg-1",
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi"],
        "documents": [
            {
                "document_id": "doc-tz", "doc_role": "tz",
                "prepared_artifact": {
                    "document_id": "doc-tz", "doc_role": "tz",
                    "fragments": [
                        {"fragment_id": "f1", "section_id": "4.1",
                         "text": "Система должна обеспечивать журналирование событий."}
                    ],
                    "sections": [{"section_id": "4.1", "title": "Требования"}],
                    "requirement_candidates": [
                        {"text": "Система должна обеспечивать журналирование событий.",
                         "section_id": "4.1", "fragment_id": "f1"}
                    ],
                },
            },
            {
                "document_id": "doc-pmi", "doc_role": "pmi",
                "prepared_artifact": {
                    "document_id": "doc-pmi", "doc_role": "pmi",
                    "fragments": [
                        {"fragment_id": "p1", "section_id": "3.1",
                         "text": "Случайный фрагмент про что-то иное."}
                    ],
                    "sections": [{"section_id": "3.1", "title": "Порядок"}],
                },
            },
        ],
        "options": {
            "enable_llm_judge": False,
            "min_retrieval_score": 0.0,
        },
    }

    result = pipeline.run(request)
    pmi_results = [r for r in result.requirement_results
                   if r.target_document_id == "doc-pmi"]
    assert pmi_results, "expected at least one PMI result"
    assert all(r.low_confidence for r in pmi_results), (
        f"every below-floor result must be low_confidence; got: "
        f"{[(r.req_id, r.low_confidence, r.status) for r in pmi_results]}"
    )
