"""
PR-G refactor: requirement typing, applicability/severity routing, and
same-aspect / negation-compatibility validation in PairVerifier.

Test cases A-J mirror the spec's "must not regress" matrix.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.aggregate_coverage import CoverageAggregator
from app.application.use_cases.applicability import (
    applicability_for,
    severity_for,
    should_affect_critical,
    should_affect_grade,
)
from app.application.use_cases.classify_requirement import classify_requirement
from app.application.use_cases.verify_pairs import PairVerifier
from app.domain.c_quality_enums import (
    Applicability,
    CoverageStatus,
    LLMLabel,
    Modality,
    RequirementType,
)
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)


# ── Classifier sanity ───────────────────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("Время отклика приложения не должно превышать 3 секунд.", RequirementType.PERFORMANCE),
    ("Система должна быть устойчива к атакам типа «Внедрение кода».", RequirementType.SECURITY),
    ("Система не должна аварийно завершаться при возникновении ошибки.", RequirementType.RELIABILITY),
    ("Исходные коды должны быть написаны на TypeScript с использованием Angular.",
     RequirementType.ARCHITECTURE_IMPLEMENTATION),
    # PR-K follow-up: extended tech-stack regex (Python / FastAPI / backend / etc).
    ("Backend-часть должна быть реализована на Python с использованием FastAPI.",
     RequirementType.ARCHITECTURE_IMPLEMENTATION),
    ("Серверная часть приложения построена на Go с фреймворком Gin.",
     RequirementType.ARCHITECTURE_IMPLEMENTATION),
    ("Развёртывание выполняется через Docker Compose и Kubernetes.",
     RequirementType.ARCHITECTURE_IMPLEMENTATION),
    ("Документация должна быть загружена в SmartLMS за три дня до защиты.",
     RequirementType.DELIVERY_REQUIREMENT),
    ("Все входные данные отправляются через REST API в формате JSON.",
     RequirementType.DATA_IO),
    ("Макет интерфейса должен быть разработан в Figma.", RequirementType.INTERFACE),
    ("Регистрация, авторизация и аутентификация пользователей.", RequirementType.SECURITY),
    ("Хранить журнал событий 90 дней.", RequirementType.LOGGING),
    ("Климатические условия эксплуатации должны соответствовать ГОСТ.",
     RequirementType.ENVIRONMENT_REQUIREMENT),
])
def test_classify_text_only(text, expected):
    assert classify_requirement(text) == expected, text


@pytest.mark.parametrize("text, section_title, expected", [
    ("Любой текст", "СТАДИИ И ЭТАПЫ РАЗРАБОТКИ", RequirementType.PROCESS_REQUIREMENT),
    ("Любой текст", "Технико-экономические показатели", RequirementType.ECONOMIC_OR_NEED),
    ("Любой текст", "Требования к программной документации", RequirementType.DOCUMENTATION_REQUIREMENT),
    ("Любой текст", "Требования к интерфейсу", RequirementType.INTERFACE),
    ("Любой текст", "Контроль входной информации", RequirementType.SECURITY),
    ("Любой текст", "Требования к временным характеристикам", RequirementType.PERFORMANCE),
])
def test_classify_section_title_takes_priority(text, section_title, expected):
    assert classify_requirement(text, section_title) == expected


def test_classify_other_when_no_rule_fires():
    assert classify_requirement("Просто прозаический текст без триггеров.") == RequirementType.OTHER


# ── Applicability / severity matrix ─────────────────────────────────────


def test_delivery_requirement_is_out_of_scope_everywhere():
    for role in ("pmi", "pz"):
        assert applicability_for(RequirementType.DELIVERY_REQUIREMENT, role) == Applicability.OUT_OF_SCOPE


def test_process_requirement_is_out_of_scope_everywhere():
    for role in ("pmi", "pz"):
        assert applicability_for(RequirementType.PROCESS_REQUIREMENT, role) == Applicability.OUT_OF_SCOPE


def test_architecture_is_pz_only():
    assert applicability_for(RequirementType.ARCHITECTURE_IMPLEMENTATION, "pz") == Applicability.APPLICABLE
    assert applicability_for(RequirementType.ARCHITECTURE_IMPLEMENTATION, "pmi") == Applicability.NOT_APPLICABLE


def test_economic_is_pz_only():
    assert applicability_for(RequirementType.ECONOMIC_OR_NEED, "pz") == Applicability.APPLICABLE
    assert applicability_for(RequirementType.ECONOMIC_OR_NEED, "pmi") == Applicability.NOT_APPLICABLE


def test_security_applies_to_both():
    assert applicability_for(RequirementType.SECURITY, "pmi") == Applicability.APPLICABLE
    assert applicability_for(RequirementType.SECURITY, "pz") == Applicability.APPLICABLE


def test_should_affect_critical_excludes_out_of_scope():
    assert not should_affect_critical(
        RequirementType.DELIVERY_REQUIREMENT, Applicability.OUT_OF_SCOPE, CoverageStatus.MISSING
    )


def test_should_affect_critical_excludes_documentation_missing():
    assert not should_affect_critical(
        RequirementType.DOCUMENTATION_REQUIREMENT, Applicability.APPLICABLE, CoverageStatus.MISSING
    )


def test_should_affect_critical_includes_security_missing():
    assert should_affect_critical(
        RequirementType.SECURITY, Applicability.APPLICABLE, CoverageStatus.MISSING
    )


def test_should_affect_critical_includes_any_conflict():
    assert should_affect_critical(
        RequirementType.PERFORMANCE, Applicability.APPLICABLE, CoverageStatus.CONFLICT
    )


def test_should_affect_grade_excludes_out_of_scope():
    assert not should_affect_grade(RequirementType.DELIVERY_REQUIREMENT, Applicability.OUT_OF_SCOPE)


# ── PZ asymmetry: spec-class types are OPTIONAL in PZ ────────────────────


def test_pz_functional_is_optional_not_required():
    """Audit (Polyakov ВКР): functional MISSING in PZ is structural — the
    PZ describes implementation, not the spec. Must be OPTIONAL so it
    doesn't inflate criticalCount."""
    from app.application.use_cases.applicability import (
        coverage_requirement_level_for,
    )
    from app.domain.c_quality_enums import CoverageRequirementLevel

    assert coverage_requirement_level_for(
        RequirementType.FUNCTIONAL, "pz",
    ) == CoverageRequirementLevel.OPTIONAL
    assert coverage_requirement_level_for(
        RequirementType.DATA_IO, "pz",
    ) == CoverageRequirementLevel.OPTIONAL
    # In PMI same types remain REQUIRED.
    assert coverage_requirement_level_for(
        RequirementType.FUNCTIONAL, "pmi",
    ) == CoverageRequirementLevel.REQUIRED


def test_pz_security_and_reliability_remain_required():
    """SECURITY / RELIABILITY / ARCHITECTURE_IMPLEMENTATION genuinely
    belong in a design document — they stay REQUIRED in PZ."""
    from app.application.use_cases.applicability import (
        coverage_requirement_level_for,
    )
    from app.domain.c_quality_enums import CoverageRequirementLevel

    for t in (
        RequirementType.SECURITY,
        RequirementType.RELIABILITY,
        RequirementType.ARCHITECTURE_IMPLEMENTATION,
    ):
        assert coverage_requirement_level_for(
            t, "pz",
        ) == CoverageRequirementLevel.REQUIRED, f"{t} must be REQUIRED in PZ"


def test_should_affect_critical_pz_functional_missing_is_warning():
    """Polyakov-style: functional MISSING in PZ must NOT contribute to
    criticalCount. Same call without target_role keeps legacy behaviour."""
    # With target_role="pz" — non-critical (warning).
    assert not should_affect_critical(
        RequirementType.FUNCTIONAL,
        Applicability.APPLICABLE,
        CoverageStatus.MISSING,
        target_role="pz",
    )
    # With target_role="pmi" — critical (legacy).
    assert should_affect_critical(
        RequirementType.FUNCTIONAL,
        Applicability.APPLICABLE,
        CoverageStatus.MISSING,
        target_role="pmi",
    )
    # No target_role argument — backwards-compatible behaviour: critical.
    assert should_affect_critical(
        RequirementType.FUNCTIONAL,
        Applicability.APPLICABLE,
        CoverageStatus.MISSING,
    )


def test_should_affect_critical_pz_security_missing_still_critical():
    """SECURITY/RELIABILITY missing in PZ remains critical — those
    requirements DO belong in PZ."""
    assert should_affect_critical(
        RequirementType.SECURITY,
        Applicability.APPLICABLE,
        CoverageStatus.MISSING,
        target_role="pz",
    )
    assert should_affect_critical(
        RequirementType.RELIABILITY,
        Applicability.APPLICABLE,
        CoverageStatus.MISSING,
        target_role="pz",
    )


def test_severity_security_missing_pmi_is_high():
    sev = severity_for(RequirementType.SECURITY, "pmi", CoverageStatus.MISSING, Applicability.APPLICABLE)
    assert sev == "high"


def test_severity_documentation_missing_is_medium():
    sev = severity_for(RequirementType.DOCUMENTATION_REQUIREMENT, "pmi", CoverageStatus.MISSING, Applicability.APPLICABLE)
    assert sev == "medium"


def test_severity_delivery_out_of_scope_is_low():
    sev = severity_for(RequirementType.DELIVERY_REQUIREMENT, "pmi", CoverageStatus.MISSING, Applicability.OUT_OF_SCOPE)
    assert sev == "low"


# ── PairVerifier: same-aspect + negation compatibility ──────────────────


def _req(text: str, req_type: RequirementType, **kw) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="doc-tz",
        text=text,
        normalized_text=text.lower(),
        requirement_type=req_type,
        modality=kw.get("modality", Modality.MUST),
        constraints=kw.get("constraints", []),
        entities=kw.get("entities", []),
    )


def _unit(text: str, **kw) -> CoverageUnit:
    return CoverageUnit(
        unit_id=kw.get("unit_id", "u1"),
        target_document_id=kw.get("doc_id", "doc-pmi"),
        target_doc_role=kw.get("role", "pmi"),
        text=text,
        normalized_text=text.lower(),
        constraints=kw.get("constraints", []),
        entities=kw.get("entities", []),
    )


# Case A: false conflict prevention — performance vs reliability metrics
def test_case_A_false_conflict_performance_vs_reliability():
    """PERFORMANCE требование о времени отклика 3 сек НЕ должно
    конфликтовать с RELIABILITY-фрагментом про время восстановления."""
    req = _req(
        "Время отклика приложения не должно превышать 3 секунд.",
        RequirementType.PERFORMANCE,
        constraints=[Constraint(kind="response_time", operator="<=", value=3.0, unit="sec")],
    )
    unit = _unit(
        "Время восстановления не должно превышать времени перезагрузки операционной системы.",
        constraints=[],  # no numeric constraint extracted
    )
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id, target_document_id=unit.target_document_id,
        llm_label=LLMLabel.IRRELEVANT, llm_confidence=0.4,
    )
    out = PairVerifier().verify(j, req, unit)
    # No numeric_conflict between [response_time=3sec] and [no constraints]
    # → must NOT be CONFLICT.
    assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
        f"PERFORMANCE vs RELIABILITY without same metric must NOT be CONFLICT, got {out.rule_adjusted_label}"
    )


# Case B: negation compatibility — "не должна падать" ≡ "должна продолжать работать"
def test_case_B_negation_compatible_not_conflict():
    req = _req(
        "Система не должна аварийно завершаться при ошибке.",
        RequirementType.RELIABILITY,
        modality=Modality.MUST_NOT,
    )
    unit = _unit(
        "В случае ошибки система должна продолжать корректно функционировать.",
    )
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id, target_document_id=unit.target_document_id,
        llm_label=LLMLabel.CONFLICT, llm_confidence=0.6,
        explanation="LLM detected modality mismatch (false positive).",
    )
    out = PairVerifier().verify(j, req, unit)
    assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
        f"same-outcome negation pair must not be CONFLICT, got {out.rule_adjusted_label}"
    )
    assert "[rule]" in out.explanation


# Case G: delivery requirement is out of scope for C-quality
def test_case_G_delivery_out_of_scope():
    req = _req(
        "Документация должна быть загружена в SmartLMS за три дня до защиты.",
        RequirementType.DELIVERY_REQUIREMENT,
    )
    unit = _unit("Произвольный фрагмент.")

    # Even if LLM mistakenly says CONFLICT, verifier must demote.
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id, target_document_id=unit.target_document_id,
        llm_label=LLMLabel.CONFLICT, llm_confidence=0.8,
    )
    out = PairVerifier().verify(j, req, unit)
    assert out.rule_adjusted_label != LLMLabel.CONFLICT

    # And aggregator must mark the row OUT_OF_SCOPE / not affecting critical.
    cand = RetrievedCandidate(
        req_id=req.req_id, unit_id=unit.unit_id, target_document_id=unit.target_document_id,
        retrieval_score=0.5,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[out],
        candidates_by_unit_id={unit.unit_id: cand},
        units_by_id={unit.unit_id: unit},
        target_document_id="doc-pmi", target_doc_role="pmi",
    )
    assert res.applicability == Applicability.OUT_OF_SCOPE
    assert res.should_affect_critical is False
    assert res.should_affect_grade is False


# Case H: process requirement is out of scope
def test_case_H_process_out_of_scope():
    req = _req(
        "Стадии и этапы разработки: ТЗ, рабочий проект, испытания.",
        RequirementType.PROCESS_REQUIREMENT,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[], candidates_by_unit_id={}, units_by_id={},
        target_document_id="doc-pmi", target_doc_role="pmi",
    )
    assert res.applicability == Applicability.OUT_OF_SCOPE
    assert res.should_affect_critical is False
    assert res.severity == "low"


def test_aggregator_propagates_typing_for_applicable_security():
    req = _req(
        "Система должна быть устойчива к атакам типа «Внедрение кода».",
        RequirementType.SECURITY,
    )
    res = CoverageAggregator().aggregate(
        requirement=req, judgments=[], candidates_by_unit_id={}, units_by_id={},
        target_document_id="doc-pmi", target_doc_role="pmi",
    )
    # SECURITY is applicable in both PMI and PZ; MISSING SECURITY → critical.
    assert res.applicability == Applicability.APPLICABLE
    assert res.requirement_type == RequirementType.SECURITY
    assert res.should_affect_critical is True
    assert res.severity == "high"


def test_aggregator_architecture_in_pmi_is_not_applicable():
    req = _req(
        "Исходные коды должны быть написаны на TypeScript с использованием Angular.",
        RequirementType.ARCHITECTURE_IMPLEMENTATION,
    )
    res_pmi = CoverageAggregator().aggregate(
        requirement=req, judgments=[], candidates_by_unit_id={}, units_by_id={},
        target_document_id="doc-pmi", target_doc_role="pmi",
    )
    assert res_pmi.applicability == Applicability.NOT_APPLICABLE
    assert res_pmi.should_affect_critical is False

    res_pz = CoverageAggregator().aggregate(
        requirement=req, judgments=[], candidates_by_unit_id={}, units_by_id={},
        target_document_id="doc-pz", target_doc_role="pz",
    )
    assert res_pz.applicability == Applicability.APPLICABLE


# Defense-in-depth: a CONFLICT verdict on a PROCESS requirement is forced
# down to PARTIAL by the verifier (type-cannot-conflict gate).
def test_verifier_blocks_conflict_for_process_requirement():
    req = _req(
        "Стадии и этапы разработки: ТЗ, рабочий проект, испытания.",
        RequirementType.PROCESS_REQUIREMENT,
    )
    unit = _unit("Совершенно произвольный фрагмент про числа: 90 vs 30 дней.")
    j = PairJudgment(
        req_id=req.req_id, unit_id=unit.unit_id, target_document_id=unit.target_document_id,
        llm_label=LLMLabel.CONFLICT, llm_confidence=0.9,
    )
    out = PairVerifier().verify(j, req, unit)
    assert out.rule_adjusted_label != LLMLabel.CONFLICT
