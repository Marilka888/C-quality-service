"""
Task P0 #6 — DELIVERY / DOCUMENTATION classifier extension.

Hamroev-package symptom: requirements about marking, packaging, USB-носитель,
LMS, .rar/.zip, distribution, storage of printed documents, "пояснительная
записка / руководство пользователя / описание применения" all leaked into
FUNCTIONAL/OTHER and produced spurious MISSING in PMI/PZ, depressing the
C-score. Coverage rows for these types are now OUT_OF_SCOPE (delivery) or
PMI-only (documentation), and surface in a dedicated report section.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.applicability import applicability_for
from app.application.use_cases.build_coverage_report import CoverageReportBuilder
from app.application.use_cases.classify_requirement import classify_requirement
from app.domain.c_quality_enums import (
    Applicability,
    CoverageStatus,
    RequirementType,
)
from app.domain.c_quality_models import RequirementCoverageResult


# ── classifier: DELIVERY ────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "Дистрибутив поставляется на USB-носителе.",
    "Программа должна поставляться в виде архива .rar.",
    "Дистрибутив должен распространяться в виде .zip-архива.",
    "Маркировка дистрибутива должна содержать наименование и версию.",
    "Упаковка должна обеспечивать сохранность носителя при транспортировании.",
    "Транспортирование осуществляется в стандартной таре.",
    "Хранение печатных документов должно производиться в архиве 5 лет.",
    "Программное обеспечение распространяется через LMS.",
    "Комплект поставки включает руководство и установочный диск.",
])
def test_delivery_text_patterns_hamroev(text: str) -> None:
    assert classify_requirement(text) == RequirementType.DELIVERY_REQUIREMENT


def test_delivery_negative_lookahead_preserves_storage_semantics_hamroev() -> None:
    # «Хранение данных» / «хранение информации» must NOT be reclassified
    # as delivery — the storage axis still owns them. Negative lookahead
    # in the delivery pattern guards this.
    assert classify_requirement(
        "Срок хранения данных — не менее 90 дней."
    ) == RequirementType.STORAGE


def test_delivery_is_out_of_scope_in_pmi_hamroev() -> None:
    # Type→applicability is unchanged but explicit: a delivery requirement
    # is OUT_OF_SCOPE in any target role, so the row no longer participates
    # in coverage_rate or criticalCount.
    rt = classify_requirement(
        "Дистрибутив должен распространяться через USB-носитель."
    )
    assert rt == RequirementType.DELIVERY_REQUIREMENT
    assert applicability_for(rt, "PMI") == Applicability.OUT_OF_SCOPE
    assert applicability_for(rt, "PZ") == Applicability.OUT_OF_SCOPE


# ── classifier: DOCUMENTATION ───────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "Программа должна сопровождаться пояснительной запиской.",
    "Должно быть составлено руководство пользователя.",
    "Описание применения должно соответствовать ГОСТ 19.502.",
    "В состав программной документации входит руководство оператора.",
    "Текст программы должен быть оформлен по ГОСТ 19.401.",
])
def test_documentation_text_patterns_hamroev(text: str) -> None:
    assert classify_requirement(text) == RequirementType.DOCUMENTATION_REQUIREMENT


# ── report section: out_of_scope_requirements ──────────────────────────


def _row(rt: RequirementType, status: CoverageStatus, target: str) -> RequirementCoverageResult:
    return RequirementCoverageResult(
        req_id=f"r-{rt.value}",
        source_document_id="tz",
        target_document_id="doc-pmi",
        target_doc_role=target,
        requirement_type=rt,
        applicability=applicability_for(rt, target),
        status=status,
    )


def test_out_of_scope_rows_surface_in_dedicated_report_section_hamroev() -> None:
    rows = [
        _row(RequirementType.FUNCTIONAL, CoverageStatus.COVERED, "PMI"),
        _row(RequirementType.DELIVERY_REQUIREMENT, CoverageStatus.MISSING, "PMI"),
        _row(RequirementType.PROCESS_REQUIREMENT, CoverageStatus.MISSING, "PMI"),
    ]
    result = CoverageReportBuilder().build(
        job_id="j", package_id="p", source_document_id="s",
        requirement_results=rows,
    )

    # Functional row counts toward total; out-of-scope rows are tallied
    # under not_applicable and surface as a separate list for the UI.
    assert result.summary.total_requirements == 1
    assert result.summary.not_applicable == 2
    assert {r.requirement_type for r in result.out_of_scope_requirements} == {
        RequirementType.DELIVERY_REQUIREMENT,
        RequirementType.PROCESS_REQUIREMENT,
    }
    # Functional row stays in main table only.
    assert all(
        r.requirement_type != RequirementType.FUNCTIONAL
        for r in result.out_of_scope_requirements
    )
