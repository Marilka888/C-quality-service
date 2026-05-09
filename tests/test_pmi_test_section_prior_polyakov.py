"""
Polyakov-regression: ПМИ units from canonical test sections must get a
section_prior boost regardless of req type.

Real-data observation on Polyakov 5th demo run (after the unit_type
schema fix): 31 ТЗ requirements ran through the pipeline but coverage
table on PMI side stayed at 23 MISSING / 4 NOT_APPLICABLE / 2 PARTIAL
/ 2 COVERED. The MISSING rows had top retrieval candidates from the
ПМИ environment section («Windows 10 Pro», «Intel i5-7500») — pure
noise — while the genuine test descriptions in «Методы испытаний»
section («Для проверки авторизации необходимы логин и пароль…»)
ranked lower because BoW had no lexical overlap with the requirement.

Per ГОСТ 19.301 the canonical PMI sections that hold requirement
verification are:
  * «Требования к программе» — the requirement restatement section
  * «Методы испытаний» — the verification procedure section
  * «Состав и порядок испытаний» — the test plan section
  * «Объект/Цель испытаний» — scope and goal sections
  * «Проверка требований к программной документации» — doc tests

Units from these sections now get section_prior=1.0 in the hybrid
score regardless of req type. Without this boost the LLM judge never
sees them and the report drowns in spurious MISSING.
"""
from __future__ import annotations

from app.application.use_cases.retrieve_candidates import _section_prior
from app.domain.c_quality_enums import CoverageUnitType, RequirementType
from app.domain.c_quality_models import CoverageUnit, RequirementUnit


def _req(rt: RequirementType = RequirementType.SECURITY) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Система должна обеспечивать аутентификацию.",
        normalized_text="система должна обеспечивать аутентификацию.",
        requirement_type=rt,
    )


def _unit(section_title: str, role: str = "pmi") -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id=f"doc-{role}",
        target_doc_role=role,
        unit_type=CoverageUnitType.PARAGRAPH,
        text="Для проверки авторизации необходимы логин и пароль.",
        normalized_text="для проверки авторизации необходимы логин и пароль.",
        metadata={"section_title": section_title},
    )


# ── Canonical PMI test sections must boost ────────────────────────────


def test_methods_of_testing_section_boosts_pmi_unit_polyakov() -> None:
    # «Методы испытаний» is the canonical ГОСТ 19.301 section for
    # verification procedures. Units from it must score 1.0 on the
    # section prior even when the req type isn't in _PMI_PREFERRED
    # (SECURITY isn't, but the prior must still fire).
    assert _section_prior(_req(RequirementType.SECURITY),
                          _unit("МЕТОДЫ ИСПЫТАНИЙ")) == 1.0


def test_program_requirements_section_boosts_pmi_unit_polyakov() -> None:
    assert _section_prior(_req(RequirementType.FUNCTIONAL),
                          _unit("ТРЕБОВАНИЯ К ПРОГРАММЕ")) == 1.0


def test_composition_and_test_order_section_boosts_pmi_unit_polyakov() -> None:
    assert _section_prior(_req(RequirementType.RELIABILITY),
                          _unit("Состав и порядок испытаний")) == 1.0


def test_object_of_testing_section_boosts_pmi_unit_polyakov() -> None:
    assert _section_prior(_req(RequirementType.FUNCTIONAL),
                          _unit("ОБЪЕКТ ИСПЫТАНИЙ")) == 1.0


def test_goal_of_testing_section_boosts_pmi_unit_polyakov() -> None:
    assert _section_prior(_req(RequirementType.PERFORMANCE),
                          _unit("ЦЕЛЬ ИСПЫТАНИЙ")) == 1.0


def test_program_documentation_check_section_boosts_pmi_unit_polyakov() -> None:
    assert _section_prior(_req(RequirementType.DOCUMENTATION_REQUIREMENT),
                          _unit("Проверка требований к программной документации")) == 1.0


def test_test_procedure_order_section_boosts_pmi_unit_polyakov() -> None:
    assert _section_prior(_req(RequirementType.RELIABILITY),
                          _unit("Порядок проведения испытаний")) == 1.0


# ── Non-test PMI sections do NOT boost ────────────────────────────────


def test_pmi_environment_section_does_not_boost_polyakov() -> None:
    # Environment / hardware-spec section («Технические средства»,
    # «Окружение») in ПМИ is supporting material, not test coverage.
    # The section prior must NOT fire on it — otherwise random hardware
    # mentions would outrank genuine test descriptions.
    assert _section_prior(_req(RequirementType.SECURITY),
                          _unit("Технические средства")) == 0.0


def test_pmi_references_section_does_not_boost_polyakov() -> None:
    assert _section_prior(_req(RequirementType.FUNCTIONAL),
                          _unit("Список использованных источников")) == 0.0


# ── Type-based prior still works ──────────────────────────────────────


def test_performance_in_pmi_still_boosts_via_type_polyakov() -> None:
    # PERFORMANCE was already in _PMI_PREFERRED — it must keep boosting
    # even on a non-test section title.
    assert _section_prior(_req(RequirementType.PERFORMANCE),
                          _unit("Введение")) == 1.0


def test_pz_prior_unaffected_by_pmi_test_section_change_polyakov() -> None:
    # The ПМИ test-section list must NOT add a boost to ПЗ scoring.
    # OTHER req type is not in _PZ_PREFERRED, so a ПЗ unit titled
    # «Методы испытаний» (which would boost in ПМИ) must NOT boost
    # in ПЗ — only the type-based fallback applies there.
    assert _section_prior(_req(RequirementType.OTHER),
                          _unit("Методы испытаний", role="pz")) == 0.0
    # FUNCTIONAL in PZ is in _PZ_PREFERRED — type-based prior fires
    # regardless of section title.
    assert _section_prior(_req(RequirementType.FUNCTIONAL),
                          _unit("Описание алгоритма", role="pz")) == 1.0
