"""
Polyakov-regression (2026-05-10, R2): narrative-PZ retrieval boost.

The Polyakov ВКР PZ side gave 0 COVERED / 1 PARTIAL / 30 MISSING. The
ВКР clearly contains implementation evidence for many TZ requirements
(Angular/TypeScript/REST/JSON, role hierarchy, DSpace community
structure), but lex/sem retrieval misses them because:
  * PZ uses implementation vocabulary («.ts», «.html», «архитектур»,
    «иерархи коллекций», «авторизованный пользователь»);
  * TZ uses specification vocabulary («HTML и CSS», «TypeScript с
    Angular», «многоуровневая иерархия данных репозитория», «роль»).
  * The vocab gap stops candidates above evidence_floor=0.30 → no
    LLM call → MISSING_NO_EVIDENCE.

Three R2 fixes shipped together:

  1. File-extension token extraction (`_FILE_EXT_TOKEN_RE`): PZ form
     «.ts/.html/.css/.scss» now extractable as aspect tokens (the old
     `_ASPECT_TOKEN_RE` required a leading letter and missed them).
     Both `.html` and the bare `html` form are added so they overlap
     with TZ «HTML».

  2. New aspect-alias families and `_BOOSTABLE_ASPECTS`:
     web_markup / container_docker / vcs_git / spa_arch /
     user_roles / dspace_hierarchy / auth_access / rest_api.
     These light up on PZ implementation paragraphs that lex/sem
     retrieval routinely missed.

  3. `_PZ_NARRATIVE_SECTION_RE`: section-title boost mirror of
     `_PMI_TEST_SECTION_RE`. Sections like «Архитектура клиентской
     части», «Обоснование средств разработки», «Иерархия данных
     DSpace», «Реализация», «Развёртывание» get section_prior=1.0
     unconditionally so units from those headings don't get filtered
     out by evidence_floor before the LLM gets a chance to evaluate.
"""
from __future__ import annotations

from app.application.use_cases.retrieve_candidates import (
    _PZ_NARRATIVE_SECTION_RE,
    _aspect_tokens,
    _boostable_aspect_overlap,
    _section_prior,
)
from app.domain.c_quality_enums import CoverageUnitType, RequirementType
from app.domain.c_quality_models import CoverageUnit, RequirementUnit


# ── Token extraction: file extensions ──────────────────────────────


def test_file_extension_html_extracted_as_aspect_token() -> None:
    """Polyakov 0.27::sent2 PZ shape: PZ describes «.ts, .html, .css,
    .scss». Both the dot-prefixed and bare names must be extractable
    as aspect tokens so an overlap with TZ «HTML» fires."""
    tokens = _aspect_tokens("Среди них: .ts, .html, .css, .scss")
    # Dot-prefixed form.
    assert ".ts" in tokens
    assert ".html" in tokens
    assert ".css" in tokens
    assert ".scss" in tokens
    # Bare-name form (so overlap with TZ «HTML и CSS» fires).
    assert "html" in tokens
    assert "css" in tokens
    assert "scss" in tokens
    assert "ts" in tokens


def test_html_css_overlap_between_tz_bare_and_pz_extension_form() -> None:
    """Polyakov 0.27::sent2 reproduction: TZ «Также используется HTML
    и CSS» vs PZ «.ts, .html, .css, .scss». Old `_aspect_tokens`
    extracted nothing from the PZ side because of the leading dot →
    overlap was 0. With R2 the overlap is non-trivial."""
    overlap = _boostable_aspect_overlap(
        "Также используется язык гипертекстовой разметки HTML и язык "
        "описания внешнего вида CSS.",
        "Среди них: .ts, .html, .css, .scss (последний в случае "
        "необходимости).",
    )
    assert overlap > 0.0, (
        "TZ HTML+CSS vs PZ .html+.css must produce non-zero boostable "
        "aspect overlap after R2"
    )


# ── Aspect families ────────────────────────────────────────────────


def test_dspace_hierarchy_aspect_family_lights_up() -> None:
    """TZ requirement «многоуровневая иерархия данных репозитория»
    vs PZ «Иерархия данных DSpace построена на сообществах и
    коллекциях» — both touch the dspace_hierarchy family. The
    aspect token must be extracted on both sides for overlap to fire."""
    tz = "Наличие интерфейса для ориентирования по многоуровневой системе иерархии данных репозитория."
    pz = "Иерархия данных DSpace построена на сообществах и коллекциях."
    tz_tokens = _aspect_tokens(tz)
    pz_tokens = _aspect_tokens(pz)
    assert "dspace_hierarchy" in tz_tokens, tz_tokens
    assert "dspace_hierarchy" in pz_tokens, pz_tokens


def test_user_roles_aspect_family_matches() -> None:
    """TZ «Ограничение доступа в соответствии с ролью» vs PZ
    «Доступны роли: анонимный пользователь, зарегистрированный
    пользователь, автор, ревьюер, редактор, администратор» — both
    light up the user_roles family."""
    tz_tokens = _aspect_tokens(
        "Ограничение доступа в соответствии с ролью."
    )
    pz_tokens = _aspect_tokens(
        "Доступны роли: анонимный пользователь, зарегистрированный "
        "пользователь, автор, ревьюер, редактор, администратор."
    )
    assert "user_roles" in tz_tokens
    assert "user_roles" in pz_tokens


def test_container_docker_family_matches() -> None:
    """TZ «Docker для развертывания» vs PZ «Серверная часть
    запускается в Docker-контейнере»."""
    tz_tokens = _aspect_tokens("Docker для развертывания серверной части приложения.")
    pz_tokens = _aspect_tokens("Серверная часть запускается в Docker-контейнере.")
    assert "container_docker" in tz_tokens
    assert "container_docker" in pz_tokens


def test_vcs_git_family_matches() -> None:
    """TZ «Система контроля версий Git» vs PZ «Исходный код хранится
    в Git-репозитории»."""
    tz_tokens = _aspect_tokens("Система контроля версий Git.")
    pz_tokens = _aspect_tokens(
        "Исходный код хранится в Git-репозитории, размещённом на GitHub."
    )
    assert "vcs_git" in tz_tokens
    assert "vcs_git" in pz_tokens


def test_spa_arch_family_matches() -> None:
    """SPA single-page architecture overlap."""
    tz_tokens = _aspect_tokens("Клиентская часть реализована как SPA.")
    pz_tokens = _aspect_tokens(
        "Клиентское приложение построено по single-page архитектуре."
    )
    assert "spa_arch" in tz_tokens
    assert "spa_arch" in pz_tokens


# ── Section-title boost ────────────────────────────────────────────


def _pz_unit(section_title: str = "") -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pz",
        target_doc_role="pz",
        unit_type=CoverageUnitType.PARAGRAPH,
        text="implementation paragraph text",
        normalized_text="implementation paragraph text",
        metadata={"section_title": section_title} if section_title else {},
    )


def _req(req_type: RequirementType = RequirementType.OTHER) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Test req.",
        normalized_text="test req.",
        requirement_type=req_type,
    )


def test_pz_narrative_section_titles_match_regex() -> None:
    """Sanity-check the regex against canonical narrative-PZ section
    titles seen in real ВКР documents."""
    canonical_titles = [
        "Архитектура клиентской части",
        "Архитектура серверной части",
        "Структура клиентской части",
        "Обоснование средств разработки",
        "Обоснование выбора технологий",
        "Реализация",
        "Реализация клиентской части",
        "Развёртывание системы",
        "Развертывание приложения",
        "Иерархия данных DSpace",
        "Сценарии использования",
        "Сценарии работы пользователя",
        "Ролевая модель",
        "Роли пользователей",
        "Клиент-серверное взаимодействие",
        "Пользовательский интерфейс",
        "Описание реализации",
        "Описание компонентов",
    ]
    for title in canonical_titles:
        assert _PZ_NARRATIVE_SECTION_RE.search(title), (
            f"narrative-PZ section title not recognised: {title!r}"
        )


def test_non_narrative_section_titles_dont_match() -> None:
    """The regex must NOT match titles that aren't narrative-PZ
    implementation sections (otherwise we'd boost noise)."""
    not_narrative = [
        "Введение",
        "Существующие аналоги",
        "Сравнительный анализ",
        "Список использованных источников",
        "Заключение",
        # Should NOT match — looks like a section but is bibliography.
        "Список литературы",
    ]
    for title in not_narrative:
        assert not _PZ_NARRATIVE_SECTION_RE.search(title), (
            f"non-narrative title incorrectly matched: {title!r}"
        )


def test_section_prior_pz_narrative_section_returns_1_0() -> None:
    """End-to-end: a PZ unit whose section_title matches the
    narrative-section regex gets section_prior=1.0 regardless of
    requirement type."""
    unit = _pz_unit(section_title="Архитектура клиентской части")
    # Use a type that's NOT in _PZ_PREFERRED to confirm the section
    # boost is independent of the type fallback.
    req = _req(RequirementType.ENVIRONMENT_REQUIREMENT)
    assert _section_prior(req, unit) == 1.0


def test_section_prior_pz_unrelated_section_falls_through() -> None:
    """Sanity: a PZ unit with a non-narrative section title and a
    non-preferred type returns 0 (no spurious boost)."""
    unit = _pz_unit(section_title="Введение")
    req = _req(RequirementType.ENVIRONMENT_REQUIREMENT)
    assert _section_prior(req, unit) == 0.0


def test_section_prior_pz_preferred_type_still_boosted() -> None:
    """Sanity: preferred-type fallback still works when the section
    title doesn't match the narrative regex."""
    unit = _pz_unit(section_title="Введение")
    req = _req(RequirementType.ARCHITECTURE_IMPLEMENTATION)
    assert _section_prior(req, unit) == 1.0
