"""
Polyakov-regression: shared-subject-head PARTIAL path in
DisabledCoverageJudge.

Real-data observation on the Polyakov demo: many TZ↔PMI pairs that
SHOULD have been COVERED/PARTIAL ended up MISSING because:
  * the requirement uses a be-class verb («представлять», «являться»,
    «обладать»), which is not in _ACTION_VERB_RE — verb_match=False;
  * the PMI side is a long section_window («Требования к функциональным
    характеристикам. Клиентская часть приложения должна обеспечивать
    возможность выполнения следующих функций: …») which dilutes
    Jaccard below the 0.20 P9 threshold;
  * but BOTH texts have the same subject head («Клиентская часть»)
    and BOTH carry a normative modal («должна»).

The new PX_SUBJECT path catches exactly this pattern: shared canonical
subject head + both texts normative-modal-bearing + lex ≥ 0.10 → PARTIAL.
The lex floor protects against pairs that share only the subject and
nothing topical.
"""
from __future__ import annotations

import pytest

from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, RequirementUnit
from app.infrastructure.llm.disabled_coverage_judge import (
    DisabledCoverageJudge,
    _has_normative_modal,
    _subject_head,
)


# ── _subject_head helper ─────────────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("Клиентская часть должна обеспечивать выполнение функций.", "client_part"),
    ("клиентская часть приложения должна работать.", "client_part"),
    ("Серверная часть приложения должна корректно отвечать.", "server_part"),
    ("Программа должна работать без сбоев.", "program"),
    ("Программа и методика испытаний.", "program"),
    ("Система должна обеспечивать поиск.", "system"),
    ("систему необходимо защитить.", "system"),
    ("Подсистема должна предоставлять API.", "subsystem"),
    ("Сервис должен использовать TLS.", "service"),
    ("Модуль должен импортировать данные.", "module"),
    ("Пользовательский интерфейс должен быть удобным.", "user_interface"),
    ("Программный интерфейс должен соответствовать REST.", "prog_interface"),
    ("Приложение должно работать на устройстве.", "application"),
    ("Пользователь должен авторизоваться.", "user"),
    ("Администратор должен иметь полный доступ.", "admin"),
    ("API должен возвращать JSON.", "api"),
    ("api должен поддерживать пагинацию.", "api"),
])
def test_subject_head_canonical(text: str, expected: str) -> None:
    assert _subject_head(text) == expected, (
        f"expected subject head {expected!r} for {text!r}, got {_subject_head(text)!r}"
    )


@pytest.mark.parametrize("text", [
    "Время отклика приложения не должно превышать 3 секунд.",
    "Функциональные требования к программе.",
    "При проверке поиска выполняются запросы.",
    "Также используется HTML.",
    "",
])
def test_subject_head_returns_none_when_no_anchor(text: str) -> None:
    assert _subject_head(text) is None


# ── _has_normative_modal helper ──────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("Система должна работать.", True),
    ("Программа должно завершаться.", True),
    ("Сервис должны поддерживать.", True),
    ("Пользователь должен войти.", True),
    ("Сервис обязан использовать TLS.", True),
    ("Необходимо запустить серверную часть.", True),
    ("Следует проверить корректность.", True),
    ("Система имеет три режима.", False),
    ("Программа была разработана.", False),
])
def test_normative_modal_detection(text: str, expected: bool) -> None:
    assert _has_normative_modal(text) is expected


# ── End-to-end: Polyakov-style pair ──────────────────────────────────


def test_polyakov_client_part_subject_anchor_partial() -> None:
    """The exact Polyakov 0.10::sent1 vs PMI section_window case:
    «Клиентская часть должна представлять из себя пользовательский
    интерфейс…» vs «Клиентская часть приложения должна обеспечивать
    возможность выполнения следующих функций: …». No verb in
    _ACTION_VERB_RE matches on the TZ side («представлять» not in
    whitelist) and the PMI side is long, so lex < 0.20 — but the
    subject head «client_part» matches and both are modal-bearing,
    so the subject-anchor path produces PARTIAL."""
    req = RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text=(
            "Клиентская часть должна представлять из себя пользовательский "
            "интерфейс для осуществления просмотра проектов, загрузки файлов "
            "в и из системы."
        ),
        normalized_text=(
            "клиентская часть должна представлять из себя пользовательский "
            "интерфейс для осуществления просмотра проектов, загрузки файлов "
            "в и из системы."
        ),
    )
    unit = CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        text=(
            "Требования к функциональным характеристикам. Клиентская часть "
            "приложения должна обеспечивать возможность выполнения следующих "
            "функций. Система должна обеспечивать работу с научными "
            "проектами, включая создание проектов, их редактирование и "
            "управление метаданными. Система должна обеспечивать загрузку."
        ),
        normalized_text=(
            "требования к функциональным характеристикам. клиентская часть "
            "приложения должна обеспечивать возможность выполнения следующих "
            "функций. система должна обеспечивать работу с научными "
            "проектами, включая создание проектов, их редактирование и "
            "управление метаданными. система должна обеспечивать загрузку."
        ),
    )
    judgment = DisabledCoverageJudge().judge(req, unit)
    assert judgment.llm_label == LLMLabel.PARTIAL, (
        f"expected PARTIAL via subject-anchor path; got "
        f"{judgment.llm_label}: {judgment.explanation}"
    )
    aspects_str = ",".join(judgment.matched_aspects)
    assert "shared_subject:client_part" in aspects_str
    assert "both_normative_modal" in aspects_str


def test_subject_anchor_no_modal_does_not_fire() -> None:
    """Sanity: shared subject head BUT one side has no normative
    modal — must NOT trigger the subject-anchor path. Otherwise
    descriptive prose («Система имеет три режима работы.») would
    falsely match against requirements about that subject."""
    req = RequirementUnit(
        req_id="r1", source_document_id="tz",
        text="Система должна обеспечивать аутентификацию.",
        normalized_text="система должна обеспечивать аутентификацию.",
    )
    unit = CoverageUnit(
        unit_id="u1", target_document_id="doc-pmi", target_doc_role="pmi",
        text="Система имеет три режима работы.",
        normalized_text="система имеет три режима работы.",
    )
    judgment = DisabledCoverageJudge().judge(req, unit)
    # Should NOT match the subject-anchor path (unit has no modal).
    aspects_str = ",".join(judgment.matched_aspects)
    assert "shared_subject" not in aspects_str


def test_subject_anchor_different_subjects_does_not_fire() -> None:
    """Sanity: different subjects must not collapse via the canon map."""
    req = RequirementUnit(
        req_id="r1", source_document_id="tz",
        text="Сервер должен принимать REST-запросы.",
        normalized_text="сервер должен принимать rest-запросы.",
    )
    unit = CoverageUnit(
        unit_id="u1", target_document_id="doc-pmi", target_doc_role="pmi",
        text="Пользователь должен авторизоваться через форму.",
        normalized_text="пользователь должен авторизоваться через форму.",
    )
    judgment = DisabledCoverageJudge().judge(req, unit)
    aspects_str = ",".join(judgment.matched_aspects)
    assert "shared_subject" not in aspects_str
