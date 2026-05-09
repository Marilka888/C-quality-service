"""
Polyakov-regression: DisabledCoverageJudge must recognise broader
PMI verification-step phrasings.

Symptom on Polyakov 5th demo run: LLM (Ollama qwen2.5:7b) timed out
on 56 of ~62 pairs (`LLM_UNAVAILABLE: judge backend fell back to
disabled mode for 56 pair(s)`). The DisabledCoverageJudge ran on
those pairs and judged most as IRRELEVANT despite the retriever
having scored them ≥ 0.45 — because the disabled judge ignores
retrieval_score and re-evaluates from scratch using lex+verb rules.

Concrete miss on req 0.11::sent10 «Веб-интерфейс для поиска,
загрузки и просмотра научных материалов» vs PMI unit «Для проверки
поиска выполняются запросы к системе поиска»:
  * lex = 0.11 (only «поиск» shared)
  * no _ACTION_VERB in either text (req is nominal, unit verb is
    «выполняются» which isn't whitelisted)
  * the unit is a textbook PMI verification step but the
    _VERIFICATION_UNIT_RE pattern «для\\s+проверки\\s+(пункта|
    требования|данного|указанного)» didn't match because «поиска»
    is the SUBJECT noun, not a meta-token.

The fix loosens the regex: «для\\s+проверки\\s+\\w{3,}» (any noun)
plus three more unambiguous PMI verification patterns («при
проверке X», «проверка X», «выполняются запросы/тесты/проверки/
испытания/шаги/операции»). With the new pattern the unit hits the
PX_VERIFY path → PARTIAL with low confidence, instead of falling
through to IRRELEVANT.
"""
from __future__ import annotations

import pytest

from app.infrastructure.llm.disabled_coverage_judge import (
    _is_verification_unit,
    DisabledCoverageJudge,
)
from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, RequirementUnit


# ── Polyakov-style verification phrasings ────────────────────────────


@pytest.mark.parametrize("text", [
    "Для проверки поиска выполняются запросы к системе поиска.",
    "Для проверки авторизации необходимы логин и пароль существующего пользователя.",
    "Для проверки разграничения доступа используются пользователи с ролями.",
    "Для проверки работы фильтра в форму поиска вводится строка.",
    "При проверке загрузки файлов выбирается тестовый документ.",
    "При проверке корректности обработки ошибок отправляется некорректный запрос.",
    "Проверка поискового модуля.",
    "Проверка авторизации пользователя.",
    "Проверка работы разграничения доступа.",
    "Выполняются запросы к серверной части.",
    "Выполняются тесты на сохранение и загрузку метаданных.",
    "Выполняются проверки соответствия выходных данных.",
])
def test_polyakov_verification_unit_recognised(text: str) -> None:
    assert _is_verification_unit(text), (
        f"text should be recognised as a PMI verification step: {text!r}"
    )


# ── Negatives — not verification steps ───────────────────────────────


@pytest.mark.parametrize("text", [
    # Plain requirement statements — NOT verification.
    "Система должна обеспечивать поиск по публикациям.",
    "Программный интерфейс должен быть представлен в виде REST API.",
    # Bare topic mention without verification framing.
    "Поисковый модуль реализован на стороне сервера.",
    # Hardware spec / environment text.
    "Процессор Intel(R) Core(TM) i5-7500 (4 ядра, 3.4 ГГц).",
    "Более 900 ГБ доступного дискового пространства.",
    # Heading-only text.
    "Объект испытаний.",
    # Too-short noun after «проверки» (we required ≥3 chars to avoid noise).
    "Для проверки.",
    # Too-short noun after «проверка».
    "Проверка ОК.",
])
def test_non_verification_text_not_flagged(text: str) -> None:
    assert not _is_verification_unit(text), (
        f"text wrongly flagged as a verification step: {text!r}"
    )


# ── End-to-end via DisabledCoverageJudge ─────────────────────────────


def test_polyakov_search_requirement_against_verify_unit_yields_partial() -> None:
    """The exact Polyakov case: req «Веб-интерфейс для поиска…» vs
    unit «Для проверки поиска выполняются…». Without the regex
    extension this fell through to IRRELEVANT (lex=0.11, no verb
    match); with the extension the verification gate fires and the
    disabled judge returns PARTIAL with the verification rationale."""
    req = RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Веб-интерфейс для поиска, загрузки и просмотра научных материалов.",
        normalized_text="веб-интерфейс для поиска, загрузки и просмотра научных материалов.",
    )
    unit = CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        text="Для проверки поиска выполняются запросы к системе поиска.",
        normalized_text="для проверки поиска выполняются запросы к системе поиска.",
    )
    judgment = DisabledCoverageJudge().judge(req, unit)
    assert judgment.llm_label == LLMLabel.PARTIAL, (
        f"expected PARTIAL via PMI verification gate; got "
        f"{judgment.llm_label} ({judgment.explanation})"
    )
    # The matched_aspects should mention the verification path.
    aspects_str = " ".join(judgment.matched_aspects)
    assert "verification" in aspects_str or "is_verify" in (judgment.explanation or "")
