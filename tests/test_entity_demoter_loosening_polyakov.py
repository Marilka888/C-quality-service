"""
Polyakov-regression (2026-05-10): the Rule-5 entity-overlap demoter
in `verify_pairs` was too aggressive on the Polyakov package, turning
several legitimate LLM-PARTIAL verdicts into MISSING:

  * 0.17::sent1 «Необходимо запустить серверную часть и базу данных…»
    vs PMI «Для недопущения отказа программы … обработка некорректных
    данных…» — judge PARTIAL conf 0.7, demoted to IRRELEVANT because
    entity_overlap=0 and lex_jac=0.04. Both sides have only 2-3
    extracted entities, but the topics overlap genuinely.

  * 0.27::sent1 «Исходные коды программы должны быть написаны на
    языке программирования TypeScript с использованием библиотеки
    Angular» vs PZ «.ts, .html, .css, .scss» — judge PARTIAL conf 0.7,
    demoted because entity_overlap=0 and lex_jac=0. Both clearly point
    to the same client-side tech stack, but neither lex nor entity
    bridges the syntactic gap.

Two surgical fixes:

  1. Demoter requires `len(req.entities) >= 3` (was `>= 2`). When the
     requirement is genuinely entity-rich (3+ named heads) and the
     evidence has zero overlap, that's diagnostic. With 2 entities the
     extractor often missed one — too noisy a signal to demote on.

  2. New preservation path `_tech_stack_co_occurrence(req, unit)`: if
     both texts mention identifiers from the same tech-stack family
     (TypeScript/Angular/REST/JSON/Docker/Git/SPA/DSpace/Figma/HTML/
     CSS/SCSS/extensions), preserve the LLM-PARTIAL verdict regardless
     of entity_overlap.

Both changes are conservative — they REDUCE demoter firing, not
expand it. The aggregator still requires retrieval ≥ medium for any
PARTIAL to be accepted; spurious PARTIAL would be filtered there.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.verify_pairs import (
    PairVerifier,
    _tech_stack_co_occurrence,
)
from app.domain.c_quality_enums import (
    CoverageUnitType,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)


# ── _tech_stack_co_occurrence ──────────────────────────────────────


@pytest.mark.parametrize("req_text, unit_text, family", [
    # Polyakov 0.27::sent1 — extension vs full name, mixed languages.
    (
        "Исходные коды программы должны быть написаны на TypeScript с Angular.",
        "Среди них: .ts, .html, .css, .scss.",
        "client_lang",
    ),
    # REST + JSON family (TZ data_io vs PZ implementation note).
    (
        "Все входные данные отправляются через REST API в формате JSON.",
        "Сервер обрабатывает запросы по REST и отдаёт ответы в JSON.",
        "transport",
    ),
    # Docker family.
    (
        "Docker для развертывания серверной части приложения.",
        "Серверная часть упакована в Docker-контейнер.",
        "container",
    ),
    # Git VCS family.
    (
        "Система контроля версий Git.",
        "Исходный код хранится в Git-репозитории.",
        "vcs",
    ),
    # Figma design-tool family.
    (
        "Макет интерфейса должен быть разработан в Figma.",
        "Пользовательский интерфейс спроектирован в Figma.",
        "design_tool",
    ),
])
def test_tech_stack_match_recognised(req_text: str, unit_text: str, family: str) -> None:
    assert _tech_stack_co_occurrence(req_text, unit_text) == family


@pytest.mark.parametrize("req_text, unit_text", [
    # Different families — no co-occurrence.
    ("Исходные коды на TypeScript.", "Развертывание в Docker."),
    # Family present in only one side.
    ("Используется Angular.", "Запуск серверной части и базы данных."),
    # Empty texts.
    ("", "Использует TypeScript."),
    ("Использует TypeScript.", ""),
])
def test_tech_stack_no_match_returns_none(req_text: str, unit_text: str) -> None:
    assert _tech_stack_co_occurrence(req_text, unit_text) is None


# ── End-to-end: PairVerifier preserves PARTIAL on tech-stack match ─


def _req(text: str, entities: list[str]) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text=text,
        normalized_text=text.lower(),
        entities=entities,
        requirement_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
    )


def _unit(text: str, entities: list[str]) -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pz",
        target_doc_role="pz",
        unit_type=CoverageUnitType.PARAGRAPH,
        text=text,
        normalized_text=text.lower(),
        entities=entities,
    )


def _partial_judgment(conf: float = 0.7) -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pz",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=conf,
        explanation="Test PARTIAL.",
    )


def test_polyakov_typescript_angular_preserved_via_tech_stack(monkeypatch) -> None:
    """0.27::sent1 reproduction: req mentions TypeScript+Angular,
    evidence mentions .ts files. Entity_overlap=0, lex_jac low, judge
    PARTIAL conf 0.7 — old code demoted to IRRELEVANT, new code
    preserves PARTIAL via tech_stack family `client_lang`."""
    req = _req(
        "Исходные коды программы должны быть написаны на языке "
        "программирования TypeScript с использованием библиотеки Angular.",
        entities=["TypeScript", "Angular", "исходные коды программы"],
    )
    unit = _unit(
        "Среди них: .ts, .html, .css, .scss (последний в случае "
        "необходимости), каждый из которых имеет одинаковое название.",
        entities=["файлы", "директория"],
    )
    # Force the unit to have 0 extracted entities — that is the only
    # path under R1+ that exercises the entity-overlap demoter, and
    # therefore the only path where `preserve_partial_tech_stack`
    # would fire as the override. With a non-empty unit-entities the
    # demoter is skipped entirely (which is also acceptable but
    # doesn't exercise the tech-stack path).
    unit_zero_ent = _unit(
        "Среди них: .ts, .html, .css, .scss (последний в случае "
        "необходимости), каждый из которых имеет одинаковое название.",
        entities=[],
    )
    judgment = _partial_judgment(conf=0.7)
    out = PairVerifier().verify(judgment, req, unit_zero_ent)

    assert out.rule_adjusted_label == LLMLabel.PARTIAL, (
        f"PARTIAL must be preserved via tech-stack co-occurrence, "
        f"got {out.rule_adjusted_label}; explanation={out.explanation!r}"
    )
    assert any(
        "preserve_partial_tech_stack" in (a or "")
        for a in out.verifier_actions
    ), out.verifier_actions
    assert "tech-stack co-occurrence" in out.explanation


def test_polyakov_0_17_sent1_no_longer_demoted_via_unit_entities() -> None:
    """Polyakov 0.17::sent1 reproduction shape: req has 3+ entities
    (above the threshold) BUT evidence ALSO has entities — under the
    R1+ tightening (`len(unit.entities) == 0`), the demoter no longer
    fires when the evidence is non-empty. Old: demoted IRRELEVANT.
    New: stays PARTIAL — judge verdict survives."""
    req = _req(
        "Необходимо запустить серверную часть и базу данных в целях "
        "получения информации о проектах и во избежание утерянных данных.",
        entities=["серверная часть", "база данных", "информация о проектах",
                  "утерянные данные"],  # 4 entities — passes ≥3
    )
    unit = _unit(
        "Для недопущения отказа программы предусмотрена обработка "
        "некорректных данных оператором.",
        entities=["отказ программы", "обработка данных",
                  "оператор"],  # 3 entities — fails == 0 → demoter skipped
    )
    judgment = _partial_judgment(conf=0.7)
    out = PairVerifier().verify(judgment, req, unit)
    assert out.rule_adjusted_label == LLMLabel.PARTIAL, (
        f"unit with non-zero entities must NOT trigger demoter; "
        f"got {out.rule_adjusted_label}; explanation={out.explanation!r}"
    )


def test_demoter_still_fires_on_zero_unit_entities() -> None:
    """Sanity: demoter still fires when req is entity-rich (≥3) AND
    evidence has ZERO extracted entities — that's the only case where
    entity_overlap=0 is genuinely diagnostic (truly empty evidence,
    not extractor noise)."""
    req = _req(
        "Регистрация, авторизация и аутентификация пользователей системы.",
        entities=["регистрация", "авторизация", "аутентификация", "пользователи"],
    )
    unit = _unit(
        "Текст без распознанных именованных сущностей.",
        entities=[],  # zero — passes the demoter gate
    )
    judgment = _partial_judgment(conf=0.7)
    out = PairVerifier().verify(judgment, req, unit)
    assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
        f"req-rich + unit-empty must still demote; "
        f"got {out.rule_adjusted_label}; explanation={out.explanation!r}"
    )


def test_high_confidence_85_still_protects() -> None:
    """Sanity: judge confidence ≥ 0.85 protects regardless of all the
    other gates — preserved from original behaviour."""
    req = _req(
        "Регистрация, авторизация и аутентификация пользователей системы.",
        entities=["регистрация", "авторизация", "аутентификация", "пользователи"],
    )
    unit = _unit(
        "DSpace анализ конкурентов.",
        entities=["DSpace", "анализ", "конкуренты"],
    )
    judgment = _partial_judgment(conf=0.90)
    out = PairVerifier().verify(judgment, req, unit)
    assert out.rule_adjusted_label == LLMLabel.PARTIAL, (
        f"conf ≥ 0.85 should protect from demotion; got {out.rule_adjusted_label}"
    )
