"""
Rule-based requirement classifier.

Maps a (text, section_title) pair to a RequirementType. Used downstream
to drive type-aware retrieval, CONFLICT same-aspect validation,
applicability decisions and severity scoring.

Cascade:
    1. section_title — most reliable: GOST templates use stable headings,
       and a section-level signal correctly classifies even text that
       could read multiple ways in isolation.
    2. text patterns — when the section says nothing distinctive (or no
       title was supplied), fall back to lexical / structural signals on
       the requirement text itself.

Non-goals:
    * No ML classifier here — patterns are deliberately tight so they
      don't drift with corpus changes. The cost of a wrong type is high
      (it changes applicability + severity), so we prefer "OTHER" over
      a guess.
    * No hard-coded matches against specific package texts.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from app.domain.c_quality_enums import RequirementType


# ── Section-title rules (highest priority) ─────────────────────────────
#
# Order matters within the list — earlier matches win. Specific patterns
# (PROCESS / ECONOMIC / DELIVERY) come before broad ones (FUNCTIONAL).

_SECTION_TYPE_RULES: List[Tuple[re.Pattern, RequirementType]] = [
    # Process / stages-and-phases tables — never a coverage requirement.
    (re.compile(r"стади[ия]\s+и\s+этап", re.I), RequirementType.PROCESS_REQUIREMENT),

    # Economic justification / market need.
    (re.compile(
        r"технико-?\s*эконом|"
        r"предполагаем\w+\s+потребн|"
        r"экономическ\w+\s+преимущ|"
        r"сущест(?:вующи)?\w*\s+решени|"
        r"актуальн\w+\s+(?:разработк|проект)",
        re.I,
    ), RequirementType.ECONOMIC_OR_NEED),

    # Delivery / submission / signatures.
    (re.compile(
        r"специальн\w+\s+требован\w+\s+к\s+программн|"
        r"транспортирован\w+\s+и\s+хранени|"
        r"требован\w+\s+к\s+транспортирован",
        re.I,
    ), RequirementType.DELIVERY_REQUIREMENT),

    # Environment / climate / hardware bench.
    (re.compile(
        r"климатическ\w+\s+услов|"
        r"услов\w+\s+эксплуатац|"
        r"техническ\w+\s+средств",
        re.I,
    ), RequirementType.ENVIRONMENT_REQUIREMENT),

    # Performance — explicit time/throughput section.
    (re.compile(
        r"времен\w+\s+характеристик|"
        r"требован\w+\s+к\s+быстродействи|"
        r"производительн",
        re.I,
    ), RequirementType.PERFORMANCE),

    # Reliability — устойчивость, восстановление, надежность.
    (re.compile(
        r"обеспеч\w+\s+устойчив|"
        r"восстановлен\w+\s+после\s+отказ|"
        r"надежност|"
        r"\bотказ\w+\s+(?:оборудован|програм)",
        re.I,
    ), RequirementType.RELIABILITY),

    # Security — access control, prohibitions, injections.
    (re.compile(
        r"безопасност|"
        r"разграничен\w+\s+доступ|"
        r"контрол\w+\s+входн|"
        r"\bинъекци|"
        r"\bxss\b|"
        r"внедрен\w+\s+код",
        re.I,
    ), RequirementType.SECURITY),

    # Interface / UX / mockups.
    (re.compile(
        r"требован\w+\s+к\s+интерфейс|"
        r"пользовательск\w+\s+интерфейс|"
        r"требован\w+\s+к\s+ui|"
        r"\bмакет\w+\s+интерфейс",
        re.I,
    ), RequirementType.INTERFACE),

    # Architecture / implementation stack.
    (re.compile(
        r"исходн\w+\s+код|"
        r"язык\w+\s+программир|"
        r"средств\w+\s+разработ|"
        r"архитектур|"
        r"структур\w+\s+(?:проект|компонент)",
        re.I,
    ), RequirementType.ARCHITECTURE_IMPLEMENTATION),

    # Data IO — input/output organisation.
    (re.compile(
        r"организац\w+\s+входн|"
        r"организац\w+\s+выходн|"
        r"\bвходн\w+\s+данн|"
        r"\bвыходн\w+\s+данн|"
        r"контрол\w+\s+выходн",
        re.I,
    ), RequirementType.DATA_IO),

    # Documentation — composition / GOST conformance.
    (re.compile(
        r"требован\w+\s+к\s+программн\w+\s+документ|"
        r"состав\s+программн\w+\s+документ|"
        r"требован\w+\s+к\s+документ",
        re.I,
    ), RequirementType.DOCUMENTATION_REQUIREMENT),

    # Functional — broad, last so specific patterns win.
    (re.compile(
        r"требован\w+\s+к\s+функционал|"
        r"функциональн\w+\s+характеристик|"
        r"состав\s+(?:выполн|функц)|"
        r"функц\w+\s+(?:программ|систем)",
        re.I,
    ), RequirementType.FUNCTIONAL),
]


# ── Text-level rules (used when section title is uninformative) ─────────

_TEXT_TYPE_RULES: List[Tuple[re.Pattern, RequirementType]] = [
    # Delivery — explicit LMS / Antiplagiat / archive submission / deadline.
    (re.compile(
        r"\bsmartlms\b|"
        r"\bантиплагиат\b|"
        r"загруж\w+\s+в\s+систем\w+\s+антиплагиат|"
        r"загруж\w+\s+в\s+(?:личн\w+\s+кабинет|smartlms)|"
        r"\bархив\w+\s+проект|"
        r"подпис\w+\s+(?:руководител|исполнител)|"
        r"за\s+\d+\s+дн\w+\s+до\s+(?:начала|защит|представл)",
        re.I,
    ), RequirementType.DELIVERY_REQUIREMENT),

    # Process — stages / coordination.
    (re.compile(
        r"^\s*стади\w+\s+(?:и\s+)?этап|"
        r"порядок\s+согласован|"
        r"организацион\w+\s+действ",
        re.I,
    ), RequirementType.PROCESS_REQUIREMENT),

    # Economic / need.
    (re.compile(
        r"технико-?\s*эконом|"
        r"экономическ\w+\s+преимущ|"
        r"предполагаем\w+\s+потребност",
        re.I,
    ), RequirementType.ECONOMIC_OR_NEED),

    # Performance — explicit metrics.
    (re.compile(
        r"\bвремя\s+отклик|"
        r"\bвремя\s+ответ|"
        r"времен\w+\s+характеристик|"
        r"\bRPS\b|\bпропускн\w+\s+способ|"
        r"\bне\s+должн\w+\s+превышат\w*\s+\d+\s*(?:сек|секунд|мс)|"
        r"производительн",
        re.I,
    ), RequirementType.PERFORMANCE),

    # Reliability — abnormal termination, recovery.
    (re.compile(
        r"аварийн\w+\s+заверш|"
        r"восстановлен\w+\s+после\s+отказ|"
        r"устойчив\w+\s+к\s+(?:ошибк|отказ|сбо)|"
        r"\bотказ\w+\s+систем|"
        r"непрерывн\w+\s+работ",
        re.I,
    ), RequirementType.RELIABILITY),

    # Security — injection, XSS, access control.
    (re.compile(
        r"внедрен\w+\s+код|"
        r"\bxss\b|"
        r"\bsql[\s-]?injection|"
        r"\bинъекци|"
        r"разграничен\w+\s+доступ|"
        r"роле\w+\s+модел|"
        r"\bаутентифик|\bавторизац|"
        r"\bтокен\w*\s+доступ",
        re.I,
    ), RequirementType.SECURITY),

    # Architecture — concrete tech stack.
    (re.compile(
        # Web/UI frameworks
        r"\b(?:typescript|javascript|angular|react|vue\.?js|vue|svelte|next\.?js|nuxt|spa)\b|"
        # Backend frameworks
        r"\b(?:spring|django|flask|fastapi|fast-api|laravel|express|nestjs|gin|fiber)\b|"
        # Programming languages (in tech-stack context — bare "python" needs the word boundary)
        r"\b(?:python|golang|kotlin|scala|rust|ruby on rails|rails)\b|"
        # Runtimes / build / CI
        r"\b(?:node\.?js|deno|bun|github(?:\s+actions)?|gitlab(?:\s+ci)?|jenkins|circleci)\b|"
        # Containers / orchestration / infra
        r"\b(?:docker|docker-?compose|kubernetes|k8s|helm|terraform|ansible|nginx)\b|"
        # Auth / identity
        r"\bkeycloak\b|\bauth0\b|\boauth2?\b|"
        # Databases / queues
        r"\b(?:postgresql|postgres|mysql|mariadb|mongodb|redis|kafka|rabbitmq|elasticsearch)\b|"
        # Generic role markers + tech: "backend на X", "frontend на Y",
        # "реализован[ао] на|с использованием X" — strong architecture signal.
        r"\b(?:backend|back-?end|frontend|front-?end|серверн\w+\s+часть|"
        r"клиентск\w+\s+часть)\b|"
        r"реализован\w+\s+(?:на|с\s+использованием|при\s+помощи)\s+\w+|"
        r"использован\w+\s+(?:фреймворк|библиотек|стек\w+\s+технологий)",
        re.I,
    ), RequirementType.ARCHITECTURE_IMPLEMENTATION),

    # Data IO — REST / JSON / formats / API.
    (re.compile(
        r"\brest\b|\brest\s*api\b|"
        r"\bjson\b|\bxml\b|\bcsv\b|\byaml\b|"
        r"формат\w+\s+(?:данн|обмен|json|xml)|"
        r"\bвходн\w+\s+данн|\bвыходн\w+\s+данн",
        re.I,
    ), RequirementType.DATA_IO),

    # Interface — Figma, UI, UX, mockup.
    (re.compile(
        r"\bfigma\b|"
        r"макет\w+\s+интерфейс|"
        r"стилизац\w+\s+интерфейс|"
        r"фирмен\w+\s+(?:дизайн|стил)|"
        r"пользовательск\w+\s+интерфейс|"
        r"\bui\b|\bux\b",
        re.I,
    ), RequirementType.INTERFACE),

    # Documentation — GOST 19 / docs composition.
    (re.compile(
        r"\bгост\s*19\b|"
        r"программн\w+\s+документац|"
        r"оформлен\w+\s+документ",
        re.I,
    ), RequirementType.DOCUMENTATION_REQUIREMENT),

    # Environment — climate / OS / browser / hardware.
    (re.compile(
        r"климатическ\w+\s+услов|"
        r"операционн\w+\s+систем|"
        r"\bбраузер\w*|"
        r"\bмонитор\w+\s+(?:разрешен|разреш)",
        re.I,
    ), RequirementType.ENVIRONMENT_REQUIREMENT),

    # Logging / audit — kept as a separate axis because legacy callers
    # rely on it. Rules placed before broad FUNCTIONAL so "хранить
    # журнал" classifies as LOGGING, not FUNCTIONAL.
    (re.compile(
        r"\bжурнал\w*\s+(?:событ|событ|операц|действ)|"
        r"\bвести\s+журнал|"
        r"\bаудит\w*|"
        r"\blogging\b|"
        r"\bхранит\w*\s+журнал",
        re.I,
    ), RequirementType.LOGGING),

    # Storage — retention / archival.
    (re.compile(
        r"\bхранит\w*\s+\w*\s*(?:\d+\s+)?(?:дн|сут|месяц|лет|год)|"
        r"архивн\w+\s+хранен|"
        r"срок\w*\s+хранен",
        re.I,
    ), RequirementType.STORAGE),

    # Functional — broad, last (modal verbs of capability + verbs of doing).
    (re.compile(
        r"регистрац|"
        r"\bпоиск\w*|"
        r"фильтрац|"
        r"\bзагрузк\w+\s+(?:файл|публикац|данн)|"
        r"\bworkflow\b|"
        r"\bcrud\b|"
        r"\bредактирован\w+\s+(?:объект|метаданн|публикац)|"
        r"\bпубликац\w+\s+(?:материал|объект)",
        re.I,
    ), RequirementType.FUNCTIONAL),
]


def classify_requirement(text: str, section_title: str = "") -> RequirementType:
    """Classify a requirement into one of the typed buckets.

    Section title takes precedence — GOST templates have stable
    headings and that's the most reliable signal. Text patterns are
    a fallback. Returns OTHER when no rule fires (better to abstain
    than mis-classify and propagate the wrong applicability).
    """
    t = (text or "").strip()
    s = (section_title or "").strip()

    if s:
        for pat, rt in _SECTION_TYPE_RULES:
            if pat.search(s):
                return rt

    for pat, rt in _TEXT_TYPE_RULES:
        if pat.search(t):
            return rt

    return RequirementType.OTHER
