"""
Stage 3 + 4: hybrid retrieval → top-N shortlist per RequirementUnit.

Score = w_lex * lexical + w_sem * semantic + w_con * constraint_overlap + w_sec * section_prior
All weights come from CoverageRetrievalConfig.

PR-K additions (additive, no contract breaks):
  * Each RetrievedCandidate carries `score_reason` (one-liner explaining
    which signal drove the score) and `evidence_strength` (STRONG /
    MEDIUM / WEAK / NO_EVIDENCE bin).
  * `initial_top_n` controls how many candidates are returned to the
    caller; the AdaptiveCandidateSelector trims further before LLM.
  * Reranker can run unconditionally (mode="always", legacy) or only
    when first-stage signals are weak (mode="conditional"): top1 below
    a threshold, narrow top1-top2 margin, requirement carries numeric
    constraints, or paraphrase indicated by high semantic / low lexical.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from app.application.use_cases.applicability import evidence_strength_from_score
from app.core.config import CoverageRerankerConfig, CoverageRetrievalConfig
from app.core.logging import get_logger
from app.core.text import tokenize_content
from app.domain.c_quality_enums import RequirementType
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.embeddings.base import EmbeddingBackend
from app.infrastructure.reranker.base import NoopReranker, Reranker

logger = get_logger(__name__)

# RequirementTypes that get a section_prior bonus on PMI / PZ docs
_PMI_PREFERRED: Set[RequirementType] = {
    RequirementType.PERFORMANCE,
    RequirementType.LOGGING,
}

# Polyakov-regression: canonical ПМИ test sections per ГОСТ 19.301 —
# units extracted from these sections describe the verification of ТЗ
# requirements and should win retrieval over noise from other ПМИ
# parts (cover page, list of references, climate notes). Without a
# section-name boost the BoW retriever ranks units purely by lexical
# overlap; on Polyakov-class packages legitimate test descriptions
# («Для проверки авторизации необходимы логин и пароль…») score
# 0.30-0.55 — close to noise units like «Windows 10 Pro» from the
# environment section. The boost moves test-section units above
# evidence_floor so the LLM judge actually sees them.
_PMI_TEST_SECTION_RE = re.compile(
    r"требовани\w*\s+к\s+программ"
    r"|метод\w*\s+испытани"
    r"|состав\w*\s+и\s+порядок\s+испытани"
    r"|порядок\s+(?:проведени|испытани|приёмки|приемки)"
    r"|проверк\w+\s+(?:требовани|выполнени)"
    r"|объект\s+испытани"
    r"|цель\s+испытани"
    r"|программа\s+(?:и\s+методика\s+)?испытани",
    re.IGNORECASE | re.UNICODE,
)
_PZ_PREFERRED: Set[RequirementType] = {
    RequirementType.FUNCTIONAL,
    RequirementType.SECURITY,
    RequirementType.INTERFACE,
    RequirementType.STORAGE,
    RequirementType.ARCHITECTURE_IMPLEMENTATION,
    RequirementType.RELIABILITY,
}

# Critical types that always justify the more expensive reranker
# (in conditional mode). Mirrors `_is_critical` in the adaptive selector.
_CRITICAL_TYPES: Set[RequirementType] = {
    RequirementType.SECURITY,
    RequirementType.PERFORMANCE,
    RequirementType.RELIABILITY,
}

# Sections whose content is descriptive/comparative rather than
# implementation-specific (competitor analysis, UI styling notes, …).
# Evidence units from these sections are penalised at retrieval time so
# they don't win over genuine implementation evidence when the BoW score
# happens to be similar (both mention "интерфейс", "система", etc.).
# The penalty is additive with the evidence_floor: a unit that scores
# 0.45 from a competitor-analysis section becomes 0.30 → below the
# default floor of 0.30, so it will never produce a confident verdict.
_NON_IMPL_SECTION_RE = re.compile(
    r"аналог|сравнительн|конкурент|стилизац",
    re.IGNORECASE | re.UNICODE,
)
_NON_IMPL_SECTION_TEXT_RE = re.compile(
    r"repo\.hse|существующ\w*\s+аналог|сравнительн\w*\s+анализ|"
    r"ближайш\w*\s+аналог|на\s+фоне\s+большинства\s+конкурент",
    re.IGNORECASE | re.UNICODE,
)
_FILE_STRUCTURE_TEXT_RE = re.compile(
    r"\.ts,\s*\.html,\s*\.css|одинаковое\s+название|директори[ия]\s+с\s+таким\s+же\s+назв",
    re.IGNORECASE | re.UNICODE,
)
_ADMIN_CAPABILITY_TEXT_RE = re.compile(
    r"администратор\s+[–-]\s+имеет\s+возможность\s+выдавать\s+роли|"
    r"удалять\s+уже\s+принятые\s+исследования|без\s+необходимости\s+проходить\s+модерац",
    re.IGNORECASE | re.UNICODE,
)
_COLLECTION_REPAIR_TEXT_RE = re.compile(
    r"внутрь\s+этой\s+коллекции\s+добавляются|одинокое\s+исследование|"
    r"связать\s+с\s+новой\s+коллекцией",
    re.IGNORECASE | re.UNICODE,
)
_NON_IMPL_SECTION_PENALTY = 0.35
_FILE_STRUCTURE_PENALTY = 0.30
_ADMIN_CAPABILITY_PENALTY = 0.20
_COLLECTION_REPAIR_PENALTY = 0.25

_ASPECT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:[.,]\d+)?", re.UNICODE)
# Polyakov-regression (R2): file-extension form is the canonical PZ
# notation for client-side files («.ts», «.html», «.css», «.scss»),
# but `_ASPECT_TOKEN_RE` requires a leading letter and won't extract
# `.html` as a token. Extract these explicitly so an aspect overlap
# between TZ «HTML и CSS» and PZ «.ts, .html, .css, .scss» is captured.
_FILE_EXT_TOKEN_RE = re.compile(r"\.(?:ts|tsx|html?|css|scss|sass|less|js|jsx)\b", re.IGNORECASE)
_TECH_KEYWORDS = {
    "api", "rest", "json", "xml", "http", "https",
    "typescript", "angular", "react", "vue", "javascript",
    # Polyakov-regression (R2): markup/style stack — TZ requirements
    # use bare names («HTML», «CSS», «SCSS»), PZ uses extension forms
    # («.html», «.css»). Both forms must be aspect-token extractable
    # so overlap can fire even when lex retrieval fails.
    "html", "css", "scss", "sass", "less",
    ".ts", ".tsx", ".html", ".css", ".scss", ".sass", ".js",
    "github", "gitlab", "git", "docker", "figma",
    "sql", "postgresql", "mysql", "mongodb",
    # Polyakov-regression (R2): platform / framework names found in
    # ВКР PZ documents that lex retrieval routinely misses against
    # narrative TZ requirements.
    "dspace", "spa",
}

# Domain anchors that ordinary BoW/embedding retrieval often underweights in
# Russian student documents. They are intentionally stem-ish substrings: the
# goal is recall of the right evidence section, while the LLM/verifier still
# decides COVERED/PARTIAL/MISSING afterwards.
_ASPECT_ALIASES: dict[str, tuple[str, ...]] = {
    "registration": ("регистрац", "зарегистр"),
    "authorization": ("авторизац", "авториз", "разграничен"),
    "authentication": ("аутентификац", "логин", "парол"),
    "project": ("проект", "коллекц", "исследован", "публикац"),
    "upload_download_files": ("загруз", "скач", "файл", "прикреп"),
    "search_filter": ("поиск", "фильтр", "ключев", "метадан", "автор", "дата"),
    "personal_account": ("личн", "кабинет", "профил", "учетн", "учётн"),
    "server_db": ("сервер", "баз", "данн", "backend", "бэкенд"),
    "error_handling": ("ошиб", "неверн", "корректн", "аварийн", "отказ"),
    "interface_mockup": ("интерфейс", "макет", "figma", "ui", "ux", "прототип"),
    "rest_json": ("rest", "api", "json"),
    "github": ("github", "git"),
    "typescript_angular": ("typescript", "angular", ".ts"),
    "performance_time": ("отклик", "секунд", "response", "time"),
    # Polyakov-regression (R2 / narrative-PZ retrieval boost). Without
    # these aliases the PZ-side coverage stays at 0 even when the ВКР
    # clearly mentions the corresponding implementation-stack term.
    "web_markup": (
        "html", ".html", "css", ".css", "scss", ".scss", "разметк", "стилев",
    ),
    "container_docker": ("docker", "docker-compose", "контейнер", "контейнериз"),
    "vcs_git": ("git", "github", "gitlab", "git-репозитор", "версион"),
    "spa_arch": ("spa", "single-page", "одностраничн", "single page"),
    "user_roles": (
        "анонимн", "зарегистрирован", "автор", "ревьюер", "редактор",
        "администратор", "роль", "рол",
    ),
    "dspace_hierarchy": (
        "dspace", "сообществ", "коллекц", "иерархи", "репозитори", "объект",
    ),
    "auth_access": (
        "авторизац", "аутентификац", "разграничен", "доступ", "роль",
        "регистрац",
    ),
    "rest_api": ("rest", "rest api", "rest-api", "api", "endpoint", "эндпоинт"),
}

_BOOSTABLE_ASPECTS: Set[str] = {
    "registration",
    "authorization",
    "authentication",
    "upload_download_files",
    "search_filter",
    "personal_account",
    "error_handling",
    "interface_mockup",
    "rest_json",
    "github",
    "typescript_angular",
    "performance_time",
    # Polyakov-regression (R2): narrative-PZ implementation aspects.
    "web_markup",
    "container_docker",
    "vcs_git",
    "spa_arch",
    "user_roles",
    "dspace_hierarchy",
    "auth_access",
    "rest_api",
}

_PZ_TOPIC_ALIASES: dict[RequirementType, tuple[str, ...]] = {
    RequirementType.ARCHITECTURE_IMPLEMENTATION: (
        "обоснование средств", "средства разработки", "архитектур",
        "развертыван", "развёртыван", "технолог", "github", "docker",
        "typescript", "angular",
        # Polyakov-regression (R2): narrative-PZ implementation
        # vocabulary that appears in ВКР but was missed by retrieval.
        "html", "css", "scss", ".ts", ".html", ".css", "spa",
        "single-page", "клиентск", "серверн", "контейнер",
        "git", "версион", "реализац", "компонент", "модул",
    ),
    RequirementType.DATA_IO: (
        "архитектур", "взаимодейств", "сервер", "rest", "api", "json",
        "входн", "выходн", "данн",
        # R2: narrative-PZ data-flow vocabulary.
        "endpoint", "эндпоинт", "запрос", "ответ", "модель данн",
        "формат", "типизированн", "веб-форм",
    ),
    RequirementType.INTERFACE: (
        "прототип", "figma", "макет", "интерфейс", "стилизац", "ui", "ux",
        # R2: narrative-PZ UI vocabulary.
        "страниц", "форм", "кнопк", "виджет", "компонент",
        "цветов", "палитр", "стил", "дизайн", "браузер",
    ),
    RequirementType.FUNCTIONAL: (
        "пользовательские сценар", "сценар", "интерфейс веб", "страница",
        "проект", "поиск", "фильтр", "скач", "загруз", "файл",
        "регистрац", "авторизац",
        # R2: narrative-PZ functional/scenario vocabulary.
        "роль", "анонимн", "зарегистрирован", "автор", "ревьюер",
        "редактор", "администратор", "коллекц", "иерархи",
        "сообществ", "dspace", "публикац", "исследован",
    ),
    RequirementType.SECURITY: (
        "регистрац", "авторизац", "аутентификац", "доступ", "роль",
        "безопас", "инъекц", "внедрен", "ошиб",
        # R2: narrative-PZ security vocabulary.
        "разграничен", "пароль", "логин", "сессия", "токен",
        "защит", "проверк", "валидац", "санитайз",
    ),
    RequirementType.RELIABILITY: (
        "развертыван", "развёртыван", "сервер", "ошиб", "отказ",
        "коррект", "устойчив", "восстанов",
        # R2: narrative-PZ reliability vocabulary.
        "перезагрузк", "перезапуск", "обработк", "продолж",
        "функционир", "работоспособ", "сбой", "неисправн",
    ),
}

# Polyakov-regression (R2): narrative-PZ section title boost — mirror
# of `_PMI_TEST_SECTION_RE`. ВКР sections like «Архитектура клиентской
# части», «Обоснование средств разработки», «Иерархия данных DSpace»,
# «Структура клиентской части», «Реализация», «Развёртывание»
# canonically describe HOW the requirement is implemented in PZ. Boost
# them to 1.0 unconditionally (ignoring requirement type) so the
# evidence_floor doesn't suppress narrative implementation evidence
# whose lex retrieval is intrinsically low (PZ paragraphs use
# implementation vocabulary, TZ requirements use specification
# vocabulary — vocab gap is the structural problem).
_PZ_NARRATIVE_SECTION_RE = re.compile(
    r"архитектур"
    r"|структур\w*\s+(?:клиентск|серверн|систем|приложен|реализац)"
    r"|обоснован\w*\s+(?:средств|выбор)"
    r"|средств\w*\s+разработк"
    r"|реализац"
    r"|развертыван|развёртыван"
    r"|иерархи\w*\s+данн"
    r"|сценар\w*\s+(?:использовани|пользоват|работ)"
    r"|ролев\w*\s+модел"
    r"|роли\s+пользоват"
    r"|клиент-серверн"
    r"|пользовательск\w*\s+интерфейс"
    r"|описани\w*\s+(?:реализац|компонент|систем)",
    re.IGNORECASE | re.UNICODE,
)

_PMI_TOPIC_ALIASES: dict[RequirementType, tuple[str, ...]] = {
    RequirementType.FUNCTIONAL: (
        "функциональн", "авторизац", "регистрац", "добавлен", "проект",
        "разгранич", "поиск", "фильтр", "файл",
    ),
    RequirementType.INTERFACE: ("программному интерфейсу", "интерфейс", "ui", "ux"),
    RequirementType.RELIABILITY: ("надеж", "надёж", "восстанов", "ошиб", "отказ"),
    RequirementType.SECURITY: ("авторизац", "доступ", "роль", "инъекц", "некорректн"),
    RequirementType.DATA_IO: ("ошиб", "неверн", "сообщен", "запрос", "данн"),
    RequirementType.PERFORMANCE: ("отклик", "секунд", "время", "производительн"),
    RequirementType.ENVIRONMENT_REQUIREMENT: ("браузер", "windows", "процессор", "памят", "монитор"),
    RequirementType.DOCUMENTATION_REQUIREMENT: ("документац", "методик", "испытан", "гост"),
}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _constraint_overlap(req_constraints: List[Constraint], unit_constraints: List[Constraint]) -> float:
    """
    Score in [0, 1]:
      1.0 per constraint pair with matching unit class AND matching value
          (comparing canonical values — "90 дней" matches "7776000 секунд"
           if we ever saw such a conversion)
      0.5 per constraint pair with matching unit class only
          (same topic, different value → possible conflict, keep as retrieval signal)
    Normalized by number of req constraints.
    """
    if not req_constraints:
        return 0.0
    if not unit_constraints:
        return 0.0

    from app.application.use_cases.verify_pairs import (
        _canonical_value,
        _same_unit_class,
    )

    total = 0.0
    for rc in req_constraints:
        for uc in unit_constraints:
            if not _same_unit_class(rc.unit, uc.unit):
                continue
            # Compare on canonical scale when both units are convertible;
            # this makes the retriever treat "2 сек" and "2000 мс" as an
            # exact match instead of a possible-conflict.
            rc_canon = _canonical_value(rc.value, rc.unit)
            uc_canon = _canonical_value(uc.value, uc.unit)
            if rc_canon is not None and uc_canon is not None:
                tol = max(abs(rc_canon), abs(uc_canon)) * 1e-3
                if abs(rc_canon - uc_canon) <= max(tol, 1e-6):
                    total += 1.0
                else:
                    total += 0.5
            else:
                if abs(rc.value - uc.value) < 1e-6:
                    total += 1.0
                else:
                    total += 0.5
    return min(total / len(req_constraints), 1.0)


def _section_prior(req: RequirementUnit, unit: CoverageUnit) -> float:
    role = unit.target_doc_role.lower()
    if role == "pmi":
        # Polyakov-regression: ПМИ units coming from a canonical test
        # section («Требования к программе», «Методы испытаний»,
        # «Состав и порядок испытаний», «Проверка требований к
        # программной документации», «Объект/Цель испытаний») are
        # exactly where coverage of ТЗ requirements lives by ГОСТ
        # 19.301 convention — boost them regardless of req type so
        # the retriever surfaces them above evidence_floor and the
        # LLM judge gets to evaluate the pair instead of being
        # short-circuited by NO_EVIDENCE / OPTIONAL_NOT_FOUND.
        section_title = _unit_section_title(unit)
        if section_title and _PMI_TEST_SECTION_RE.search(section_title):
            return 1.0
        if req.requirement_type in _PMI_PREFERRED:
            return 1.0
    if role == "pz":
        # Polyakov-regression (R2): narrative-PZ section title boost.
        # ВКР-style PZ documents express coverage as descriptions of
        # implementation under headings like «Архитектура клиентской
        # части», «Обоснование средств разработки», «Иерархия данных
        # DSpace». Their lex/sem against TZ requirement text is
        # intrinsically low because PZ uses implementation-vocab and
        # TZ uses specification-vocab; without a section boost they
        # land below evidence_floor and never reach the LLM judge,
        # producing the «0 COVERED / 31 MISSING» PZ headline. Boost
        # unconditionally — type-level prefs are a separate fall-back
        # path right below.
        section_title = _unit_section_title(unit)
        if section_title and _PZ_NARRATIVE_SECTION_RE.search(section_title):
            return 1.0
        if req.requirement_type in _PZ_PREFERRED:
            return 1.0
    return 0.0


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(n and n.lower() in lower for n in needles)


def _unit_section_title(unit: CoverageUnit) -> str:
    return str(
        unit.metadata.get("section_title")
        or unit.metadata.get("sectionTitle")
        or ""
    )


def _topic_prior(req: RequirementUnit, unit: CoverageUnit) -> float:
    """Type-aware boost for evidence from the natural section/topic.

    This is intentionally a retrieval-only prior. It makes the right section
    visible to the judge; it does not itself declare coverage.
    """
    role = unit.target_doc_role.lower()
    title = _unit_section_title(unit)
    haystack = f"{title}\n{unit.text}".lower()
    if role == "pz":
        aliases = _PZ_TOPIC_ALIASES.get(req.requirement_type, ())
    elif role == "pmi":
        aliases = _PMI_TOPIC_ALIASES.get(req.requirement_type, ())
    else:
        aliases = ()
    if not aliases:
        return 0.0
    title_hit = _contains_any(title, aliases)
    text_hit = _contains_any(haystack, aliases)
    if title_hit and text_hit:
        return 1.0
    if title_hit:
        return 0.85
    if text_hit:
        return 0.55
    return 0.0


def _noise_penalty(req: RequirementUnit, unit: CoverageUnit) -> float:
    """Demote recurring descriptive PZ fragments that are not evidence.

    These fragments repeatedly win lexical/semantic retrieval in Polyakov-like
    packages but describe analogues, admin capabilities, file layout, or a
    one-off data repair operation rather than coverage of a TZ requirement.
    """
    text = unit.text or ""
    section_title = _unit_section_title(unit)
    penalty = 0.0

    if (
        (section_title and _NON_IMPL_SECTION_RE.search(section_title))
        or _NON_IMPL_SECTION_TEXT_RE.search(text)
    ):
        penalty += _NON_IMPL_SECTION_PENALTY

    if _FILE_STRUCTURE_TEXT_RE.search(text):
        if req.requirement_type == RequirementType.ARCHITECTURE_IMPLEMENTATION:
            req_aspects = _aspect_tokens(req.text)
            unit_aspects = _aspect_tokens(text)
            if not ({"typescript_angular", "github"} & req_aspects & unit_aspects):
                penalty += _FILE_STRUCTURE_PENALTY
            else:
                penalty += 0.12
        else:
            penalty += _FILE_STRUCTURE_PENALTY

    if _ADMIN_CAPABILITY_TEXT_RE.search(text):
        req_aspects = _aspect_tokens(req.text)
        unit_aspects = _aspect_tokens(text)
        if not ({"authorization", "authentication"} & req_aspects & unit_aspects):
            penalty += _ADMIN_CAPABILITY_PENALTY
        else:
            penalty += 0.10

    if _COLLECTION_REPAIR_TEXT_RE.search(text):
        penalty += _COLLECTION_REPAIR_PENALTY

    return min(penalty, 0.75)


def _aspect_tokens(text: str) -> Set[str]:
    """Exact anchors that should survive paraphrase-heavy retrieval."""
    out: Set[str] = set()
    lower_text = (text or "").lower()
    for canonical, variants in _ASPECT_ALIASES.items():
        if _contains_any(lower_text, variants):
            out.add(canonical)
    # Polyakov-regression (R2): file-extension tokens («.ts», «.html»,
    # «.css», «.scss») are the canonical PZ form for client-side
    # files but `_ASPECT_TOKEN_RE` requires a leading letter, so they
    # never get extracted. Pull them explicitly so an aspect overlap
    # between TZ «HTML и CSS» and PZ «.ts, .html, .css, .scss» fires.
    for raw_ext in _FILE_EXT_TOKEN_RE.findall(text or ""):
        out.add(raw_ext.lower())
        # Also add the bare-name form so `.html` overlaps with «html»
        # extracted from a TZ requirement that uses the unprefixed name.
        out.add(raw_ext.lower().lstrip("."))
    for raw in _ASPECT_TOKEN_RE.findall(text or ""):
        token = raw.lower().strip(".,;:()[]{}")
        if not token:
            continue
        if token in _TECH_KEYWORDS or token.replace(".", "", 1).isdigit():
            out.add(token)
            continue
        has_upper = any(c.isupper() for c in raw)
        has_digit = any(c.isdigit() for c in raw)
        if len(token) >= 3 and (has_upper or has_digit):
            out.add(token)
    return out


def _aspect_overlap(req_text: str, unit_text: str) -> float:
    req_aspects = _aspect_tokens(req_text)
    if not req_aspects:
        return 0.0
    unit_aspects = _aspect_tokens(unit_text)
    if not unit_aspects:
        return 0.0
    return len(req_aspects & unit_aspects) / len(req_aspects)


def _boostable_aspect_overlap(req_text: str, unit_text: str) -> float:
    req_aspects = _aspect_tokens(req_text) & _BOOSTABLE_ASPECTS
    if not req_aspects:
        return 0.0
    unit_aspects = _aspect_tokens(unit_text) & _BOOSTABLE_ASPECTS
    if not unit_aspects:
        return 0.0
    return len(req_aspects & unit_aspects) / len(req_aspects)


def _build_score_reason(
    lex: float, sem: float, con: float, sec: float, total: float, exact: float = 0.0,
) -> str:
    """One-line, human-readable explanation of which component drove
    the score. Rendered in evidence_trace and also handy for log
    inspection. Pure formatting — no thresholds outside what the
    reader of a score-breakdown intuitively expects."""
    parts: List[str] = []
    # Identify dominant component(s). Treat anything within 70% of the
    # max as "co-leading".
    components = {
        "lex": lex,
        "sem": sem,
        "con": con,
        "sec": sec,
        "exact": exact,
    }
    max_v = max(components.values()) if components else 0.0
    if max_v <= 0.0:
        return f"all signals near zero (score={total:.2f})"
    leaders = [k for k, v in components.items() if v >= 0.7 * max_v and v > 0.0]

    label = {
        "lex": "lexical",
        "sem": "semantic",
        "con": "constraint",
        "sec": "section",
        "exact": "exact-aspect",
    }
    if len(leaders) == 1:
        parts.append(f"{label[leaders[0]]} dominant")
    else:
        parts.append("+".join(label[k] for k in leaders) + " co-leading")

    parts.append(
        f"lex={lex:.2f} sem={sem:.2f} con={con:.2f} sec={sec:.2f} ⇒ {total:.2f}"
    )
    return " | ".join(parts)


def _conditional_should_rerank(
    requirement: RequirementUnit,
    sorted_candidates: List[RetrievedCandidate],
    rr_cfg: CoverageRerankerConfig,
) -> tuple[bool, str]:
    """Return (should_rerank, reason). Pure decision over the first-stage
    shortlist. The thresholds live in CoverageRerankerConfig so they
    can be retuned without code changes.

    Rules (any one fires):
      * Top-1 score below `conditional_top1_threshold` — first stage is weak.
      * Top-1 minus top-2 below `conditional_min_margin` — close call.
      * Requirement is critical (SECURITY / PERFORMANCE / RELIABILITY).
      * Requirement carries numeric constraints — verify mismatch risk.
      * Top-1 has high semantic but low lexical (paraphrase) — bi-encoder
        risks a false positive without cross-encoder confirmation.
    """
    if not sorted_candidates:
        return False, "empty shortlist"
    top1 = sorted_candidates[0]
    top2 = sorted_candidates[1] if len(sorted_candidates) > 1 else None
    margin = (top1.retrieval_score - top2.retrieval_score) if top2 else top1.retrieval_score

    if top1.retrieval_score < rr_cfg.conditional_top1_threshold:
        return True, (
            f"top1 score {top1.retrieval_score:.3f} < "
            f"{rr_cfg.conditional_top1_threshold:.2f}"
        )
    if top2 is not None and margin < rr_cfg.conditional_min_margin:
        return True, (
            f"top1-top2 margin {margin:.3f} < "
            f"{rr_cfg.conditional_min_margin:.2f}"
        )
    if requirement.requirement_type in _CRITICAL_TYPES:
        return True, (
            f"critical type {requirement.requirement_type.value}"
        )
    if requirement.constraints:
        return True, (
            f"requirement carries {len(requirement.constraints)} numeric constraint(s)"
        )
    # Paraphrase signal: high semantic, low lexical on top-1.
    if top1.semantic_score >= 0.55 and top1.lexical_score <= 0.20:
        return True, (
            f"top1 paraphrase-like (sem={top1.semantic_score:.2f}, "
            f"lex={top1.lexical_score:.2f})"
        )
    return False, "first-stage signals strong; reranker skipped"


class CandidateRetriever:
    """Hybrid first-stage scoring + optional cross-encoder rerank.

    Pipeline:
        1. Score every unit with the hybrid formula
           (lex + semantic + constraint + section).
        2. Keep top-N (`top_k_before_rerank`) above `min_retrieval_score`.
        3. Reranker decision:
             - `mode == "always"` → run on the top-N, overwrite scores.
             - `mode == "conditional"` → run only when first-stage signals
               are weak (see `_conditional_should_rerank`).
             - `enabled == False` → skip.
        4. Fill `score_reason` and `evidence_strength` on every
           returned candidate, then trim to `initial_top_n` (PR-K) or
           `top_k` (legacy fallback).
    """

    def __init__(
        self,
        config: CoverageRetrievalConfig,
        embedding_backend: EmbeddingBackend,
        reranker: Reranker | None = None,
        reranker_config: Optional[CoverageRerankerConfig] = None,
    ) -> None:
        self._cfg = config
        self._emb = embedding_backend
        self._reranker: Reranker = reranker or NoopReranker()
        # `reranker_config` is optional so existing callers keep working.
        # When omitted we default to "always" (legacy behaviour) so a
        # wired reranker is exercised on every shortlist as before.
        self._rr_cfg: CoverageRerankerConfig = (
            reranker_config or CoverageRerankerConfig(mode="always")
        )

    # ------------------------------------------------------------------

    def retrieve(
        self,
        requirement: RequirementUnit,
        coverage_units: List[CoverageUnit],
    ) -> List[RetrievedCandidate]:
        """Return up to `initial_top_n` candidates above
        `min_retrieval_score`, sorted descending. Falls back to `top_k`
        when `initial_top_n` is unset (older configs)."""
        if not coverage_units:
            return []

        req_tokens: Set[str] = tokenize_content(requirement.normalized_text)
        candidate_texts = [u.normalized_text for u in coverage_units]

        semantic_scores = self._emb.similarity(requirement.normalized_text, candidate_texts)

        results: List[RetrievedCandidate] = []
        for i, unit in enumerate(coverage_units):
            unit_tokens: Set[str] = tokenize_content(unit.normalized_text)

            lex = _jaccard(req_tokens, unit_tokens)
            sem = float(semantic_scores[i]) if i < len(semantic_scores) else 0.0
            con = _constraint_overlap(requirement.constraints, unit.constraints)
            topic = _topic_prior(requirement, unit)
            sec = max(_section_prior(requirement, unit), topic)
            exact = _boostable_aspect_overlap(requirement.text, unit.text)

            score = (
                self._cfg.lexical_weight * lex
                + self._cfg.semantic_weight * sem
                + self._cfg.constraint_weight * con
                + self._cfg.section_prior_weight * sec
            )
            if exact > 0.0:
                score += min(0.22, 0.06 + 0.16 * exact)
            if topic > 0.0:
                score += min(0.14, 0.04 + 0.10 * topic)
            score = min(score, 0.9999)

            penalty = _noise_penalty(requirement, unit)
            if penalty > 0.0:
                score = max(0.0, score - penalty)
                logger.debug(
                    "Coverage noise penalty: unit=%s penalty=%.2f score→%.3f",
                    unit.unit_id[:8], penalty, score,
                )

            if score < self._cfg.min_retrieval_score:
                continue

            results.append(
                RetrievedCandidate(
                    req_id=requirement.req_id,
                    unit_id=unit.unit_id,
                    target_document_id=unit.target_document_id,
                    lexical_score=round(lex, 4),
                    semantic_score=round(sem, 4),
                    constraint_overlap_score=round(con, 4),
                    section_prior_score=round(sec, 4),
                    retrieval_score=round(score, 4),
                    unit_type=unit.unit_type,
                )
            )

        results.sort(key=lambda c: c.retrieval_score, reverse=True)

        # --- Second stage: cross-encoder rerank of the top-N -----------
        #
        # PR-K change: the decision to rerank is now driven by
        # CoverageRerankerConfig.mode. "always" preserves legacy
        # behaviour; "conditional" runs only when first-stage signals
        # are weak.
        if not isinstance(self._reranker, NoopReranker):
            shortlist = results[: self._cfg.top_k_before_rerank]
            mode = (self._rr_cfg.mode or "always").lower()
            should_rerank = True
            rerank_reason = "mode=always"
            if mode == "conditional":
                should_rerank, rerank_reason = _conditional_should_rerank(
                    requirement, shortlist, self._rr_cfg,
                )

            if shortlist and should_rerank:
                unit_by_id = {u.unit_id: u for u in coverage_units}
                unit_text_by_id = {u.unit_id: u.normalized_text for u in coverage_units}
                texts = [unit_text_by_id.get(c.unit_id, "") for c in shortlist]
                try:
                    rr_scores = self._reranker.score(requirement.normalized_text, texts)
                except Exception as exc:
                    logger.warning(
                        "Reranker failed (%s); falling back to hybrid order", exc,
                    )
                    rr_scores = None
                if rr_scores is not None and len(rr_scores) == len(shortlist):
                    # Overwrite retrieval_score with rerank score so the
                    # rest of the pipeline (and the final report) reflects
                    # what actually determined the top-K order.
                    for c, s in zip(shortlist, rr_scores):
                        unit = unit_by_id.get(c.unit_id)
                        score = float(s)
                        if unit is not None:
                            exact = _boostable_aspect_overlap(requirement.text, unit.text)
                            topic = _topic_prior(requirement, unit)
                            if exact >= 0.5:
                                score += min(0.12, 0.04 + 0.08 * exact)
                            elif topic > 0.0:
                                score += min(0.06, 0.02 + 0.04 * topic)
                        # Defense-in-depth: clamp to (0, 0.9999) even
                        # though BGE adapter now sigmoids its output.
                        # Future reranker backends or boost-stacking
                        # bugs (exact/topic boost added above can push
                        # over 1.0) get neutralised here.
                        c.retrieval_score = round(max(0.0, min(score, 0.9999)), 4)
                        c.reranker_used = True
                        c.reranker_score = round(float(s), 4)
                    shortlist.sort(key=lambda c: c.retrieval_score, reverse=True)
                    results = shortlist
            elif shortlist and not should_rerank:
                logger.debug(
                    "Reranker skipped for req=%s: %s",
                    requirement.req_id[:12], rerank_reason,
                )

        # PR-K: populate explainability fields. Done AFTER any reranking
        # so the score_reason reflects the final retrieval_score and the
        # evidence_strength binning matches what downstream sees.
        strong = self._cfg.evidence_strength_strong_threshold
        medium = self._cfg.evidence_strength_medium_threshold
        weak = self._cfg.evidence_strength_weak_threshold
        for c in results:
            if c.reranker_used:
                # When the cross-encoder rewrote the score, the original
                # component breakdown is no longer the determinant — flag
                # that explicitly.
                c.score_reason = (
                    f"reranker score {c.retrieval_score:.2f} "
                    f"(first-stage lex={c.lexical_score:.2f} "
                    f"sem={c.semantic_score:.2f} con={c.constraint_overlap_score:.2f})"
                )
            else:
                c.score_reason = _build_score_reason(
                    c.lexical_score, c.semantic_score,
                    c.constraint_overlap_score, c.section_prior_score,
                    c.retrieval_score,
                    _boostable_aspect_overlap(
                        requirement.text,
                        next((u.text for u in coverage_units if u.unit_id == c.unit_id), ""),
                    ),
                )
            c.evidence_strength = evidence_strength_from_score(
                c.retrieval_score, strong=strong, medium=medium, weak=weak,
            )

        # PR-K: return up to initial_top_n. Falls back to top_k when the
        # config predates PR-K (e.g. unit tests that explicitly set top_k
        # to a small value and never touch initial_top_n).
        cap = max(
            getattr(self._cfg, "initial_top_n", 0) or 0,
            self._cfg.top_k,
        )
        if cap <= 0:
            cap = self._cfg.top_k
        return results[:cap]
