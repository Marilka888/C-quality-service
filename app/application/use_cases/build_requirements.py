"""
Stage 1 of the coverage pipeline: build RequirementUnit list from a TZ artifact.

Priority:
  1. prepared_artifact.requirement_candidates (if present and non-empty)
  2. Heuristic extraction from fragments[] via modality / constraint patterns
"""
from __future__ import annotations

import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

from app.core.logging import get_logger
from app.core.config import CoverageConfig
from app.domain.c_quality_enums import Modality, RequirementType
from app.domain.c_quality_models import Constraint, RequirementUnit

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Boilerplate / non-requirement post-filter
# ---------------------------------------------------------------------------
#
# The fine-tuned classifier still occasionally marks document-form stamps,
# bibliography entries and project-background prose as requirements — these
# were underrepresented in the v5 training set. A cheap regex pass catches
# the most common false-positive shapes. Applied AFTER the classifier gate,
# BEFORE constructing a RequirementUnit.
#
# Added conservatively: each pattern corresponds to a concrete false
# positive observed in manual review. Over-filtering loses real
# requirements, so keep the rule set tight and test-covered.

_BOILERPLATE_PATTERNS = [
    # GOST-style form field / cover-page stamp
    re.compile(r"\bПодп\.\s*Дата\b", re.I),
    # Document-code lines like "RU.17701729.04.01-01 ТЗ 01-1"
    re.compile(r"\bRU\.\d{5,}\.\d", re.I),
    # Archival inventory stamps
    re.compile(r"\bинв\.?\s*№\s*подп", re.I),
    # Bibliography entries: "– М.: Изд-во стандартов, 1997"
    re.compile(r"[–\-]\s*М\.\s*:\s*Изд-во", re.I),
    # Title-page duplication: "Наименование программы Наименование темы …"
    re.compile(r"^\s*Наименование\s+программы\s+Наименование\s+", re.I),
    # Project rationale headers
    re.compile(r"^\s*Предполагаемая\s+потребность\b", re.I),
    re.compile(r"^\s*Актуальность\s+(разработки|автоматизации|проекта)", re.I),
    re.compile(r"^\s*ПОРЯДОК\s+КОНТРОЛЯ\s+И\s+ПРИЕМКИ\b", re.I),
]

# Trailing section marker: short text ending in a bare section number with
# no body content after it. Examples from v3 review:
#   "Разрабатываемая программа должно иметь следующий функционал: 4.2."
#   "Требования к функциональным характеристикам 4.1.1."
# These are headings glued onto truncated text by the PDF parser.
_TRAILING_SECTION_NUM_RE = re.compile(r"\s\d+(?:\.\d+)+\.?\s*$")

# Standalone GOST reference lines that list a standard without imposing a
# requirement: "ГОСТ 19.101-77 Виды программ и программных документов."
# Heuristic: starts with "ГОСТ N...", has no modality trigger, ≤ 14 words.
_GOST_LINE_RE = re.compile(r"^\s*ГОСТ\s+\d", re.I)

# Glossary-style entry: single term followed by em-dash / en-dash and a
# definition, within a short fragment. Example:
#   "Триггер – пороговое значение свойства объекта мониторинга..."
# We only fire when the LEFT side is 1-3 words (a term, not a sentence).
_GLOSSARY_RE = re.compile(
    r"^\s*[А-ЯЁA-Z][\wа-яёА-ЯЁ\-]*(?:\s+[\wа-яёА-ЯЁ\-]+){0,2}\s*[\u2013\u2014\-]\s+[А-ЯЁа-яёa-zA-Z]",
)


def _is_document_boilerplate(text: str) -> bool:
    """True if the text is a form stamp / citation / rationale header.

    Intended as a conservative net: matches a short list of concrete
    non-requirement shapes. Requirement-bearing sentences that happen to
    contain one of these patterns (e.g. "Подп." occurring mid-sentence)
    can still slip past because the patterns are anchored or specific.
    """
    if not text:
        return True
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(text):
            return True

    stripped = text.strip()
    # Strip common list markers before pattern matching so "2) ГОСТ N ..." is
    # recognised as the same shape as "ГОСТ N ...".
    stripped_nolist = re.sub(
        r"^\s*(?:\d+[.)]|[а-яёa-z][.)]|[•\-\u2013\u2014])\s+",
        "",
        stripped,
    )
    words = stripped.split()
    has_modality = (
        bool(_BODY_TRIGGER_RE.search(stripped))
        if "_BODY_TRIGGER_RE" in globals()
        else False
    )

    # Trailing bare section number ("... 4.2." at end) in a short fragment —
    # heading got glued onto section marker.
    if len(words) <= 14 and _TRAILING_SECTION_NUM_RE.search(stripped):
        return True

    # Leading page-number + section-number chain: "11 4.1.1.2 Требования…".
    # Two numeric groups with the second one multi-level is the hallmark of
    # a PDF-extracted heading line.
    if len(words) <= 16 and re.match(r"^\s*\d+\s+\d+(?:\.\d+)+\s+", stripped):
        return True

    # GOST standard reference line without a modality verb. A real
    # requirement might say "должны соответствовать ГОСТ 19.101"; a bare
    # list entry is just the reference. Also handles leading list markers.
    if _GOST_LINE_RE.match(stripped_nolist) and not has_modality and len(words) <= 16:
        return True

    # "Something (ГОСТ N-N)" standalone short line — document name in
    # parentheses is a citation shape, not a requirement.
    if (
        len(words) <= 12
        and not has_modality
        and re.search(r"\(ГОСТ\s+\d+[\-.]\d", stripped, re.I)
    ):
        return True

    # Glossary / terminology definition in a short block.
    if len(words) <= 18 and not has_modality and _GLOSSARY_RE.match(stripped):
        return True

    # Stamp heuristic: very short line (≤ 8 words) with a dense mix of
    # digits / dots / dashes — typical of document codes lines.
    if len(words) <= 8:
        digit_punct_chars = sum(1 for c in text if c.isdigit() or c in ".-/№")
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars and digit_punct_chars / max(1, alpha_chars + digit_punct_chars) > 0.4:
            return True

    # Audit (Annenkov package): section-header text glued to the start
    # of a bullet list, e.g.
    #   "Требования к программе 4.1. Требования к функциональным
    #    характеристикам Модуль ... обеспечивать выполнение
    #    перечисленных ниже функций:."
    # Two "Требования к ..." headers separated by a section number —
    # almost always a glued heading. ONLY fire on short fragments
    # (<= 50 words) — long sections legitimately contain multiple
    # "Требования к ..." subheadings that aren't pasted together.
    # Regression note: an earlier broader version filtered any
    # sentence ending with "следующие/перечисленных ... функций:"
    # which incorrectly killed legitimate requirement-list intros.
    # Removed.
    if len(words) <= 50 and re.search(
        r"Требования\s+к\s+\S+.*?\d+\.\d+\.?.*?Требования\s+к\s+\S+",
        stripped, re.IGNORECASE | re.DOTALL,
    ):
        return True

    return False


_MUST_NOT_RE = re.compile(
    r"\b(не должен|не должна|не должны|запрещено|недопустимо|не допускается)\b", re.I
)
_MUST_RE = re.compile(
    r"\b(должен|должна|должны|должно|необходимо|обязан|обязана|обязаны)\b", re.I
)
_SHOULD_RE = re.compile(
    r"\b(следует|рекомендуется|желательно|рекомендован)\b", re.I
)
_MAY_RE = re.compile(
    r"\b(может|могут|допускается|разрешено|разрешается)\b", re.I
)

# Requirement-trigger words — a fragment containing these is a candidate
_TRIGGER_RE = re.compile(
    r"\b(должен|должна|должны|не должен|следует|необходимо|обязан|"
    r"обеспечивать|реализовывать|поддерживать|хранить|предусматривать|"
    r"осуществлять|выполнять|предоставлять|контролировать)\b",
    re.I,
)

_OPERATOR_WORDS: Dict[str, str] = {
    "не менее": ">=",
    "не более": "<=",
    "не превышает": "<=",
    "не превышать": "<=",
    "не меньше": ">=",
    "не больше": "<=",
    "минимум": ">=",
    "максимум": "<=",
    "более": ">",
    "менее": "<",
    "от": ">=",
    "до": "<=",
}

# Regex: optional operator-word, number, optional unit
_CONSTRAINT_RE = re.compile(
    r"(?P<op_sym>>=|<=|>|<|=)?\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>дн(?:ей|я|ей)?|суток?|сут\b|час(?:а|ов|ах)?\b|ч\b|"
    r"мин(?:ут(?:ы)?)?\b|сек(?:унд(?:ы)?)?\b|мс\b|ms\b|с\b|s\b|"
    r"МБ\b|ГБ\b|КБ\b|мб\b|гб\b|кб\b|[kK][bB]\b|[mM][bB]\b|[gG][bB]\b|[tT][bB]\b|"
    r"rps\b|rpm\b|%\b|раз\b|попытк(?:и|ок)?\b)?",
    re.I,
)

_UNIT_NORM: Dict[str, str] = {
    # time-days
    "дней": "days", "дня": "days", "день": "days",
    "суток": "days", "сут": "days",
    # time-hours
    "часа": "hours", "часов": "hours", "час": "hours", "ч": "hours",
    # time-min
    "минут": "min", "минуты": "min", "мин": "min",
    # time-sec
    "секунд": "sec", "секунды": "sec", "сек": "sec", "с": "sec", "s": "sec",
    # time-ms
    "мс": "ms", "ms": "ms",
    # size
    "мб": "mb", "МБ": "mb", "mb": "mb",
    "гб": "gb", "ГБ": "gb", "gb": "gb",
    "кб": "kb", "КБ": "kb", "kb": "kb",
    "tb": "tb",
    # rate
    "rps": "rps", "rpm": "rpm",
    # other
    "%": "%", "раз": "times",
    "попытки": "attempts", "попытка": "attempts", "попыток": "attempts",
}

_KIND_KEYWORDS: Dict[str, List[str]] = {
    "response_time": ["время", "ответ", "latency", "задержка", "отклик", "response"],
    "retention_period": ["хранен", "хранить", "журнал", "лог", "архив", "retention", "хранится"],
    "throughput": ["rps", "rpm", "запрос", "пропускн", "throughput"],
    "size": ["размер", "объем", "объём", "size", "МБ", "ГБ"],
    "attempts": ["попытк", "attempt", "повтор"],
    "timeout": ["таймаут", "timeout", "ожидан"],
    "availability": ["доступност", "uptime", "availability"],
}

# ---------------------------------------------------------------------------
# Section relevance (GOST TZ structure)
# ---------------------------------------------------------------------------

_SECTION_NUM_RE = re.compile(r"^\s*(\d+)(?:[.\s]|$)")

_REQUIREMENT_SECTION_NUMBERS = {4}          # Section 4: Требования к программе
_NON_REQUIREMENT_SECTION_NUMBERS = {1, 2, 3}  # Обозначения, Введение, Назначение

# Keywords in section titles that mark a section as carrying requirements.
# Matched case-insensitively against whole-word substrings of the title.
_REQUIREMENT_TITLE_KEYWORDS = (
    "требовани",           # требования, требованиям, требований
    "характеристик",        # характеристики надёжности / производительности
    "функциональн",         # функциональные возможности / требования
    "назначен",             # назначение и функции (часто содержит MUST)
    "условия применени",
    "состав и параметр",
    "режимы работы",
)

# Keywords that mark a section as clearly non-requirement.
_NON_REQUIREMENT_TITLE_KEYWORDS = (
    "введени",
    "термин",
    "обозначени",
    "сокращени",
    "перечень",
    "библиограф",
    "литератур",
    "содержани",
    "оглавлени",
    "приложени",          # ГОСТ: приложения как правило — справочные
    # Documentation-deliverable sections: "Требования к программной
    # документации", "Требования к документации программы", etc.
    # These describe WHAT DOCUMENTS to produce, not WHAT THE PROGRAM must
    # do — they are project-management obligations, not program requirements.
    "программной документаци",   # Требования к программной документации
    "к документации",            # Требования к документации (any form)
)


def _leading_section_number(section_id: str) -> Optional[int]:
    m = _SECTION_NUM_RE.match(section_id)
    return int(m.group(1)) if m else None


def _is_requirement_section(section_id: Optional[str]) -> Optional[bool]:
    """
    True  — definitely a requirement section (GOST TZ section 4.x)
    False — definitely NOT a requirement section (sections 1, 2, 3)
    None  — unknown / ambiguous (no section_id, or sections 5+)
    """
    if not section_id:
        return None
    num = _leading_section_number(section_id.strip())
    if num is None:
        return None
    if num in _REQUIREMENT_SECTION_NUMBERS:
        return True
    if num in _NON_REQUIREMENT_SECTION_NUMBERS:
        return False
    return None


def _is_requirement_section_by_title(title: Optional[str]) -> Optional[bool]:
    """
    Title-based requirement-section classifier. Returns True/False/None with
    the same semantics as `_is_requirement_section`. Used when section
    numbering does not follow GOST TZ conventions (e.g. GOST 19 specs,
    internal templates).
    """
    if not title:
        return None
    lower = title.lower()
    for kw in _NON_REQUIREMENT_TITLE_KEYWORDS:
        if kw in lower:
            return False
    for kw in _REQUIREMENT_TITLE_KEYWORDS:
        if kw in lower:
            return True
    return None


def _classify_section(section_id: Optional[str], title: Optional[str]) -> Optional[bool]:
    """
    Combined classifier: numbering takes precedence (GOST TZ is explicit),
    title is the tiebreaker for ambiguous numbering.
    """
    by_num = _is_requirement_section(section_id)
    if by_num is not None:
        return by_num
    return _is_requirement_section_by_title(title)


# ---------------------------------------------------------------------------
# Russian sentence splitter
# ---------------------------------------------------------------------------

# Abbreviations whose trailing period must NOT terminate a sentence.
_RU_ABBREV = (
    "т.е.", "т. е.", "т.д.", "т. д.", "т.п.", "т. п.", "т.к.", "т. к.",
    "и.о.", "и. о.", "др.", "пр.", "см.", "стр.", "рис.", "табл.",
    "гл.", "п.", "пп.", "ст.", "абз.",
    "г.", "гг.", "в.", "вв.", "н.э.",
    "млн.", "млрд.", "тыс.",
    "руб.", "коп.",
)

_SENT_END_RE = re.compile(r"([.!?…]+)(\s+|$)")


def _split_sentences_ru(text: str) -> List[str]:
    """
    Rule-based Russian sentence splitter.

    Strategy: protect known abbreviations by substituting their period with a
    sentinel, split on [.!?…] + whitespace, restore sentinels. Good enough for
    ТЗ text (formal register, relatively few exotic abbreviations).
    """
    if not text:
        return []
    sentinel = "\x00"
    protected = text
    for abbr in _RU_ABBREV:
        protected = re.sub(
            re.escape(abbr),
            abbr.replace(".", sentinel),
            protected,
            flags=re.IGNORECASE,
        )

    parts: List[str] = []
    buf: List[str] = []
    last = 0
    for m in _SENT_END_RE.finditer(protected):
        chunk = protected[last : m.end()]
        buf.append(chunk)
        # Close the sentence only if the next character starts a new one
        # (capital letter, digit, or end of string).
        tail = protected[m.end():].lstrip()
        if not tail or tail[0].isupper() or tail[0].isdigit() or tail[0] == "-":
            parts.append("".join(buf))
            buf = []
        last = m.end()
    if last < len(protected):
        buf.append(protected[last:])
    if buf:
        parts.append("".join(buf))

    result: List[str] = []
    for p in parts:
        s = p.replace(sentinel, ".").strip()
        if s:
            result.append(s)
    return result


_TYPE_KEYWORDS: Dict[RequirementType, List[str]] = {
    RequirementType.PERFORMANCE: [
        "время", "производительность", "rps", "задержка", "скорость",
        "latency", "throughput", "мс", "сек", "отклик", "нагрузк",
    ],
    RequirementType.SECURITY: [
        "безопасность", "аутентификац", "авторизац", "шифрован",
        "доступ", "права", "пароль", "токен", "ключ",
    ],
    RequirementType.LOGGING: [
        "журнал", "лог", "logging", "аудит", "запись", "регистрац",
    ],
    RequirementType.STORAGE: [
        "хранен", "хранить", "архив", "база данных", "бд", "дней", "суток", "данных",
    ],
    RequirementType.INTERFACE: [
        "интерфейс", "api", "протокол", "формат", "rest", "soap", "json", "xml",
    ],
    RequirementType.FUNCTIONAL: [
        "функция", "функционал", "возможность", "операция", "обеспечивать",
        "реализовывать", "выполнять", "поддерживать",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _extract_modality(text: str) -> Modality:
    if _MUST_NOT_RE.search(text):
        return Modality.MUST_NOT
    if _MUST_RE.search(text):
        return Modality.MUST
    if _SHOULD_RE.search(text):
        return Modality.SHOULD
    if _MAY_RE.search(text):
        return Modality.MAY
    return Modality.UNKNOWN


def _extract_requirement_type(text: str, section_title: str = "") -> RequirementType:
    """Classify a requirement by type. Delegates to the rule-based
    classifier in `classify_requirement`, which uses both the section
    title (most reliable signal) and the requirement text. Legacy
    callers passing only `text` fall back to text-only classification."""
    from app.application.use_cases.classify_requirement import classify_requirement
    return classify_requirement(text, section_title)


def _legacy_extract_requirement_type_keywords(text: str) -> RequirementType:
    """Fallback keyword classifier (kept for tests that pin old behaviour
    on the FUNCTIONAL / PERFORMANCE / SECURITY / LOGGING / STORAGE /
    INTERFACE axes). Not used by the production builder anymore."""
    lower = text.lower()
    for req_type, keywords in _TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return req_type
    return RequirementType.OTHER


def _determine_kind(context: str, unit: Optional[str]) -> str:
    lower = context.lower()
    for kind, keywords in _KIND_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return kind
    if unit in ("days",):
        return "retention_period"
    if unit in ("ms", "sec"):
        return "response_time"
    if unit in ("rps", "rpm"):
        return "throughput"
    return "generic"


def _norm_unit(raw: str) -> Optional[str]:
    if not raw:
        return None
    return _UNIT_NORM.get(raw.strip(), raw.strip().lower())


# PR-K P2: regex patterns that, when found near a numeric match, mark it as
# a document/standard reference rather than a measurable constraint.
# Examples ловятся:
#   "ГОСТ 19.301-79"             → numbers 19.301, 79 pulled in as constraints
#   "согласно ГОСТ 34.601-90"    → 34.601, 90
#   "[18]"                       → 18 (markdown footnote ref)
#   "пункт 4.1.1"                → 4.1, 1
#   "п. 5.1"                     → 5.1
#   "статья 153 УК РФ"           → 153
# Без фильтра эти числа летят в Constraint(kind="generic", unit=None) и
# мусорят `_find_numeric_conflict` + `uncoveredAspects` в UI.
_REFERENCE_CONTEXT_RES: list[re.Pattern] = [
    # ГОСТ + (опц. латиница R/ИСО) + цифры (с точкой/дефисом)
    re.compile(r"\bГОСТ\b(?:\s*[РR])?\s*\d", re.I | re.UNICODE),
    re.compile(r"\bГОСТ\s*ИСО\b", re.I | re.UNICODE),
    # ИСО / ISO + цифры
    re.compile(r"\b(?:ИСО|ISO|МЭК|IEC)\s*\d", re.I | re.UNICODE),
    # Markdown / academic footnote refs: [12], [18, c.5]
    re.compile(r"\[\s*\d+(?:\s*[,;]\s*[^\]]*)?\s*\]"),
    # пункт / п. / подпункт / подп. + цифры
    re.compile(r"\b(?:пункт\w*|п\.|подпункт\w*|подп\.|раздел\w*)\s*\d", re.I | re.UNICODE),
    # статья / статьёй / статьи / ст. (statutes — all morphological cases)
    re.compile(r"\b(?:стать[яёеию]\w*|ст\.)\s*\d", re.I | re.UNICODE),
    # таблица / рисунок / приложение N
    re.compile(r"\b(?:таблиц\w+|рисун\w+|приложен\w+)\s*\d", re.I | re.UNICODE),
]


def _is_reference_number(text: str, match_start: int, match_end: int) -> bool:
    """Return True when the numeric match at [match_start, match_end] in
    `text` is a document / section / standard reference rather than a
    measurable constraint. Uses a 30-char window before and 5 after."""
    win_start = max(0, match_start - 30)
    win_end = min(len(text), match_end + 5)
    window = text[win_start:win_end]
    for pat in _REFERENCE_CONTEXT_RES:
        if pat.search(window):
            return True
    return False


def _extract_constraints(text: str) -> List[Constraint]:
    constraints: List[Constraint] = []
    text_lower = text.lower()

    # Check for word-based operators in the text
    op_override: Optional[str] = None
    for phrase, sym in sorted(_OPERATOR_WORDS.items(), key=lambda x: -len(x[0])):
        if phrase in text_lower:
            op_override = sym
            break

    seen: set = set()
    for m in _CONSTRAINT_RE.finditer(text):
        value_str = m.group("value")
        if not value_str:
            continue
        value = float(value_str.replace(",", "."))
        raw_unit = m.group("unit") or ""
        norm_unit = _norm_unit(raw_unit) if raw_unit else None
        op_sym = m.group("op_sym") or op_override or "="

        # PR-K P2: skip GOST / standard / section / footnote numbers — they
        # are document references, not measurable constraints. Only applied
        # to unit-less matches because a measurement like "5 секунд" can
        # never be a section ref. Real-package symptom (Polyakov 0.41::sent4):
        # extracted seven bogus constraints from "ГОСТ 19.301-79 [18] п.4.1.1".
        if norm_unit is None and _is_reference_number(text, m.start("value"), m.end("value")):
            continue

        # Use surrounding 60 chars as kind context
        start = max(0, m.start() - 60)
        context = text[start : m.end() + 20]
        kind = _determine_kind(context, norm_unit)

        key = (kind, op_sym, value, norm_unit)
        if key not in seen:
            seen.add(key)
            constraints.append(
                Constraint(kind=kind, operator=op_sym, value=value, unit=norm_unit)
            )
    return constraints


# Entity stop-list: words/phrases that the capitalisation-based extractor
# picks up but which carry no discriminative meaning. They appear in
# virtually every TZ / PMI — leaving them in the entity set poisons the
# entity-overlap score in PairVerifier (any two random sentences would
# "share" these tokens).
#
# Compared case-insensitively and also against the lemmatised form when
# pymorphy3 is available.
_ENTITY_STOPLIST = {
    # Document/structure chrome
    "система", "программа", "приложение", "подсистема", "подсистемы",
    "требование", "требования", "документ", "раздел", "подраздел",
    "пункт", "подпункт", "глава", "параграф", "пример", "таблица",
    "рисунок", "приложение", "настоящий", "данный", "указанный",
    "соответствующий", "следующий",
    # Generic actors / objects
    "пользователь", "оператор", "администратор", "разработчик",
    "исполнитель", "заказчик", "клиент",
    # Common words that get capitalised at sentence start
    "необходимо", "следует", "требуется", "должен", "должна", "должно",
    "обязательно", "рекомендуется", "допускается",
    # Document metadata
    "гост", "фстэк", "фсб", "ту",
}


def _is_stop_entity(term: str) -> bool:
    """True if `term` or its lemma is in the entity stop-list."""
    low = term.lower().strip()
    if low in _ENTITY_STOPLIST:
        return True
    # Multi-word terms: drop only if EVERY word is a stop-entity lemma —
    # "Пользователь системы" is all-stop and drops; "Модуль Учёта" stays.
    words = low.split()
    if not words:
        return True
    lemmas = [_lemma(w) for w in words]
    return all(l in _ENTITY_STOPLIST for l in lemmas)


def _lemma(word: str) -> str:
    """Thin wrapper — avoids importing the lemmatiser at module top-level
    (keeps the circular-import risk zero if build_requirements is ever
    imported from a lemma consumer)."""
    from app.core.lemmatize import lemma as _lemma_impl
    return _lemma_impl(word)


# spaCy noun-chunk extractor: lazy-loaded so the regex fallback path
# stays clean when spaCy / ru_core_news_md isn't installed. Loading is
# expensive (~500ms once), so we cache the pipeline at module level.
# `_SPACY_NLP` ∈ { None (not yet probed), False (probed, unavailable),
# spacy.Language (loaded) }.
_SPACY_NLP = None


def _get_spacy_nlp():
    """Return loaded spaCy pipeline or False if unavailable.

    Resolution order:
      1. CQUALITY_DISABLE_SPACY=1 in env → forced fallback (tests / CI).
      2. import spacy → spacy.load("ru_core_news_md") → cached.
      3. ImportError / OSError on missing model → cached False, regex
         fallback used forever after for this process.
    """
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if os.environ.get("CQUALITY_DISABLE_SPACY", "").strip() in ("1", "true", "yes"):
        _SPACY_NLP = False
        return False
    try:
        import spacy

        nlp = spacy.load(
            "ru_core_news_md",
            # Tagger is needed for noun_chunks; disable parser/NER to keep
            # the pipeline fast — we only consume noun_chunks + lemma_.
            disable=["parser", "ner", "attribute_ruler"],
        )
        _SPACY_NLP = nlp
        logger.info("entity extractor: spaCy ru_core_news_md loaded")
        return nlp
    except Exception as exc:
        # Broad except: spaCy can raise ImportError / OSError / RuntimeError /
        # ValueError depending on what's broken (missing core, missing model,
        # numpy/torch version mismatch, model file corruption). In any of
        # those cases we want the regex fallback, never a 500 for the API.
        logger.info(
            "entity extractor: spaCy unavailable (%s: %s); using regex "
            "fallback. Install with: pip install spacy && python -m spacy "
            "download ru_core_news_md",
            type(exc).__name__, exc,
        )
        _SPACY_NLP = False
        return False


def _extract_entities_regex(text: str) -> List[str]:
    """Legacy regex extractor — Title-Case + acronyms only.

    Misses lower-case domain phrases ("методы испытаний", "входные данные",
    "пользовательский интерфейс") because Russian doesn't capitalize them.
    Used as fallback when spaCy isn't available.
    """
    domain_terms = re.findall(
        r"\b([А-ЯЁ][а-яё]+(?:\s+[а-яёА-ЯЁ]+){0,2}|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,1})\b",
        text,
    )
    acronyms = re.findall(r"\b([А-ЯЁ]{2,}|[A-Z]{2,})\b", text)
    seen: set = set()
    result: List[str] = []
    for term in domain_terms + acronyms:
        t = term.strip()
        if not t or len(t) <= 2:
            continue
        if _is_stop_entity(t):
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(t)
    return result[:15]


def _spacy_noun_chunks_ru(doc) -> List[Tuple[str, str]]:
    """POS-tag-based noun-phrase extraction for Russian.

    spaCy's built-in `doc.noun_chunks` is **not implemented for ru**
    (raises NotImplementedError E894 — known spaCy limitation: only
    en/de/fr/es/pt/el/sv/nb/zh ship with a noun-chunk syntax iterator).
    Workaround: walk tokens, group consecutive ADJ/NOUN/PROPN spans
    into chunks. The Russian tagger works fine — only the chunk
    iterator is missing.

    Returns a list of (surface_form, lemma_key) tuples. Lemma key uses
    space-joined lowercase lemmas of non-stop tokens — same key shape
    spaCy noun_chunks produced, so downstream dedup is unchanged.
    """
    chunks: List[Tuple[str, str]] = []
    current_tokens = []  # spaCy Token objects

    def _flush():
        if not current_tokens:
            return
        # Build surface form preserving spacing, lemma key for dedup.
        surface = "".join(t.text_with_ws for t in current_tokens).strip()
        lemma_parts = [
            t.lemma_.lower() for t in current_tokens
            if not t.is_stop and not t.is_punct and t.lemma_.strip()
        ]
        if surface and lemma_parts:
            chunks.append((surface, " ".join(lemma_parts)))
        current_tokens.clear()

    for token in doc:
        # Allow ADJ/NOUN/PROPN as chunk content. ADP (предлог) breaks
        # the chunk: "перечень функций программы" — three NOUNs in a row,
        # no preposition between them, all stay in one chunk; "защита от
        # инъекций" — ADP "от" cuts; we get "защита" + "инъекций" as
        # separate chunks (acceptable — both still useful for dedup).
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
            current_tokens.append(token)
        else:
            _flush()
    _flush()
    return chunks


def _extract_entities(text: str) -> List[str]:
    """Extract domain entities from a requirement / coverage-unit text.

    Audit (Polyakov: every false `Near-zero entity overlap` demotion):
    Russian technical text doesn't capitalize domain nouns, so the old
    Title-Case regex returned ~0-2 entities for typical reqs and Rule 5
    (verify_pairs) demoted genuine PARTIAL pairs as off-topic. spaCy's
    POS tagger groups consecutive NOUN/PROPN/ADJ tokens, so "перечень
    функций программы", "методы испытаний", "входные данные" all extract
    correctly. Lemmas let the same phrase match across morphological variants.

    Resolution order:
      1. spaCy POS-tag noun-phrase scan (lemmatised) — preferred path.
      2. Acronym sweep (spaCy may not surface short ALL-CAPS as nouns).
      3. Regex fallback when spaCy isn't installed / model isn't downloaded.
    """
    if not text or not text.strip():
        return []

    nlp = _get_spacy_nlp()
    if not nlp:
        return _extract_entities_regex(text)

    seen: set = set()
    result: List[str] = []
    try:
        doc = nlp(text)
        chunks = _spacy_noun_chunks_ru(doc)
    except Exception as exc:
        # Defensive: any spaCy/tagger failure → regex fallback rather
        # than losing the whole requirement / failing the API request.
        logger.warning(
            "entity extractor: spaCy crashed on text (%s: %s); regex fallback",
            type(exc).__name__, exc,
        )
        return _extract_entities_regex(text)

    for surface, key in chunks:
        # Skip 1-letter / overly long chunks; cap len at 5 words.
        words_in_chunk = len(surface.split())
        if not key or len(key) <= 2 or words_in_chunk > 5:
            continue
        if key in seen:
            continue
        if _is_stop_entity(key):
            continue
        seen.add(key)
        result.append(surface)

    # Acronym sweep — spaCy may treat "REST" / "JSON" inconsistently.
    for acronym in re.findall(r"\b([А-ЯЁ]{2,}|[A-Z]{2,})\b", text):
        key = acronym.lower()
        if key not in seen and len(acronym) > 2:
            seen.add(key)
            result.append(acronym)

    return result[:15]


def _is_requirement_fragment(text: str) -> bool:
    return bool(_TRIGGER_RE.search(text)) and len(text.split()) >= 5


# ---------------------------------------------------------------------------
# Compound requirement splitting
# ---------------------------------------------------------------------------
#
# A common TZ shape is:
#     "Приложение должно предоставлять следующие функции:
#        • Регистрация и управление учётной записью;
#        • Просмотр списка проектов;
#        • Создание нового проекта."
#
# Until now we extracted this as ONE RequirementUnit. The cross-encoder
# judge would find PMI steps that test ANY of the sub-items and return
# COVERED, which the manual v3/v4 review flagged as misleading — the
# requirement is really compound and should be evaluated per-item.
#
# Heuristic splitter: find a colon, detect a consistent list marker in the
# body (bullets, numbered items, semi-colon separated dashes), split, and
# prepend the intro to each item so the sub-requirement remains
# self-contained. Modality / requirement_type / entities are re-extracted
# per sub-item by the normal helpers.
#
# Conservative: at least 2 items, each ≥ 2 words, each sharing the intro's
# subject line. If any check fails we fall back to the original text.

_INTRO_COLON_RE = re.compile(r":\s*", re.UNICODE)

# A marker that, appearing ≥ 2 times, signals an enumerated list. Order is
# important: bullets are unambiguous; semicolons are a last-resort signal.
_LIST_MARKERS = [
    # Cyrillic / Latin bullets glyphs
    re.compile(r"(?:(?<=\n)|(?<=\s)|^)\s*[•●◦▪]\s+", re.MULTILINE),
    # En-dash / em-dash bullet
    re.compile(r"(?:(?<=\n)|(?<=;))\s*[\u2013\u2014]\s+"),
    # Leading "1) " or "1. "
    re.compile(r"(?:(?<=\n)|(?<=\s)|^)\s*\d+[\.\)]\s+", re.MULTILINE),
    # Leading "а) " or "а."
    re.compile(r"(?:(?<=\n)|(?<=\s)|^)\s*[а-яё][\.\)]\s+", re.MULTILINE),
    # Semicolon between non-empty phrases (≥ 3 words each side)
    re.compile(r";\s*"),
]


def _split_compound_requirement(text: str) -> List[str]:
    """
    Return a list of atomic requirement strings.

    - Single-statement requirement     → [original_text]
    - Compound with ≥ 2 list items     → [intro + item_1, intro + item_2, …]

    The intro is preserved verbatim so each sub-requirement is
    grammatically complete and retrieval/judge get useful context.
    """
    if not text:
        return [text]
    stripped = text.strip()

    # Find the MAIN colon that introduces the list. Skip colons inside
    # parentheses (common in citations).
    colon_match = None
    depth = 0
    for i, ch in enumerate(stripped):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0 and i + 1 < len(stripped):
            colon_match = i
            break
    if colon_match is None:
        return [stripped]

    intro = stripped[:colon_match].strip()
    body = stripped[colon_match + 1 :].strip()
    if len(intro.split()) < 3 or len(body.split()) < 4:
        return [stripped]

    # Try each marker in priority order; first one that yields ≥ 2
    # reasonable items wins.
    for marker in _LIST_MARKERS:
        parts = [p.strip(" \t\n.,;:") for p in marker.split(body)]
        # Drop empty fragments (the first one is usually empty when the
        # marker is at the very start of body).
        parts = [p for p in parts if p]
        # Each part must look like a real item and not be absurdly long
        # (cap at 40 words — beyond that it's probably a full sentence,
        # not a list entry). Single-word items are allowed when they are
        # substantive (≥ 5 chars, to exclude stray "и" / "или" noise).
        def _valid_item(p: str) -> bool:
            n_words = len(p.split())
            if n_words > 40:
                return False
            if n_words >= 2:
                return True
            # 1-word item: accept if the word itself is substantive
            return len(p.strip()) >= 5
        parts = [p for p in parts if _valid_item(p)]
        if len(parts) >= 2:
            # Rebuild full sentences: "intro: item"
            return [f"{intro}: {p}" for p in parts]

    return [stripped]


def _section_allows_candidate(
    section_id: Optional[str],
    modality: Modality,
    metadata: Optional[dict] = None,
) -> bool:
    """
    Gate for whether a candidate/fragment from a given section should be included
    in the RequirementUnit set for C-quality coverage analysis.

    Priority:
    1. prepare-service sectionCategory (gold signal from docback metadata):
       - "requirements"  → always allow (explicit requirements chapter).
       - any other value → require MUST / MUST_NOT / SHOULD modality.
         Prevents false positives from "other" / "metadata" / "environment"
         sections in unstructured TZs (Череухо-class documents where every
         section is tagged "other" and heuristic-only extraction inflates
         the requirement list).
    2. GOST TZ section numbering (fallback when sectionCategory absent):
       - Section 4.x → always allow.
       - Sections 1–3 → require explicit modality.
       - Unknown numbering (None) → require explicit modality.
         Tightened from previous "be permissive" to avoid admitting noise
         from non-GOST structured TZs via section_id alone.
    """
    _MODALITY_REQUIRED = (Modality.MUST, Modality.MUST_NOT, Modality.SHOULD)
    meta = metadata or {}

    # ── 0. Trust docback's consistency-policy verdict ────────────────────
    # docback's `scoreCandidateForConsistency` (prepared_builder.go) already
    # walks the section tree via parent_id and applies an ancestor-aware
    # scoring that incorporates "Требования к программе" ⇒ subsections like
    # "Интерфейс студента / Организация входных данных". When that policy
    # admits a candidate it stamps `isRequirementLikeForConsistency=true`
    # in metadata. This gate would otherwise drop those deeply-nested
    # subsection candidates because their leaf `sectionCategory` is
    # "other" / "input_output" (prepare-service classifies on leaf title
    # only, with no ancestor walk) and the requirement may be phrased
    # without an explicit «должен» modal (e.g. "Программа предоставляет
    # студенту возможность подачи заявки на конкурс." — declarative
    # functional spec, legitimate requirement in a level-3 subsection).
    # Kurmanova-regression: 34 reqCandidates → 0 units before this gate;
    # 34 → 34 after.
    if meta.get("isRequirementLikeForConsistency") is True:
        return True

    # ── 1. Gold signal: sectionCategory from prepare-service ─────────────
    section_category = meta.get("sectionCategory", "")
    if section_category:
        if section_category == "requirements":
            return True
        # Any other tagged category (other / metadata / environment /
        # test_steps) is explicitly NOT the requirements chapter.
        return modality in _MODALITY_REQUIRED

    # ── 2. GOST numbering heuristic (legacy / test data without metadata) ─
    relevance = _is_requirement_section(section_id)
    if relevance is True:
        return True
    # Both the definitive-no (sections 1–3) and the ambiguous (None) cases
    # require explicit modality.  "Unknown" no longer means "permissive" —
    # a section we cannot classify is more likely unrelated prose than a
    # requirements chapter.
    return modality in _MODALITY_REQUIRED


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


class RequirementBuilder:
    """Builds RequirementUnit list from a single TZ PreparedArtifact.

    Extraction modes (see `CoverageConfig.requirement_extraction`):
      - "sections"   — trust only sections hierarchy; re-segment text
      - "candidates" — trust prepare-service's requirement_candidates[]
      - "fragments"  — heuristic over fragments[] with trigger-word filter
      - "auto"       — candidates → fragments → sections (first non-empty)
    """

    def __init__(self, config: Optional[CoverageConfig] = None) -> None:
        mode = (config.requirement_extraction if config else "auto").lower()
        if mode not in {"auto", "sections", "candidates", "fragments", "model"}:
            mode = "auto"
        self._mode = mode
        self._config = config or CoverageConfig()

    def build(self, artifact: dict) -> List[RequirementUnit]:
        doc_id = artifact.get("document_id", "unknown")

        if self._mode == "model":
            logger.info("Building requirements with fine-tuned classifier in %s", doc_id)
            return self._from_model(artifact)

        if self._mode == "sections":
            logger.info("Building requirements from sections in %s", doc_id)
            return self._from_sections(artifact)

        if self._mode == "candidates":
            candidates = artifact.get("requirement_candidates") or []
            return self._from_candidates(artifact, candidates)

        if self._mode == "fragments":
            fragments = artifact.get("fragments") or []
            return self._from_fragments(artifact, fragments)

        # auto
        candidates = artifact.get("requirement_candidates") or []
        primary_units: List[RequirementUnit] = []
        primary_source = "(none)"
        if candidates:
            logger.info("Building requirements from %d candidates in %s", len(candidates), doc_id)
            primary_units = self._from_candidates(artifact, candidates)
            primary_source = "candidates"

        fragments = artifact.get("fragments") or []
        if not primary_units and fragments:
            logger.info(
                "No usable candidates in %s; falling back to %d fragments",
                doc_id, len(fragments),
            )
            primary_units = self._from_fragments(artifact, fragments)
            primary_source = "fragments"

        if not primary_units:
            logger.info(
                "Fragments yielded nothing in %s; falling back to section-driven extraction",
                doc_id,
            )
            return self._from_sections(artifact)

        # PR-K post-fix (c): section-aware boost.
        #
        # Real-package symptoms:
        #   * Cherevuyhho (78 reqs from candidates) — almost all glued
        #     to the heading-less `preamble` section because the .docx
        #     headings were not Word-styled. Section-level type-aware
        #     applicability cannot work without per-requirement
        #     section_id, so the C-quality grade is too lenient.
        #   * Polyakov (52 reqs from candidates) — cleaner structure,
        #     but the BERT classifier still drops the occasional
        #     sentence inside an EXPLICITLY requirement-bearing section
        #     ("Требования к ..."). Section-driven extraction picks
        #     them up.
        #
        # Strategy: after the primary path produces some units, run a
        # second pass over sections classified as `True` by
        # `_classify_section` (definitely requirement-bearing) and
        # admit any sentence with a modality / trigger that the primary
        # path missed. Dedup against primary_units by normalised text.
        #
        # Off-switch: env var CQUALITY_SECTION_BOOST=false disables it.
        # Default = on for `auto` mode only (other modes are explicit
        # and should not be silently augmented).
        boost_enabled = (
            os.environ.get("CQUALITY_SECTION_BOOST", "true").strip().lower()
            not in ("false", "0", "no", "off")
        )
        if not boost_enabled:
            return primary_units

        boost_units = self._section_boost(artifact, primary_units)
        if boost_units:
            logger.info(
                "[%s] section-boost: primary=%d (%s) + boost=%d = %d total",
                doc_id, len(primary_units), primary_source,
                len(boost_units), len(primary_units) + len(boost_units),
            )
            return primary_units + boost_units

        return primary_units

    def _section_boost(
        self,
        artifact: dict,
        primary_units: List[RequirementUnit],
    ) -> List[RequirementUnit]:
        """Section-aware merge-pass over sections classified as True
        ("definitely a requirement section"). Returns ONLY the units that
        primary extraction missed (dedup-by-normalised-text).

        Conservative: only fires on sections where `_classify_section`
        returns True (numbering 4.x or title contains explicit keyword
        like 'требования к ...'). Skips ambiguous sections (relevance
        is None) — those need stricter modality gate which the primary
        path already implements.
        """
        doc_id = artifact.get("document_id", "unknown")
        sections = artifact.get("sections") or []
        fragments = artifact.get("fragments") or []
        if not sections or not fragments:
            return []

        # Build a set of normalised texts already accepted by primary
        # extraction. Used for dedup.
        primary_norms = {u.normalized_text for u in primary_units}

        # Group fragment text by section_id (preserve order).
        frags_by_section: Dict[str, List[str]] = {}
        titles_by_id: Dict[str, str] = {}
        for s in sections:
            sid = s.get("section_id")
            if sid:
                titles_by_id[sid] = s.get("title") or ""
        for frag in fragments:
            sid = frag.get("section_id")
            text = (frag.get("text") or "").strip()
            if sid and text:
                frags_by_section.setdefault(sid, []).append(text)

        boosted: List[RequirementUnit] = []
        seen_req_ids = {u.req_id for u in primary_units}

        for section in sections:
            sid = section.get("section_id")
            title = section.get("title") or ""
            if not sid:
                continue
            relevance = _classify_section(sid, title)
            # Strict: only sections with relevance=True (definitely
            # requirement-bearing). Ambiguous (None) is skipped to
            # avoid false-positives on Введение / Цель / Аналоги.
            if relevance is not True:
                continue

            section_text = "\n".join(frags_by_section.get(sid, []))
            if not section_text:
                continue

            sentences = _split_sentences_ru(section_text)
            if not sentences:
                continue

            for idx, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent.split()) < 5:
                    continue
                if _is_document_boilerplate(sent):
                    continue
                modality = _extract_modality(sent)
                has_trigger = bool(_TRIGGER_RE.search(sent))
                # Strict gate inside this boost pass — must have either
                # explicit modality OR a trigger word. The primary path
                # may have skipped this sentence because the BERT model
                # was uncertain; we admit it ONLY if it looks like a
                # requirement at the modality / trigger level.
                if not (has_trigger or modality != Modality.UNKNOWN):
                    continue
                # Skip if normalised form duplicates a primary unit
                # (case where ML candidate text and section-extracted
                # sentence are the same content with different
                # whitespace / punctuation).
                norm = _normalize_text(sent)
                if norm in primary_norms:
                    continue
                primary_norms.add(norm)

                base_id = f"{doc_id}::{sid}::boost::s{idx}"
                req_id = base_id
                dedup = 0
                while req_id in seen_req_ids:
                    dedup += 1
                    req_id = f"{base_id}::{dedup}"
                seen_req_ids.add(req_id)

                boosted.append(
                    RequirementUnit(
                        req_id=req_id,
                        source_document_id=doc_id,
                        source_section_id=sid,
                        source_fragment_id=None,
                        text=sent,
                        normalized_text=norm,
                        requirement_type=_extract_requirement_type(sent, title),
                        modality=modality,
                        entities=_extract_entities(sent),
                        constraints=_extract_constraints(sent),
                        metadata={
                            "section_title": title,
                            "boost": True,
                        } if title else {"boost": True},
                    )
                )

        return boosted

    # ------------------------------------------------------------------

    def _from_candidates(self, artifact: dict, candidates: List[dict]) -> List[RequirementUnit]:
        doc_id = artifact.get("document_id", "unknown")
        units: List[RequirementUnit] = []

        # Detect document-level req_id bug: all candidates share the same req_id.
        # When this happens, raw_req_id is not usable as a stable per-candidate
        # identifier, so we fall back to position-based IDs for every candidate
        # (fragment_id-based IDs are still preferred when available).
        raw_ids = [c.get("req_id") for c in candidates if c.get("req_id")]
        _all_same_req_id = bool(len(raw_ids) > 1 and len(set(raw_ids)) == 1)
        if _all_same_req_id:
            logger.warning(
                "All %d requirement_candidates share the same req_id=%r in document %s; "
                "falling back to position-based IDs (fragment_id used where available).",
                len(candidates), raw_ids[0], doc_id,
            )

        seen_req_ids: set = set()
        n_boilerplate_dropped = 0
        for i, cand in enumerate(candidates):
            text = (cand.get("text") or "").strip()
            if not text:
                continue

            # BUG-06 fix: apply the same boilerplate filter the model path uses
            # (form stamps, document-code lines, GOST-citation-only lines,
            # glossary entries, "– М.: Изд-во стандартов, 1997"). The filter
            # is modality-aware via _is_document_boilerplate's has_modality
            # checks, so genuine requirements that mention ГОСТ ("должны
            # соответствовать ГОСТ 19.101") still pass.
            if _is_document_boilerplate(text):
                n_boilerplate_dropped += 1
                logger.debug(
                    "Skipping boilerplate candidate in %s: %r", doc_id, text[:120]
                )
                continue

            # Build deterministic unique req_id:
            #   1. fragment_id  →  "{doc_id}::{fragment_id}"   (stable, per-fragment)
            #   2. raw req_id from candidate, only when it is unique across all
            #      candidates — if all share the same req_id that value is useless
            #   3. position-based fallback
            frag_id = cand.get("fragment_id")
            raw_req_id = cand.get("req_id")
            if frag_id:
                base_id = f"{doc_id}::{frag_id}"
            elif raw_req_id and not _all_same_req_id:
                base_id = raw_req_id
            else:
                base_id = f"{doc_id}::cand::{i}"

            req_id = base_id
            if base_id in seen_req_ids:
                req_id = f"{base_id}::{i}"
                logger.warning(
                    "req_id collision in document %s at index %d; using %s", doc_id, i, req_id
                )
            seen_req_ids.add(req_id)

            section_id = cand.get("section_id")
            modality = _extract_modality(text)
            cand_meta = cand.get("metadata") or {}
            section_title = cand_meta.get("sectionTitle") or cand_meta.get("section_title") or ""

            if not _section_allows_candidate(section_id, modality, cand_meta):
                logger.debug(
                    "Skipping candidate in non-requirement section %r "
                    "(sectionCategory=%r modality=%s) in %s",
                    section_id, cand_meta.get("sectionCategory"), modality, doc_id,
                )
                continue

            # FIX-C4: exclude documentation-deliverable requirements.
            # prepare-service tags candidates that describe WHAT DOCUMENTS to
            # produce (not what the program must do) as "documentation_requirement".
            # docback maps them to ctRequirement so they reach C-quality, but they
            # must not count as program requirements in coverage analysis.
            # The original prepare-service type is preserved in metadata["prepareType"].
            prepare_type = cand_meta.get("prepareType", "")
            if prepare_type == "documentation_requirement":
                logger.debug(
                    "Skipping documentation_requirement candidate in %s (section %r): %r",
                    doc_id, section_id, text[:100],
                )
                continue

            units.append(
                RequirementUnit(
                    req_id=req_id,
                    source_document_id=doc_id,
                    source_section_id=section_id,
                    source_fragment_id=frag_id,
                    text=text,
                    normalized_text=_normalize_text(text),
                    requirement_type=_extract_requirement_type(text, section_title),
                    modality=modality,
                    entities=_extract_entities(text),
                    constraints=_extract_constraints(text),
                    metadata=cand_meta,
                )
            )

        if n_boilerplate_dropped:
            logger.info(
                "[%s] _from_candidates dropped %d boilerplate candidates",
                doc_id, n_boilerplate_dropped,
            )
        return units

    def _from_fragments(self, artifact: dict, fragments: List[dict]) -> List[RequirementUnit]:
        doc_id = artifact.get("document_id", "unknown")
        units: List[RequirementUnit] = []
        n_boilerplate_dropped = 0
        for frag in fragments:
            text = (frag.get("text") or "").strip()
            if not text or not _is_requirement_fragment(text):
                continue
            # BUG-06 fix: same boilerplate filter as in _from_model and
            # _from_candidates. Modality-aware so genuine ГОСТ-referencing
            # requirements are kept.
            if _is_document_boilerplate(text):
                n_boilerplate_dropped += 1
                logger.debug(
                    "Skipping boilerplate fragment in %s: %r", doc_id, text[:120]
                )
                continue
            section_id = frag.get("section_id")
            modality = _extract_modality(text)
            frag_meta = frag.get("metadata") or {}
            section_title = frag_meta.get("sectionTitle") or frag_meta.get("section_title") or ""
            if not _section_allows_candidate(section_id, modality, frag_meta):
                logger.debug(
                    "Skipping fragment in non-requirement section %r "
                    "(sectionCategory=%r modality=%s) in %s",
                    section_id, frag_meta.get("sectionCategory"), modality, doc_id,
                )
                continue
            units.append(
                RequirementUnit(
                    source_document_id=doc_id,
                    source_section_id=section_id,
                    source_fragment_id=frag.get("fragment_id"),
                    text=text,
                    normalized_text=_normalize_text(text),
                    requirement_type=_extract_requirement_type(text, section_title),
                    modality=modality,
                    entities=_extract_entities(text),
                    constraints=_extract_constraints(text),
                    metadata=frag_meta,
                )
            )
        if n_boilerplate_dropped:
            logger.info(
                "[%s] _from_fragments dropped %d boilerplate fragments",
                doc_id, n_boilerplate_dropped,
            )
        if not units:
            logger.warning("No requirement fragments found in %s", doc_id)
        return units

    # ------------------------------------------------------------------

    def _from_sections(self, artifact: dict) -> List[RequirementUnit]:
        """
        Section-driven extraction.

        Trust zone: `sections[]` hierarchy and each fragment's `section_id`
        binding. Everything else about prepare-service fragment splits is
        ignored — inside each requirement section we concatenate fragment
        text and re-split into sentences locally.
        """
        doc_id = artifact.get("document_id", "unknown")
        sections = artifact.get("sections") or []
        fragments = artifact.get("fragments") or []

        if not sections:
            logger.warning(
                "Section-driven extraction requested but no sections[] in %s; "
                "falling back to fragments path", doc_id,
            )
            return self._from_fragments(artifact, fragments)

        titles_by_id: Dict[str, str] = {
            s.get("section_id"): (s.get("title") or "")
            for s in sections
            if s.get("section_id")
        }

        # Group fragment text by section_id (preserve order of appearance).
        frags_by_section: Dict[str, List[str]] = {}
        for frag in fragments:
            sid = frag.get("section_id")
            text = (frag.get("text") or "").strip()
            if not sid or not text:
                continue
            frags_by_section.setdefault(sid, []).append(text)

        units: List[RequirementUnit] = []
        seen_req_ids: set = set()

        for section in sections:
            sid = section.get("section_id")
            title = section.get("title") or ""
            if not sid:
                continue
            relevance = _classify_section(sid, title)
            if relevance is False:
                logger.debug("Section %r (%s): non-requirement, skipping", sid, title[:40])
                continue
            # None (ambiguous) → be permissive: admit the section but rely on
            # sentence-level modality / trigger filter below.

            section_text = "\n".join(frags_by_section.get(sid, []))
            if not section_text:
                continue

            sentences = _split_sentences_ru(section_text)
            if not sentences:
                continue

            section_has_req_marker = relevance is True
            for idx, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent.split()) < 5:
                    continue
                modality = _extract_modality(sent)
                has_trigger = bool(_TRIGGER_RE.search(sent))

                # Acceptance policy inside a section:
                #  - requirement section (relevance=True): admit any sentence
                #    with a trigger word OR explicit modality
                #  - ambiguous section (relevance=None): stricter — require
                #    explicit modality (trigger alone is not enough to promote
                #    a random sentence from e.g. a rationale paragraph)
                if section_has_req_marker:
                    if not (has_trigger or modality != Modality.UNKNOWN):
                        continue
                else:
                    if modality not in (Modality.MUST, Modality.MUST_NOT, Modality.SHOULD):
                        continue

                base_id = f"{doc_id}::{sid}::s{idx}"
                req_id = base_id
                dedup = 0
                while req_id in seen_req_ids:
                    dedup += 1
                    req_id = f"{base_id}::{dedup}"
                seen_req_ids.add(req_id)

                units.append(
                    RequirementUnit(
                        req_id=req_id,
                        source_document_id=doc_id,
                        source_section_id=sid,
                        source_fragment_id=None,
                        text=sent,
                        normalized_text=_normalize_text(sent),
                        requirement_type=_extract_requirement_type(sent, title),
                        modality=modality,
                        entities=_extract_entities(sent),
                        constraints=_extract_constraints(sent),
                        metadata={"section_title": title} if title else {},
                    )
                )

        if not units:
            logger.warning(
                "Section-driven extraction produced 0 requirements in %s "
                "(sections=%d, fragments=%d)",
                doc_id, len(sections), len(fragments),
            )
        return units

    # ------------------------------------------------------------------

    # Lazily constructed classifier so callers who never touch the "model"
    # path do not pay the torch import cost. Reassignable from tests via
    # `RequirementBuilder._classifier_factory = lambda _: MockClassifier()`.
    _classifier = None

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier
        from app.infrastructure.ml.requirement_classifier import RequirementClassifier
        cfg = self._config.requirement_model
        self._classifier = RequirementClassifier(
            model_path=cfg.model_path,
            max_len=cfg.max_len,
            batch_size=cfg.batch_size,
        )
        return self._classifier

    def set_classifier(self, classifier) -> None:
        """Inject a classifier (primarily for tests and A/B experiments)."""
        self._classifier = classifier

    def _from_model(self, artifact: dict) -> List[RequirementUnit]:
        """
        Model-driven extraction.

        Trust zone: sections[] hierarchy (same as `_from_sections`) — we
        use it to skip the descriptive preamble / glossary sections, which
        are 20–40% of a TZ and would waste inference time. Inside every
        remaining section we re-split fragments into sentences, then run
        the fine-tuned classifier in a single batch per document. Sentences
        with P(is_requirement) >= threshold become RequirementUnits.

        Modality, requirement_type, entities and constraints continue to
        use the regex helpers — the classifier's job is just the binary
        gate, the other fields are feature extraction which the regexes
        do well enough on confirmed requirements.
        """
        doc_id = artifact.get("document_id", "unknown")
        sections = artifact.get("sections") or []
        fragments = artifact.get("fragments") or []

        threshold = self._config.requirement_model.threshold

        # Build index from section_id → title and collect per-section texts
        titles_by_id: Dict[str, str] = {
            s.get("section_id"): (s.get("title") or "")
            for s in sections
            if s.get("section_id")
        }
        frags_by_section: Dict[str, List[str]] = {}
        for frag in fragments:
            sid = frag.get("section_id")
            text = (frag.get("text") or "").strip()
            if sid and text:
                frags_by_section.setdefault(sid, []).append(text)

        # Candidate sentences — (section_id, sentence_idx, text, title)
        candidates: List[Tuple[str, int, str, str]] = []

        if sections:
            for section in sections:
                sid = section.get("section_id")
                title = (section.get("title") or "").strip()
                if not sid:
                    continue
                # Hard skip only for sections we are certain are non-req.
                # Unknown / req sections are both kept: the classifier is
                # the final arbiter, so we'd rather over-admit here than
                # miss requirements in an uncategorised section.
                if _classify_section(sid, title) is False:
                    continue
                section_text = "\n".join(frags_by_section.get(sid, []))
                for idx, sent in enumerate(_split_sentences_ru(section_text)):
                    sent = sent.strip()
                    if len(sent.split()) >= 5:
                        candidates.append((sid, idx, sent, title))
        else:
            # No sections → treat every fragment as a sentence source.
            for i, frag in enumerate(fragments):
                text = (frag.get("text") or "").strip()
                for idx, sent in enumerate(_split_sentences_ru(text)):
                    sent = sent.strip()
                    if len(sent.split()) >= 5:
                        candidates.append((f"frag::{i}", idx, sent, ""))

        if not candidates:
            logger.warning("Model extractor: no candidate sentences in %s", doc_id)
            return []

        try:
            classifier = self._get_classifier()
        except Exception as exc:
            logger.warning(
                "Model extractor: classifier unavailable (%s); falling back to sections path",
                exc,
            )
            return self._from_sections(artifact)

        texts = [c[2] for c in candidates]
        try:
            probs = classifier.predict_proba(texts)
        except Exception as exc:
            logger.warning(
                "Model extractor: inference failed (%s); falling back to sections path",
                exc,
            )
            return self._from_sections(artifact)

        units: List[RequirementUnit] = []
        seen_req_ids: set = set()
        skipped_boilerplate = 0
        compound_expansions = 0
        for (sid, idx, sent, title), p in zip(candidates, probs):
            if p < threshold:
                continue
            # Post-filter: the classifier is sometimes tricked by document
            # stamps, bibliography entries and project-background prose
            # that share surface features with requirements. Drop them
            # regardless of the classifier's confidence.
            if _is_document_boilerplate(sent):
                skipped_boilerplate += 1
                continue
            # Compound requirement detection: a single sentence of the form
            # "X должно предоставлять: • A; • B; • C" is split into one
            # sub-requirement per bullet so downstream judging evaluates
            # each aspect separately.
            sub_texts = _split_compound_requirement(sent)
            if len(sub_texts) > 1:
                compound_expansions += 1
            for sub_i, sub_text in enumerate(sub_texts):
                base_id = f"{doc_id}::{sid}::s{idx}"
                if len(sub_texts) > 1:
                    base_id = f"{base_id}::i{sub_i}"
                req_id = base_id
                dedup = 0
                while req_id in seen_req_ids:
                    dedup += 1
                    req_id = f"{base_id}::{dedup}"
                seen_req_ids.add(req_id)

                units.append(
                    RequirementUnit(
                        req_id=req_id,
                        source_document_id=doc_id,
                        source_section_id=sid,
                        source_fragment_id=None,
                        text=sub_text,
                        normalized_text=_normalize_text(sub_text),
                        requirement_type=_extract_requirement_type(sub_text, title),
                        modality=_extract_modality(sub_text),
                        entities=_extract_entities(sub_text),
                        constraints=_extract_constraints(sub_text),
                        metadata={
                            "section_title": title,
                            "classifier_score": round(float(p), 4),
                            "compound_source_idx": sub_i if len(sub_texts) > 1 else None,
                        },
                    )
                )

        logger.info(
            "Model extractor: %d/%d candidate sentences classified as requirements "
            "(threshold=%.2f, boilerplate_filtered=%d, compound_expansions=%d) in %s",
            len(units), len(candidates), threshold, skipped_boilerplate,
            compound_expansions, doc_id,
        )
        return units
