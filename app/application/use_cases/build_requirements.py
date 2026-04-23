"""
Stage 1 of the coverage pipeline: build RequirementUnit list from a TZ artifact.

Priority:
  1. prepared_artifact.requirement_candidates (if present and non-empty)
  2. Heuristic extraction from fragments[] via modality / constraint patterns
"""
from __future__ import annotations

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
    return False


_MUST_NOT_RE = re.compile(
    r"\b(не должен|не должна|не должны|запрещено|недопустимо|не допускается)\b", re.I
)
_MUST_RE = re.compile(
    r"\b(должен|должна|должны|необходимо|обязан|обязана|обязаны)\b", re.I
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


def _extract_requirement_type(text: str) -> RequirementType:
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


def _extract_entities(text: str) -> List[str]:
    # Keyword/noun-phrase extraction (MVP: title-cased words + domain nouns)
    domain_terms = re.findall(
        r"\b([А-ЯЁ][а-яё]+(?:\s+[а-яёА-ЯЁ]+){0,2}|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,1})\b",
        text,
    )
    # Also extract acronyms
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
    return result[:15]  # cap to avoid noise


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


def _section_allows_candidate(section_id: Optional[str], modality: Modality) -> bool:
    """
    Gate for whether a candidate/fragment from a given section should be included
    in the RequirementUnit set for C-quality MVP coverage analysis.

    Rules:
    - Requirement sections (4.x): always allowed
    - Non-requirement sections (1, 2, 3): only when explicit modality present
      (MUST / MUST_NOT / SHOULD) — rules out descriptive text with incidental triggers
    - Unknown section (no section_id or sections 5+): permissive — cannot filter
      without section information, so include the candidate as before
    """
    relevance = _is_requirement_section(section_id)
    if relevance is True:
        return True
    if relevance is None:
        # No section info available — cannot apply section filter; be permissive
        return True
    # Definitively non-requirement section (1, 2, 3): require explicit modality
    return modality in (Modality.MUST, Modality.MUST_NOT, Modality.SHOULD)


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
        if candidates:
            logger.info("Building requirements from %d candidates in %s", len(candidates), doc_id)
            units = self._from_candidates(artifact, candidates)
            if units:
                return units

        fragments = artifact.get("fragments") or []
        if fragments:
            logger.info(
                "No usable candidates in %s; falling back to %d fragments",
                doc_id,
                len(fragments),
            )
            units = self._from_fragments(artifact, fragments)
            if units:
                return units

        logger.info("Fragments yielded nothing in %s; falling back to section-driven extraction", doc_id)
        return self._from_sections(artifact)

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
        for i, cand in enumerate(candidates):
            text = (cand.get("text") or "").strip()
            if not text:
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

            if not _section_allows_candidate(section_id, modality):
                logger.debug(
                    "Skipping candidate in non-requirement section %r (modality=%s) in %s",
                    section_id, modality, doc_id,
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
                    requirement_type=_extract_requirement_type(text),
                    modality=modality,
                    entities=_extract_entities(text),
                    constraints=_extract_constraints(text),
                    metadata=cand.get("metadata") or {},
                )
            )
        return units

    def _from_fragments(self, artifact: dict, fragments: List[dict]) -> List[RequirementUnit]:
        doc_id = artifact.get("document_id", "unknown")
        units: List[RequirementUnit] = []
        for frag in fragments:
            text = (frag.get("text") or "").strip()
            if not text or not _is_requirement_fragment(text):
                continue
            section_id = frag.get("section_id")
            modality = _extract_modality(text)
            if not _section_allows_candidate(section_id, modality):
                logger.debug(
                    "Skipping fragment in non-requirement section %r (modality=%s) in %s",
                    section_id, modality, doc_id,
                )
                continue
            units.append(
                RequirementUnit(
                    source_document_id=doc_id,
                    source_section_id=section_id,
                    source_fragment_id=frag.get("fragment_id"),
                    text=text,
                    normalized_text=_normalize_text(text),
                    requirement_type=_extract_requirement_type(text),
                    modality=modality,
                    entities=_extract_entities(text),
                    constraints=_extract_constraints(text),
                    metadata=frag.get("metadata") or {},
                )
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
                        requirement_type=_extract_requirement_type(sent),
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
                        requirement_type=_extract_requirement_type(sub_text),
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
