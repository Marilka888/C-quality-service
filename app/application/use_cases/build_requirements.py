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
from app.domain.c_quality_enums import Modality, RequirementType
from app.domain.c_quality_models import Constraint, RequirementUnit

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

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
        if t and t.lower() not in seen and len(t) > 2:
            seen.add(t.lower())
            result.append(t)
    return result[:15]  # cap to avoid noise


def _is_requirement_fragment(text: str) -> bool:
    return bool(_TRIGGER_RE.search(text)) and len(text.split()) >= 5


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
    """Builds RequirementUnit list from a single TZ PreparedArtifact."""

    def build(self, artifact: dict) -> List[RequirementUnit]:
        doc_id = artifact.get("document_id", "unknown")
        candidates = artifact.get("requirement_candidates") or []

        if candidates:
            logger.info("Building requirements from %d candidates in %s", len(candidates), doc_id)
            return self._from_candidates(artifact, candidates)

        fragments = artifact.get("fragments") or []
        logger.info(
            "No requirement_candidates in %s; falling back to %d fragments",
            doc_id,
            len(fragments),
        )
        return self._from_fragments(artifact, fragments)

    # ------------------------------------------------------------------

    def _from_candidates(self, artifact: dict, candidates: List[dict]) -> List[RequirementUnit]:
        doc_id = artifact.get("document_id", "unknown")
        units: List[RequirementUnit] = []

        # Detect document-level req_id bug: all candidates share the same req_id
        raw_ids = [c.get("req_id") for c in candidates if c.get("req_id")]
        if raw_ids and len(set(raw_ids)) == 1:
            logger.warning(
                "All %d requirement_candidates share the same req_id=%r in document %s; "
                "using fragment_id-based unique IDs instead.",
                len(candidates), raw_ids[0], doc_id,
            )

        seen_req_ids: set = set()
        for i, cand in enumerate(candidates):
            text = (cand.get("text") or "").strip()
            if not text:
                continue

            # Build deterministic unique req_id:
            #   1. fragment_id  →  "{doc_id}::{fragment_id}"   (stable, per-fragment)
            #   2. raw req_id from candidate                    (use as-is if unique)
            #   3. position-based fallback                      (last resort)
            frag_id = cand.get("fragment_id")
            raw_req_id = cand.get("req_id")
            if frag_id:
                base_id = f"{doc_id}::{frag_id}"
            elif raw_req_id:
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
