"""
Stage 2: build CoverageUnit list from a PMI/PZ artifact.

Granularity: fragment-level (paragraph / list_item / test_step / etc.)
Reuses the same extraction helpers as build_requirements.

Fragment sources (tried in order):
  1. artifact["fragments"]   — primary; flat list from prepare-service
  2. artifact["sentences"]   — fallback; finer-grained source some versions emit
"""
from __future__ import annotations

import re
from typing import List, Tuple

from app.core.logging import get_logger
from app.domain.c_quality_enums import CoverageUnitType
from app.domain.c_quality_models import CoverageUnit
from app.application.use_cases.build_requirements import (
    _extract_constraints,
    _extract_entities,
    _normalize_text,
)

logger = get_logger(__name__)

# Minimum token count for a fragment to become a CoverageUnit.
# 2 = allow "Войти в систему." / "Ожидаемый результат:"; 1-token strings are noise.
_MIN_WORDS = 2
_SECTION_WINDOW_SIZE = 4
_MAX_WINDOW_CHARS = 1800
_PREFIX_DEDUP_CHARS = 400
_DEDUP_GROWTH_CHARS = 160

_COVERAGE_SIGNAL_RE = re.compile(
    r"\btypescript\b|\bangular\b|\bgithub\b|\bfigma\b|\brest\s*api\b|\bjson\b|"
    r"регистрац|авторизац|аутентификац|загруз|скачив|файл|ошиб|сервер|"
    r"баз[аы]\s+данн|пользовательск|интерфейс",
    re.IGNORECASE,
)

# Polyakov-regression: PMI fragments occasionally surface as list-intro
# stubs («Клиентская часть приложения должна обеспечивать возможность
# выполнения следующих функций:») or as the section heading itself
# («Требования к функциональным характеристикам.»). When these enter
# coverage units — particularly SECTION_WINDOW units that concatenate
# 4 adjacent fragments — they pollute the unit text:
#   * the list-intro adds «…:.»  noise that reads like a truncated
#     sentence to the LLM judge;
#   * the section heading adds a topical-but-not-coverage prefix that
#     dilutes Jaccard against the requirement.
# Both shapes are filtered out before paragraph and window unit
# construction. Mirror of Prepare's list-intro filter — kept independent
# so the C-quality side stays robust regardless of upstream cleanup.
# Polyakov-regression (2026-05-10): SECTION_WINDOW units bundled 4
# adjacent fragments via a fixed-size window, gluing distinct
# self-contained normative sentences («Система должна …») into one
# evidence card. Paragraph units already carry each such sentence
# atomically, so a window that contains 2+ self-contained requirements
# is pure noise on the wire and confuses the reviewer ("слишком много
# требований в одном кандидате"). Detect self-contained normative
# fragments — Subject + должен/должна/должно/должны head — and cap
# each window at one of them.
_SELF_CONTAINED_NORMATIVE_RE = re.compile(
    r"^\s*(?:[«\"']\s*)?"
    # Subject head: a capitalised noun phrase (1–4 tokens). Cyrillic
    # capitals plus a few common Latin product names appearing in TZ.
    r"(?:[А-ЯЁA-Z][\w-]+(?:\s+[а-яёa-zA-Z][\w-]+){0,3})\s+"
    r"долж(?:ен|на|но|ны)\b",
    re.UNICODE,
)


def _is_self_contained_normative(text: str) -> bool:
    """True when the fragment reads as a complete normative statement
    on its own («Система должна …», «Программа должна …»). Such
    fragments don't need windowing — the paragraph unit already covers
    them — and bundling several of them into one window produces an
    evidence card that mixes multiple distinct requirements."""
    return bool(_SELF_CONTAINED_NORMATIVE_RE.match(text or ""))


_LIST_INTRO_NOISE_RE = re.compile(
    # Pattern 1: trailing «:.» (post-period normalisation artefact).
    r":\s*\.\s*$"
    # Pattern 2: plain «:» tail + a list-cue noun anywhere in the text.
    r"|\b(?:следующ\w*\s+(?:набор|функц|операц|возможност|пункт|критери|"
    r"услови|сценари|шаг|этап|требовани|параметр)"
    r"|перечисленн\w+"
    r"|нижеследующ\w+"
    r"|(?:приведённ|приведенн|указанн)\w+\s+(?:ниже|далее))"
    r"\b\s*:?\s*\.?\s*$",
    re.IGNORECASE | re.UNICODE,
)


def _is_list_intro_stub(text: str) -> bool:
    """True when the fragment is a list-intro stub like «…следующий
    набор функций:.» — contributes no coverage information on its own."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    # Tail-shape «:.» is unambiguous extraction artefact.
    if re.search(r":\s*\.\s*$", stripped):
        return True
    # Plain trailing colon + list-cue noun anywhere in the sentence.
    if (
        stripped.rstrip().endswith(":")
        and re.search(
            r"\b(?:следующ\w*\s+(?:набор|функц|операц|возможност|пункт|критери|"
            r"услови|сценари|шаг|этап|требовани|параметр)"
            r"|перечисленн\w+"
            r"|перечен\w*"
            r"|нижеследующ\w+"
            r"|(?:приведённ|приведенн|указанн)\w+(?:\s+(?:ниже|далее))?)\b",
            stripped, re.IGNORECASE | re.UNICODE,
        )
    ):
        return True
    return False


def _is_section_heading_only(text: str, section_title: str) -> bool:
    """True when the fragment text is essentially the section heading
    (re-emitted as a body paragraph by some DOCX exports). Prevents the
    heading from polluting paragraph units AND from leading SECTION_WINDOW
    units with redundant title text."""
    if not text or not section_title:
        return False
    # Strip trailing punctuation, lowercase for comparison.
    norm_text = re.sub(r"[.\s]+$", "", text.strip()).lower()
    norm_title = re.sub(r"[.\s]+$", "", section_title.strip()).lower()
    if not norm_text or not norm_title:
        return False
    return norm_text == norm_title


def _truncate_at_sentence_boundary(text: str, limit: int) -> str:
    """Truncate `text` to ≤ `limit` chars, preferring the last sentence
    boundary («.», «!», «?», «»») within the budget. Falls back to last
    space when no boundary exists. Avoids the «Система должна
    обеспечивать загрузку» mid-sentence cut produced by the previous
    rsplit-on-space implementation. Always picks the latest boundary
    in the head — even early boundaries beat mid-word cuts."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    # Prefer last sentence boundary within head; always wins over a
    # space-fallback because it produces a clean, complete sentence.
    boundary = max(head.rfind(c) for c in ".!?»")
    if boundary > 0:
        return head[: boundary + 1].strip()
    # Fall back to last space; better than mid-word cut.
    space = head.rfind(" ")
    if space >= 0:
        return head[:space].strip()
    return head.strip()


_FRAGMENT_KIND_MAP = {
    "list_item": CoverageUnitType.LIST_ITEM,
    "list": CoverageUnitType.LIST_ITEM,
    "test_step": CoverageUnitType.TEST_STEP,
    "step": CoverageUnitType.TEST_STEP,
    "expected_result": CoverageUnitType.EXPECTED_RESULT,
    "expected": CoverageUnitType.EXPECTED_RESULT,
    "precondition": CoverageUnitType.PRECONDITION,
    "table_row": CoverageUnitType.TABLE_ROW_TEXT,
    "table_row_text": CoverageUnitType.TABLE_ROW_TEXT,
    "paragraph": CoverageUnitType.PARAGRAPH,
}


def _map_unit_type(kind: str) -> CoverageUnitType:
    if not kind:
        return CoverageUnitType.PARAGRAPH
    return _FRAGMENT_KIND_MAP.get(kind.lower(), CoverageUnitType.PARAGRAPH)


def _collect_raw_fragments(artifact: dict) -> Tuple[List[dict], str]:
    """Return (raw_frag_dicts, source_label) from the first non-empty source."""
    frags = artifact.get("fragments") or []
    if frags:
        return list(frags), "fragments"

    # Fallback: sentences[] (some prepare-service versions emit this instead)
    sentences = artifact.get("sentences") or []
    if sentences:
        logger.info(
            "fragments[] empty for %s; falling back to sentences[] (%d items)",
            artifact.get("document_id", "?"), len(sentences),
        )
        return list(sentences), "sentences"

    return [], "none"


class CoverageUnitBuilder:
    """Builds CoverageUnit list from a single PMI/PZ PreparedArtifact."""

    def build(self, artifact: dict) -> List[CoverageUnit]:
        doc_id = artifact.get("document_id", "unknown")
        doc_role = artifact.get("doc_role", "unknown")

        raw_frags, source = _collect_raw_fragments(artifact)

        # Section title lookup: populated once here and reused both for
        # annotating paragraph units (→ competitive-analysis penalty in
        # retrieve_candidates) and for building section-window units.
        _sections = artifact.get("sections") or []
        _title_by_id: dict[str, str] = {
            s.get("section_id"): (s.get("title") or "")
            for s in _sections
            if isinstance(s, dict) and s.get("section_id")
        }

        # ── step-by-step diagnostics ──────────────────────────────────────
        total_raw = len(raw_frags)
        nonempty_text = sum(1 for f in raw_frags if (f.get("text") or "").strip())
        passed_len = sum(
            1 for f in raw_frags
            if len((f.get("text") or "").strip().split()) >= _MIN_WORDS
        )
        logger.info(
            "[CoverageUnitBuilder] doc=%s role=%s source=%s "
            "raw=%d nonempty_text=%d passed_len_filter(>=%d_words)=%d",
            doc_id, doc_role, source,
            total_raw, nonempty_text, _MIN_WORDS, passed_len,
        )

        if not raw_frags:
            logger.warning(
                "No fragments or sentences in artifact %s (%s); "
                "coverage_units=0. Verify prepare-service output for this document.",
                doc_id, doc_role,
            )
            return []

        units: List[CoverageUnit] = []
        # Prepare-service emits fragments that share a long common prefix with
        # different trailing sentences (one paragraph materialized as items
        # item5/item7/item8/... appending a growing tail). Full-text dedup
        # misses these. Prefix dedup must still keep a later fragment when it
        # adds important requirement anchors such as Angular/GitHub/REST/JSON.
        seen_norm: dict[str, Tuple[int, frozenset[str]]] = {}
        dropped_dupes = 0
        dropped_noise = 0
        for frag in raw_frags:
            text = (frag.get("text") or "").strip()
            if not text:
                continue
            if len(text.split()) < _MIN_WORDS:
                logger.debug(
                    "Skipping fragment fragment_id=%s (< %d words): %r",
                    frag.get("fragment_id", "?"), _MIN_WORDS, text[:60],
                )
                continue
            # Polyakov-regression: drop list-intro stubs and section-
            # heading-only paragraphs before they enter coverage units.
            # Both shapes carry no coverage information of their own and
            # poison SECTION_WINDOW units that concatenate adjacent
            # fragments (the «:.» tail makes long windows read like
            # truncated sentences to the LLM judge).
            sec_id = frag.get("section_id")
            sec_title = _title_by_id.get(sec_id, "") if sec_id else ""
            if _is_list_intro_stub(text):
                dropped_noise += 1
                logger.debug(
                    "Skipping list-intro-stub fragment_id=%s: %r",
                    frag.get("fragment_id", "?"), text[:60],
                )
                continue
            if _is_section_heading_only(text, sec_title):
                dropped_noise += 1
                logger.debug(
                    "Skipping section-heading-only fragment_id=%s "
                    "(matches title %r): %r",
                    frag.get("fragment_id", "?"), sec_title[:40], text[:60],
                )
                continue
            norm = _normalize_text(text)
            key = norm[:_PREFIX_DEDUP_CHARS]
            if not _register_dedup_key(seen_norm, key, norm, text):
                dropped_dupes += 1
                continue
            units.append(
                CoverageUnit(
                    target_document_id=doc_id,
                    target_doc_role=doc_role,
                    section_id=frag.get("section_id"),
                    fragment_id=frag.get("fragment_id"),
                    unit_type=_map_unit_type(frag.get("kind", "")),
                    text=text,
                    normalized_text=norm,
                    entities=_extract_entities(text),
                    constraints=_extract_constraints(text),
                    metadata=_enrich_metadata(
                        frag.get("metadata"), frag.get("section_id"), _title_by_id
                    ),
                )
            )
        units.extend(self._build_section_window_units(artifact, raw_frags, seen_norm))
        if dropped_dupes:
            logger.info("doc=%s (%s): dropped %d duplicate-text fragments", doc_id, doc_role, dropped_dupes)
        if dropped_noise:
            logger.info(
                "doc=%s (%s): dropped %d list-intro/heading-only noise fragments",
                doc_id, doc_role, dropped_noise,
            )

        if not units and nonempty_text > 0:
            logger.warning(
                "doc=%s (%s): %d fragments received but all filtered "
                "(all < %d words). Shortest texts: %s",
                doc_id, doc_role, nonempty_text, _MIN_WORDS,
                [f.get("text", "")[:30] for f in raw_frags[:5]],
            )

        logger.info("Built %d coverage units from %s (%s)", len(units), doc_id, doc_role)
        return units

    def _build_section_window_units(
        self,
        artifact: dict,
        raw_frags: List[dict],
        seen_norm: dict[str, Tuple[int, frozenset[str]]],
    ) -> List[CoverageUnit]:
        """Add broader evidence units that preserve local section context.

        Paragraph-only evidence is brittle for PZ/PMI: a requirement can be
        covered by several adjacent paragraphs ("REST API" in one sentence,
        "JSON" and upload limits in the next). These window units are
        additive; paragraph units remain available for precise citations.
        """
        doc_id = artifact.get("document_id", "unknown")
        doc_role = artifact.get("doc_role", "unknown")
        sections = artifact.get("sections") or []
        title_by_id = {
            s.get("section_id"): (s.get("title") or "")
            for s in sections
            if isinstance(s, dict) and s.get("section_id")
        }
        has_section_context = bool(title_by_id) or any(
            (f.get("section_id") or "") or _section_title_from_fragments([f])
            for f in raw_frags
        )
        if not has_section_context:
            return []

        frags_by_section: dict[str, List[dict]] = {}
        for frag in raw_frags:
            text = (frag.get("text") or "").strip()
            if not text or len(text.split()) < _MIN_WORDS:
                continue
            sid = frag.get("section_id") or "__no_section__"
            section_title = title_by_id.get(sid, "")
            # Polyakov-regression: keep noise fragments out of windows
            # too. A list-intro stub or a section-heading paragraph
            # bundled into a SECTION_WINDOW poisons the unit text.
            if _is_list_intro_stub(text):
                continue
            if _is_section_heading_only(text, section_title):
                continue
            frags_by_section.setdefault(sid, []).append(frag)

        out: List[CoverageUnit] = []
        for sid, frags in frags_by_section.items():
            title = title_by_id.get(sid) or _section_title_from_fragments(frags)
            texts = [(f.get("text") or "").strip() for f in frags]
            # Polyakov-regression (2026-05-10): greedy windowing instead
            # of fixed-size grouping. Each window grows up to
            # _SECTION_WINDOW_SIZE fragments BUT closes early when a
            # second self-contained normative sentence enters — that
            # second sentence belongs in its own window so the evidence
            # card carries one requirement, not four.
            window_groups: List[Tuple[int, List[str]]] = []
            cur_start = 0
            cur: List[str] = []
            cur_self_contained = 0
            for i, t in enumerate(texts):
                is_sc = _is_self_contained_normative(t)
                # Close the window before adding a 2nd self-contained
                # sentence, OR when the size cap is reached.
                if cur and (
                    (is_sc and cur_self_contained >= 1)
                    or len(cur) >= _SECTION_WINDOW_SIZE
                ):
                    window_groups.append((cur_start, cur))
                    cur_start = i
                    cur = []
                    cur_self_contained = 0
                cur.append(t)
                if is_sc:
                    cur_self_contained += 1
            if cur:
                window_groups.append((cur_start, cur))

            for start, window_texts in window_groups:
                text = " ".join(window_texts).strip()
                if not text:
                    continue
                if len(text) > _MAX_WINDOW_CHARS:
                    # Polyakov-regression: truncate at the last sentence
                    # boundary inside the budget instead of the previous
                    # space-only cut. Mid-sentence truncation («Система
                    # должна обеспечивать загрузку») reads like a broken
                    # PMI restatement to the LLM judge — sentence-aligned
                    # truncation gives clean evidence text.
                    text = _truncate_at_sentence_boundary(text, _MAX_WINDOW_CHARS)
                # Polyakov-regression: do NOT prepend the section title
                # to the unit text. The title is already exposed via
                # metadata.section_title (and used by retrieve_candidates'
                # _PMI_TEST_SECTION_RE for section-prior boost). Prepending
                # it duplicates the topic on the wire, dilutes Jaccard
                # against the requirement, and produces «Требования к
                # функциональным характеристикам. Клиентская часть
                # приложения должна …» — exactly the noisy unit shape
                # observed on the Polyakov demo run.
                display_text = text
                norm = _normalize_text(display_text)
                key = norm[:_PREFIX_DEDUP_CHARS]
                if not _register_dedup_key(seen_norm, key, norm, display_text):
                    continue
                out.append(
                    CoverageUnit(
                        target_document_id=doc_id,
                        target_doc_role=doc_role,
                        section_id=None if sid == "__no_section__" else sid,
                        fragment_id=f"{sid}::window::{start}",
                        unit_type=CoverageUnitType.SECTION_WINDOW,
                        text=display_text,
                        normalized_text=norm,
                        entities=_extract_entities(display_text),
                        constraints=_extract_constraints(display_text),
                        metadata={
                            "section_title": title,
                            "window_start": start,
                            "window_size": len(window_texts),
                            "coverage_window": True,
                        },
                    )
                )
        if out:
            logger.info(
                "doc=%s (%s): added %d section-window coverage units",
                doc_id, doc_role, len(out),
            )
        return out


def _section_title_from_fragments(frags: List[dict]) -> str:
    for frag in frags:
        meta = frag.get("metadata") or {}
        title = meta.get("sectionTitle") or meta.get("section_title") or ""
        if title:
            return str(title)
    return ""


def _coverage_signals(text: str) -> frozenset[str]:
    return frozenset(
        re.sub(r"\s+", " ", match.group(0).lower()).strip()
        for match in _COVERAGE_SIGNAL_RE.finditer(text or "")
    )


def _register_dedup_key(
    seen_norm: "dict[str, Tuple[int, frozenset[str]]]",
    key: str,
    norm: str,
    text: str,
) -> bool:
    """Return True when a near-duplicate should still become evidence.

    Prefix-based dedup removes noisy cumulative fragments, but it must not
    hide the useful tail of a cumulative paragraph. A later fragment is kept
    when it is meaningfully longer or introduces new requirement-bearing
    anchors.
    """
    signals = _coverage_signals(text)
    prior = seen_norm.get(key)
    if prior is None:
        seen_norm[key] = (len(norm), signals)
        return True

    prev_len, prev_signals = prior
    has_new_signals = bool(signals - prev_signals)
    grew_meaningfully = len(norm) > prev_len + _DEDUP_GROWTH_CHARS
    seen_norm[key] = (max(prev_len, len(norm)), prev_signals | signals)
    return has_new_signals or grew_meaningfully


def _enrich_metadata(
    raw_meta: "dict | None",
    section_id: "str | None",
    title_by_id: "dict[str, str]",
) -> dict:
    """Return a copy of raw_meta with ``section_title`` populated from
    *title_by_id* when the fragment belongs to a known section.

    Only sets ``section_title`` if it is not already present so that any
    value injected by prepare-service is preserved.  The retrieval-stage
    competitive-analysis penalty (``_NON_IMPL_SECTION_RE`` in
    ``retrieve_candidates.py``) reads this field to demote evidence from
    sections such as "Существующие аналоги" or "Стилизация интерфейса"
    that are never implementation evidence for TZ requirements.
    """
    meta = dict(raw_meta or {})
    if not meta.get("section_title") and not meta.get("sectionTitle"):
        if section_id and section_id in title_by_id:
            title = title_by_id[section_id]
            if title:
                meta["section_title"] = title
    return meta
