"""
Stage 2: build CoverageUnit list from a PMI/PZ artifact.

Granularity: fragment-level (paragraph / list_item / test_step / etc.)
Reuses the same extraction helpers as build_requirements.

Fragment sources (tried in order):
  1. artifact["fragments"]   — primary; flat list from prepare-service
  2. artifact["sentences"]   — fallback; finer-grained source some versions emit
"""
from __future__ import annotations

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
        # misses these; keying by the first 400 chars of normalized text
        # collapses them while still separating genuinely distinct fragments.
        _PREFIX_DEDUP_CHARS = 400
        seen_norm: dict[str, int] = {}
        dropped_dupes = 0
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
            norm = _normalize_text(text)
            key = norm[:_PREFIX_DEDUP_CHARS]
            if key in seen_norm:
                dropped_dupes += 1
                continue
            seen_norm[key] = 1
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
                    metadata=frag.get("metadata") or {},
                )
            )
        if dropped_dupes:
            logger.info("doc=%s (%s): dropped %d duplicate-text fragments", doc_id, doc_role, dropped_dupes)

        if not units and nonempty_text > 0:
            logger.warning(
                "doc=%s (%s): %d fragments received but all filtered "
                "(all < %d words). Shortest texts: %s",
                doc_id, doc_role, nonempty_text, _MIN_WORDS,
                [f.get("text", "")[:30] for f in raw_frags[:5]],
            )

        logger.info("Built %d coverage units from %s (%s)", len(units), doc_id, doc_role)
        return units
