"""
FIX-C3: sectionCategory-aware candidate gate in _section_allows_candidate.

Acceptance criteria:
  * metadata.sectionCategory="requirements" → always allow (gold signal).
  * metadata.sectionCategory="other" → require MUST/MUST_NOT/SHOULD modality
    (prevents false positives from unstructured TZs like Череухо where all
    sections are tagged "other" and heuristic extraction inflates the list).
  * metadata.sectionCategory absent + section 4.x → always allow (backward compat).
  * metadata.sectionCategory absent + unknown numbering → require modality
    (tightened from previous permissive behaviour).
  * Full _from_candidates integration: candidates with sectionCategory="other"
    and no modality are filtered out; candidates in sectionCategory="requirements"
    pass regardless of modality.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.build_requirements import (
    RequirementBuilder,
    _section_allows_candidate,
)
from app.core.config import CoverageConfig
from app.domain.c_quality_enums import Modality


# ── Unit tests for _section_allows_candidate ────────────────────────────────


class TestSectionCategoryGold:
    """sectionCategory in metadata takes precedence over GOST numbering."""

    @pytest.mark.parametrize("modality", [
        Modality.MUST, Modality.MUST_NOT, Modality.SHOULD,
        Modality.MAY, Modality.UNKNOWN,
    ])
    def test_requirements_category_always_allows(self, modality):
        """sectionCategory='requirements' → allow regardless of modality."""
        assert _section_allows_candidate(
            None, modality, {"sectionCategory": "requirements"}
        ) is True

    @pytest.mark.parametrize("modality", [
        Modality.MUST, Modality.MUST_NOT, Modality.SHOULD,
    ])
    def test_other_category_allows_explicit_modality(self, modality):
        """sectionCategory='other' + explicit modality → allow."""
        assert _section_allows_candidate(
            None, modality, {"sectionCategory": "other"}
        ) is True

    @pytest.mark.parametrize("modality", [Modality.MAY, Modality.UNKNOWN])
    def test_other_category_blocks_weak_modality(self, modality):
        """sectionCategory='other' + MAY/UNKNOWN → block (prevent false positives)."""
        assert _section_allows_candidate(
            None, modality, {"sectionCategory": "other"}
        ) is False

    @pytest.mark.parametrize("section_category", ["metadata", "environment", "test_steps"])
    def test_non_req_categories_require_modality(self, section_category):
        """Any non-requirements category requires explicit modality."""
        assert _section_allows_candidate(
            None, Modality.MUST, {
                "sectionCategory": section_category
            }
        ) is True
        assert _section_allows_candidate(
            None, Modality.UNKNOWN, {
                "sectionCategory": section_category
            }
        ) is False

    def test_metadata_overrides_gost_numbering(self):
        """sectionCategory takes priority: section 4.x with 'other' still
        requires modality. Structural position alone is not sufficient when
        prepare-service explicitly tagged the section differently."""
        # Section 4.x would normally be allowed without metadata...
        assert _section_allows_candidate("4.1", Modality.UNKNOWN, None) is True
        # ...but with an explicit sectionCategory="other", require modality.
        assert _section_allows_candidate(
            "4.1", Modality.UNKNOWN, {"sectionCategory": "other"}
        ) is False


class TestSectionCategoryFallback:
    """Backward compat: when sectionCategory is absent, fall back to GOST numbering."""

    def test_no_metadata_gost_4x_allows(self):
        """Without metadata, GOST section 4.x is always allowed."""
        assert _section_allows_candidate("4.1", Modality.UNKNOWN, None) is True
        assert _section_allows_candidate("4.2.3", Modality.MAY, {}) is True

    def test_no_metadata_gost_1_2_3_require_modality(self):
        """Without metadata, introductory sections 1-3 require modality."""
        assert _section_allows_candidate("1", Modality.UNKNOWN, {}) is False
        assert _section_allows_candidate("2.1", Modality.MAY, {}) is False
        assert _section_allows_candidate("3", Modality.MUST, {}) is True

    def test_no_metadata_unknown_section_requires_modality(self):
        """Without metadata, unknown section numbering (None / 5+) now
        requires explicit modality — tightened from the previous permissive
        'return True' behaviour. Prevents noise from non-GOST TZs."""
        assert _section_allows_candidate(None, Modality.UNKNOWN, {}) is False
        assert _section_allows_candidate(None, Modality.MAY, {}) is False
        assert _section_allows_candidate(None, Modality.MUST, {}) is True
        assert _section_allows_candidate("5.1", Modality.MUST_NOT, {}) is True
        assert _section_allows_candidate("5.1", Modality.UNKNOWN, {}) is False

    def test_no_metadata_empty_section_id_requires_modality(self):
        """Empty string section_id (preamble / unknown) without metadata →
        unknown section → require modality."""
        assert _section_allows_candidate("", Modality.UNKNOWN, {}) is False
        assert _section_allows_candidate("", Modality.MUST, {}) is True


# ── Integration: _from_candidates with sectionCategory ──────────────────────


def _artifact(candidates):
    return {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "sections": [],
        "fragments": [],
        "requirement_candidates": candidates,
    }


class TestFromCandidatesWithSectionCategory:
    """End-to-end: RequirementBuilder._from_candidates respects sectionCategory."""

    def setup_method(self):
        cfg = CoverageConfig.from_options({"requirement_extraction": "candidates"})
        self.builder = RequirementBuilder(cfg)

    def test_requirements_section_candidate_passes_without_modality(self):
        """A candidate in sectionCategory='requirements' with no explicit
        modality keyword should still be included — the requirements chapter
        is the gold signal."""
        cand = {
            "text": "Приложение обеспечивает экспорт данных в формате CSV.",
            "section_id": "3.1",
            "fragment_id": "f-export",
            "metadata": {"sectionCategory": "requirements"},
        }
        units = self.builder.build(_artifact([cand]))
        assert len(units) == 1, (
            f"candidate from requirements section was filtered; got {len(units)} units"
        )
        assert "экспорт" in units[0].text

    def test_other_section_candidate_without_modality_is_dropped(self):
        """A candidate in sectionCategory='other' with no MUST/MUST_NOT/SHOULD
        must be filtered out — unstructured TZ noise."""
        cand = {
            "text": "Данная система позволяет обрабатывать входящие запросы.",
            "section_id": "preamble",
            "fragment_id": "f-noise",
            "metadata": {"sectionCategory": "other"},
        }
        units = self.builder.build(_artifact([cand]))
        assert units == [], (
            f"noise candidate from 'other' section slipped through; got: {[u.text for u in units]}"
        )

    def test_other_section_candidate_with_modality_passes(self):
        """A candidate in sectionCategory='other' WITH explicit modality
        (MUST) is still a valid requirement — accept it."""
        cand = {
            "text": "Система должна обрабатывать не менее 1000 запросов в секунду.",
            "section_id": "preamble",
            "fragment_id": "f-perf",
            "metadata": {"sectionCategory": "other"},
        }
        units = self.builder.build(_artifact([cand]))
        assert len(units) == 1, (
            f"MUST-modality candidate from 'other' section was incorrectly dropped; "
            f"got {len(units)} units"
        )

    def test_mixed_candidates_filtered_correctly(self):
        """Mix of requirements/other categories: only the right ones pass."""
        candidates = [
            # Should pass: explicit requirements chapter
            {
                "text": "Система реализует двухфакторную аутентификацию.",
                "section_id": "req_sec",
                "fragment_id": "f1",
                "metadata": {"sectionCategory": "requirements"},
            },
            # Should pass: other section but has MUST modality
            {
                "text": "Программа должна логировать все ошибки в системный журнал.",
                "section_id": "misc",
                "fragment_id": "f2",
                "metadata": {"sectionCategory": "other"},
            },
            # Should be DROPPED: other section + no modality
            {
                "text": "Данная функция предоставляет интерфейс для управления пользователями.",
                "section_id": "desc",
                "fragment_id": "f3",
                "metadata": {"sectionCategory": "other"},
            },
        ]
        units = self.builder.build(_artifact(candidates))
        texts = [u.text for u in units]
        assert any("аутентификацию" in t for t in texts), "requirements-chapter cand missing"
        assert any("логировать" in t for t in texts), "MUST-modality cand from 'other' missing"
        assert not any("предоставляет интерфейс" in t for t in texts), (
            f"noise candidate slipped through: {texts}"
        )

    def test_unstructured_tz_all_other_only_modality_passes(self):
        """Simulate Череухо-class unstructured TZ: all sections tagged 'other'.
        Only candidates with explicit MUST/MUST_NOT/SHOULD modality should
        be included — the rest are noise."""
        candidates = [
            {
                "text": "Система предназначена для обработки документов.",
                "section_id": "chunk_001",
                "fragment_id": "n1",
                "metadata": {"sectionCategory": "other"},
            },
            {
                "text": "Система должна поддерживать одновременную работу не менее 50 пользователей.",
                "section_id": "chunk_015",
                "fragment_id": "r1",
                "metadata": {"sectionCategory": "other"},
            },
            {
                "text": "Программа не должна аварийно завершать работу при ошибке ввода.",
                "section_id": "chunk_022",
                "fragment_id": "r2",
                "metadata": {"sectionCategory": "other"},
            },
            {
                "text": "Разработка ведётся в соответствии с ГОСТ Р ИСО/МЭК 25010.",
                "section_id": "chunk_003",
                "fragment_id": "n2",
                "metadata": {"sectionCategory": "other"},
            },
        ]
        units = self.builder.build(_artifact(candidates))
        texts = [u.text for u in units]
        # Only the two MUST/MUST_NOT sentences pass
        assert any("50 пользователей" in t for t in texts), "MUST candidate missing"
        assert any("аварийно завершать" in t for t in texts), "MUST_NOT candidate missing"
        # Pure narrative and GOST references drop
        assert not any("предназначена для" in t for t in texts), (
            f"descriptive noise slipped through: {texts}"
        )
        assert not any("ГОСТ Р ИСО" in t for t in texts), (
            f"GOST reference slipped through: {texts}"
        )
        assert len(units) == 2, f"expected 2 units, got {len(units)}: {texts}"
