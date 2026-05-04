"""
PR-K post-fix (c) regression tests for section-aware boost in
RequirementBuilder.

Section-boost is a merge-pass that runs AFTER the primary extraction
(candidates / fragments) in `auto` mode. It picks up requirement-
shaped sentences from sections classified as definitely
requirement-bearing (`_classify_section() == True`) that the primary
path missed.

Tests pin down:
  * boost adds requirements the primary path missed
  * boost dedups against primary by normalised text
  * boost only fires on definitely-requirement-bearing sections
    (numbering 4.x or title with explicit keyword) — never on
    Введение / Цель / Аналоги / random sections
  * boost requires modality OR trigger word at sentence level —
    pure narrative inside a requirement section is not promoted
  * boost can be disabled via CQUALITY_SECTION_BOOST=false env var
  * non-`auto` modes are NOT augmented (sections / candidates /
    fragments / model are explicit)
"""
from __future__ import annotations

import pytest

from app.application.use_cases.build_requirements import RequirementBuilder
from app.core.config import CoverageConfig


def _artifact(*, sections, fragments, requirement_candidates=None):
    return {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "sections": sections,
        "fragments": fragments,
        "requirement_candidates": requirement_candidates or [],
    }


def _section(sid, title, level=2):
    return {"section_id": sid, "title": title, "level": level}


def _frag(sid, fragment_id, text):
    return {
        "section_id": sid, "fragment_id": fragment_id, "text": text,
        "kind": "paragraph",
    }


# ── Boost adds missing requirements ───────────────────────────────────


class TestBoostFindsMissingReqs:
    def setup_method(self):
        self.cfg = CoverageConfig.from_options({"requirement_extraction": "auto"})
        self.builder = RequirementBuilder(self.cfg)

    def test_section_boost_adds_req_from_explicit_requirements_section(self):
        """Primary extraction (via candidates) returns 1 unit. The
        section also contains another sentence that's a clear
        requirement (modality + trigger), but BERT did not flag it.
        Section-boost must pick it up as a second unit."""
        sections = [
            _section("4.1", "Требования к функциональным характеристикам"),
        ]
        fragments = [
            _frag("4.1", "f1",
                  "Система должна обеспечивать аутентификацию пользователей."),
            _frag("4.1", "f2",
                  "Программа должна логировать все действия пользователя."),
        ]
        candidates = [
            {"text": "Система должна обеспечивать аутентификацию пользователей.",
             "section_id": "4.1", "fragment_id": "f1"},
        ]
        out = self.builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=candidates,
        ))
        texts = [u.text for u in out]
        assert any("аутентификацию" in t for t in texts)
        assert any("логировать" in t for t in texts), (
            f"section-boost missed the second requirement; got: {texts}"
        )

    def test_boost_dedups_by_normalised_text(self):
        """If a candidate and a section-fragment have THE SAME
        normalised text (different whitespace / quotes), boost must
        NOT add a duplicate."""
        sections = [_section("4.1", "Требования к функциональным характеристикам")]
        same_text = "Система должна обеспечивать  быстрый  доступ к данным."
        fragments = [_frag("4.1", "f1", same_text.replace("  ", " "))]
        candidates = [
            {"text": same_text, "section_id": "4.1", "fragment_id": "f1"},
        ]
        out = self.builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=candidates,
        ))
        # Single requirement total — not two near-duplicates.
        assert len(out) == 1, (
            f"normalised-dup not dedup'd; got {len(out)} units: {[u.text for u in out]}"
        )

    def test_boost_handles_multiple_sentences_in_one_fragment(self):
        """A single fragment can contain multiple sentences, each
        of which is a separate requirement (numbered list etc).
        The section-driven sentence splitter handles this."""
        sections = [_section("4.1", "Требования к надежности")]
        fragments = [
            _frag("4.1", "f1",
                  "Система должна работать без перезагрузки 24/7. "
                  "Программа не должна аварийно завершаться при ошибке. "
                  "Время восстановления должно быть менее одного часа."),
        ]
        out = self.builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=[],
        ))
        # All three sentences contain modality, all should be requirements.
        # Primary path (fragments fallback) yields 1 unit (the whole fragment text);
        # boost should add the other 2 sentences after dedup.
        assert len(out) >= 2, (
            f"expected >=2 reqs from a 3-sentence fragment, got {len(out)}"
        )


# ── Boost is conservative ─────────────────────────────────────────────


class TestBoostIsConservative:
    def setup_method(self):
        self.cfg = CoverageConfig.from_options({"requirement_extraction": "auto"})
        self.builder = RequirementBuilder(self.cfg)

    def test_boost_does_not_fire_on_intro_section(self):
        """Section 'Введение' is non-requirement; boost must NOT
        consider its fragments. Test by giving primary a candidate
        from 4.1, plus a sentence in 'Введение' with modality. Boost
        should NOT add the intro sentence as a unit. Note: primary
        fragments-fallback can independently grab modality-bearing
        sentences from any section — that's a separate issue. We
        only verify the BOOST path here by checking req_ids with
        the `::boost::` marker."""
        sections = [
            _section("1", "Введение"),
            _section("4.1", "Требования к функциональным характеристикам"),
        ]
        fragments = [
            _frag("1", "f1",
                  "Документ должен описывать архитектуру системы."),
            _frag("4.1", "f2",
                  "Система должна обеспечивать поиск по индексу."),
        ]
        candidates = [
            # Primary picks this one; boost has a chance to add others.
            {"text": "Система должна обеспечивать поиск по индексу.",
             "section_id": "4.1", "fragment_id": "f2"},
        ]
        out = self.builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=candidates,
        ))
        # Look at boost-marked units only.
        boost_units = [u for u in out if "::boost::" in u.req_id]
        for u in boost_units:
            assert u.source_section_id != "1", (
                f"boost extracted from Введение; got: {u.text}"
            )

    def test_boost_skips_ambiguous_sections(self):
        """Sections without explicit requirement-keyword AND without
        section number 4.x are 'None' (ambiguous). Boost stays silent."""
        sections = [_section("X", "Прочие соображения по проекту")]
        fragments = [
            _frag("X", "f1",
                  "Команда должна провести регрессионное тестирование."),
        ]
        candidates = [
            {"text": "Какое-то существующее требование.",
             "section_id": "5.1", "fragment_id": "x1"},
        ]
        out = self.builder.build(_artifact(
            sections=sections + [_section("5.1", "Cписок требований")],
            fragments=fragments + [_frag("5.1", "x1", "Какое-то существующее требование.")],
            requirement_candidates=candidates,
        ))
        for u in out:
            assert u.source_section_id != "X", (
                f"boost extracted from ambiguous section; got: {u.text}"
            )

    def test_boost_requires_modality_or_trigger(self):
        """Even within a requirement-bearing section, sentences without
        modality / trigger word (pure narrative) must NOT be promoted."""
        sections = [_section("4.1", "Требования к функциональным характеристикам")]
        fragments = [
            _frag("4.1", "f1",
                  "Архитектура построена на основе микросервисов. "
                  "Эта секция содержит описание базы данных."),
        ]
        out = self.builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=[],
        ))
        # Both sentences are narrative — no must / трек word. Result
        # must be empty (or very small) — definitely no promotion.
        assert all(
            "должн" in u.text.lower() or "обеспеч" in u.text.lower()
            or "поддерж" in u.text.lower()
            for u in out
        ), f"boost promoted narrative; got: {[u.text for u in out]}"


# ── Disable switch ────────────────────────────────────────────────────


class TestBoostDisableSwitch:
    def test_env_var_disables_boost(self, monkeypatch):
        monkeypatch.setenv("CQUALITY_SECTION_BOOST", "false")
        cfg = CoverageConfig.from_options({"requirement_extraction": "auto"})
        builder = RequirementBuilder(cfg)
        sections = [_section("4.1", "Требования к функциональным характеристикам")]
        fragments = [
            _frag("4.1", "f1",
                  "Система должна обеспечивать аутентификацию пользователей."),
            _frag("4.1", "f2",
                  "Программа должна логировать все действия пользователя."),
        ]
        candidates = [
            {"text": "Система должна обеспечивать аутентификацию пользователей.",
             "section_id": "4.1", "fragment_id": "f1"},
        ]
        out = builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=candidates,
        ))
        # Boost off — only the candidate is taken, second sentence dropped.
        texts = [u.text for u in out]
        assert any("аутентификацию" in t for t in texts)
        assert not any("логировать" in t for t in texts), (
            f"boost was disabled but second req leaked through; got: {texts}"
        )


# ── Non-auto modes not augmented ──────────────────────────────────────


class TestNonAutoModesNotAugmented:
    """sections / candidates / fragments / model modes are EXPLICIT —
    section-boost only fires in auto."""

    def test_candidates_mode_no_boost(self):
        cfg = CoverageConfig.from_options({"requirement_extraction": "candidates"})
        builder = RequirementBuilder(cfg)
        sections = [_section("4.1", "Требования к функциональным характеристикам")]
        fragments = [
            _frag("4.1", "f1", "Система должна обеспечивать аутентификацию."),
            _frag("4.1", "f2", "Система должна логировать действия."),
        ]
        candidates = [
            {"text": "Система должна обеспечивать аутентификацию.",
             "section_id": "4.1", "fragment_id": "f1"},
        ]
        out = builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=candidates,
        ))
        # candidates mode: only the candidate (no boost from section).
        assert len(out) == 1
        assert "аутентификацию" in out[0].text

    def test_sections_mode_no_boost(self):
        cfg = CoverageConfig.from_options({"requirement_extraction": "sections"})
        builder = RequirementBuilder(cfg)
        # sections mode runs `_from_sections` which already includes
        # section-aware logic; boost is not separately applied.
        # Use ≥5 words per sentence to clear the section-driven length
        # filter.
        sections = [_section("4.1", "Требования к функциональным характеристикам")]
        fragments = [
            _frag("4.1", "f1",
                  "Система должна обеспечивать многофакторную аутентификацию пользователей "
                  "при работе с критическими данными."),
        ]
        out = builder.build(_artifact(
            sections=sections, fragments=fragments, requirement_candidates=[],
        ))
        assert len(out) >= 1
        # And no `::boost::` marker — boost is auto-mode-only.
        for u in out:
            assert "::boost::" not in u.req_id
