"""
FIX-C4: documentation_requirement candidates must NOT be included in the
program-requirement list.

Background
----------
prepare-service tags candidates that describe WHAT DOCUMENTS to produce
(ГОСТ section "Требования к программной документации") as
  candidateType = "documentation_requirement"
docback maps ALL three types (requirement_like / documentation_requirement /
environment_requirement) → ctRequirement so they all arrive at C-quality in
requirement_candidates[].  The original prepare-service type is preserved in
  metadata["prepareType"]

C-quality must filter them out:
  * _from_candidates path: skip candidatees with prepareType="documentation_requirement"
  * _from_sections / _section_boost path: sections titled with
    "программной документаци" / "к документации" must be classified as
    non-requirement by _is_requirement_section_by_title.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.applicability import applicability_for
from app.application.use_cases.build_requirements import (
    RequirementBuilder,
    _is_requirement_section_by_title,
)
from app.core.config import CoverageConfig
from app.domain.c_quality_enums import Applicability, RequirementType


# ── Section-title classifier ───────────────────────────────────────────────


class TestDocumentationTitleClassifier:
    """_is_requirement_section_by_title must return False for documentation
    sections even though their title contains 'требовани'."""

    @pytest.mark.parametrize("title", [
        "Требования к программной документации",
        "5.4. Требования к программной документации",
        "Требования к документации",
        "Требования к документации программы",
        "Требования к документации на программу",
    ])
    def test_documentation_section_is_non_requirement(self, title):
        assert _is_requirement_section_by_title(title) is False, (
            f"expected non-requirement for {title!r}, got True"
        )

    @pytest.mark.parametrize("title", [
        "Требования к функциональным характеристикам",
        "Требования к надёжности",
        "Требования к безопасности",
        "Требования к программе",
        "Функциональные требования",
    ])
    def test_program_requirement_section_is_requirement(self, title):
        assert _is_requirement_section_by_title(title) is True, (
            f"expected requirement for {title!r}, got False/None"
        )


# ── _from_candidates filters prepareType="documentation_requirement" ──────


def _artifact(candidates, sections=None, fragments=None):
    return {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "sections": sections or [],
        "fragments": fragments or [],
        "requirement_candidates": candidates,
    }


def _doc_req_candidate(text, frag_id, section_id="6"):
    """Candidate as it arrives from docback: mapped to ctRequirement but with
    prepareType='documentation_requirement' preserved in metadata."""
    return {
        "text": text,
        "section_id": section_id,
        "fragment_id": frag_id,
        "metadata": {
            "sectionCategory": "requirements",   # section 6 tagged requirements by prepare-svc
            "prepareType": "documentation_requirement",
        },
    }


def _prog_req_candidate(text, frag_id, section_id="5.1"):
    return {
        "text": text,
        "section_id": section_id,
        "fragment_id": frag_id,
        "metadata": {
            "sectionCategory": "requirements",
            "prepareType": "requirement_like",
        },
    }


class TestDocReqCandidatesFiltered:
    def setup_method(self):
        cfg = CoverageConfig.from_options({"requirement_extraction": "candidates"})
        self.builder = RequirementBuilder(cfg)

    def test_documentation_req_candidate_is_dropped(self):
        """A candidate with prepareType='documentation_requirement' must never
        become a RequirementUnit, even when its sectionCategory='requirements'."""
        cand = _doc_req_candidate(
            "Должна быть разработана программная документация в объёме, "
            "определённом в ТЗ.",
            frag_id="f-doc1",
        )
        units = self.builder.build(_artifact([cand]))
        assert units == [], (
            f"documentation_requirement candidate leaked through: "
            f"{[u.text for u in units]}"
        )

    def test_program_req_candidate_passes(self):
        """A candidate with prepareType='requirement_like' must pass normally."""
        cand = _prog_req_candidate(
            "Система должна обеспечивать аутентификацию пользователей.",
            frag_id="f-r1",
        )
        units = self.builder.build(_artifact([cand]))
        assert len(units) == 1
        assert "аутентификацию" in units[0].text

    def test_mixed_candidates_only_program_reqs_pass(self):
        """Mix of documentation + program candidates: only program ones pass."""
        candidates = [
            _prog_req_candidate(
                "Система должна обеспечивать поиск по индексу.",
                frag_id="f-r1", section_id="5.1",
            ),
            _prog_req_candidate(
                "Программа не должна аварийно завершаться при ошибке ввода.",
                frag_id="f-r2", section_id="5.2",
            ),
            _doc_req_candidate(
                "Должно быть разработано руководство оператора по ГОСТ 19.505.",
                frag_id="f-d1", section_id="6",
            ),
            _doc_req_candidate(
                "Перечень документов, разрабатываемых на систему, определяется ТЗ.",
                frag_id="f-d2", section_id="6",
            ),
        ]
        units = self.builder.build(_artifact(candidates))
        texts = [u.text for u in units]

        # Program requirements must be present
        assert any("поиск по индексу" in t for t in texts), (
            "program req 1 missing"
        )
        assert any("аварийно завершаться" in t for t in texts), (
            "program req 2 missing"
        )

        # Documentation requirements must be absent
        assert not any("руководство оператора" in t for t in texts), (
            f"documentation req slipped through: {texts}"
        )
        assert not any("разрабатываемых на систему" in t for t in texts), (
            f"documentation req slipped through: {texts}"
        )
        assert len(units) == 2, f"expected 2, got {len(units)}: {texts}"

    def test_no_prepare_type_in_metadata_uses_section_gate(self):
        """Candidates without prepareType in metadata fall back to the
        sectionCategory gate — sectionCategory='requirements' still passes."""
        cand = {
            "text": "Система должна реализовывать многофакторную аутентификацию.",
            "section_id": "5.1",
            "fragment_id": "f-r1",
            "metadata": {"sectionCategory": "requirements"},  # no prepareType
        }
        units = self.builder.build(_artifact([cand]))
        assert len(units) == 1, (
            f"candidate without prepareType was incorrectly dropped: {units}"
        )


# ── Section-boost does NOT fire on documentation-titled sections ───────────


class TestSectionBoostExcludesDocumentationSections:
    def setup_method(self):
        cfg = CoverageConfig.from_options({"requirement_extraction": "auto"})
        self.builder = RequirementBuilder(cfg)

    def test_boost_skips_documentation_section(self):
        """Section 'Требования к программной документации' must NOT be
        boosted even though its title contains 'требовани'."""
        sections = [
            {"section_id": "5.1", "title": "Требования к функциональным характеристикам", "level": 2},
            {"section_id": "6",   "title": "Требования к программной документации", "level": 1},
        ]
        fragments = [
            {"section_id": "5.1", "fragment_id": "f1",
             "text": "Система должна обеспечивать поиск по индексу.", "kind": "paragraph"},
            {"section_id": "6",   "fragment_id": "f2",
             "text": "Должно быть разработано руководство оператора.", "kind": "paragraph"},
        ]
        candidates = [
            {"text": "Система должна обеспечивать поиск по индексу.",
             "section_id": "5.1", "fragment_id": "f1",
             "metadata": {"sectionCategory": "requirements", "prepareType": "requirement_like"}},
        ]

        out = self.builder.build({
            "document_id": "doc-tz",
            "doc_role": "tz",
            "sections": sections,
            "fragments": fragments,
            "requirement_candidates": candidates,
        })

        boost_units = [u for u in out if "::boost::" in u.req_id]
        for u in boost_units:
            assert u.source_section_id != "6", (
                f"boost extracted from documentation section: {u.text!r}"
            )
        # Only the candidate from 5.1 must be present for the doc section.
        doc_texts = [u.text for u in out if "руководство оператора" in u.text]
        assert doc_texts == [], (
            f"documentation sentence found in output: {doc_texts}"
        )

    def test_boost_still_fires_on_program_requirement_section(self):
        """Boost must still fire on genuine program-requirement sections
        with the same classification path."""
        sections = [
            {"section_id": "5.1", "title": "Требования к надёжности", "level": 2},
        ]
        fragments = [
            {"section_id": "5.1", "fragment_id": "f1",
             "text": "Система должна работать без сбоев 24/7. "
                     "Программа не должна аварийно завершаться при ошибке ввода.",
             "kind": "paragraph"},
        ]
        candidates = [
            {"text": "Система должна работать без сбоев 24/7.",
             "section_id": "5.1", "fragment_id": "f1",
             "metadata": {"sectionCategory": "requirements", "prepareType": "requirement_like"}},
        ]
        out = self.builder.build({
            "document_id": "doc-tz",
            "doc_role": "tz",
            "sections": sections,
            "fragments": fragments,
            "requirement_candidates": candidates,
        })
        texts = [u.text for u in out]
        assert any("аварийно завершаться" in t for t in texts), (
            f"boost missed program req in reliability section; got: {texts}"
        )


# ── applicability_for routing for DOCUMENTATION_REQUIREMENT ───────────────


class TestDocumentationRequirementApplicability:
    """DOCUMENTATION_REQUIREMENT belongs to the PMI-only routing bucket:
    APPLICABLE for pmi (the TZ can specify what the PMI must contain),
    NOT_APPLICABLE for pz (documentation structure is not a PZ concern),
    NOT_APPLICABLE for any other role."""

    @pytest.mark.parametrize("role", ["pmi", "PMI", "  pmi  "])
    def test_doc_req_applicable_for_pmi(self, role):
        result = applicability_for(RequirementType.DOCUMENTATION_REQUIREMENT, role)
        assert result == Applicability.APPLICABLE, (
            f"DOCUMENTATION_REQUIREMENT must be APPLICABLE for pmi (got {result!r} "
            f"with role={role!r})"
        )

    @pytest.mark.parametrize("role", ["pz", "PZ"])
    def test_doc_req_not_applicable_for_pz(self, role):
        result = applicability_for(RequirementType.DOCUMENTATION_REQUIREMENT, role)
        assert result == Applicability.NOT_APPLICABLE, (
            f"DOCUMENTATION_REQUIREMENT must be NOT_APPLICABLE for pz (got {result!r})"
        )

    @pytest.mark.parametrize("role", ["tz", "other", "", None])
    def test_doc_req_not_applicable_for_other_roles(self, role):
        result = applicability_for(RequirementType.DOCUMENTATION_REQUIREMENT, role)
        assert result == Applicability.NOT_APPLICABLE, (
            f"DOCUMENTATION_REQUIREMENT must be NOT_APPLICABLE for role={role!r} "
            f"(got {result!r})"
        )

    def test_other_optional_types_still_applicable_for_pz(self):
        """Verify _PMI_ONLY_TYPES does not accidentally affect other
        OPTIONAL types — INTERFACE and ENVIRONMENT_REQUIREMENT must
        still be APPLICABLE for pz."""
        assert applicability_for(RequirementType.INTERFACE, "pz") == Applicability.APPLICABLE
        assert applicability_for(RequirementType.ENVIRONMENT_REQUIREMENT, "pz") == Applicability.APPLICABLE

    def test_pz_only_types_still_not_applicable_for_pmi(self):
        """Adding _PMI_ONLY_TYPES must not break _PZ_ONLY_TYPES routing."""
        assert applicability_for(RequirementType.ARCHITECTURE_IMPLEMENTATION, "pmi") == Applicability.NOT_APPLICABLE
        assert applicability_for(RequirementType.ECONOMIC_OR_NEED, "pmi") == Applicability.NOT_APPLICABLE
