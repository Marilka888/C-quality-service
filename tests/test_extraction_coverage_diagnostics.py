"""
PR-I tests: extraction coverage sanity guard + diagnostics.

The «Череухо» package symptom: TZ source has 39 requirement markers
("должн*", "необходимо", "требуется") but only 3 requirements were
extracted because Word headings were inconsistent and the body of the
ТЗ landed in a heading-less preamble that the title-block stripper
dropped wholesale. The sanity guard catches this class of bug
generically — without hardcoding any package-specific check.
"""
from __future__ import annotations

from app.application.use_cases.run_coverage_analysis import (
    _build_extraction_diagnostics,
)
from app.domain.c_quality_enums import Modality, RequirementType
from app.domain.c_quality_models import RequirementUnit


def _req(text: str, sid: str = "preamble") -> RequirementUnit:
    return RequirementUnit(
        req_id=f"r-{text[:8]}",
        source_document_id="doc-tz",
        source_section_id=sid,
        text=text,
        normalized_text=text.lower(),
    )


def test_diagnostics_low_coverage_triggers():
    """39 markers in source, only 3 extracted → low coverage flag fires."""
    artifact = {
        "sections": [
            {
                "section_id": "preamble",
                "category": "other",
                "text": (
                    "Система должна обеспечивать аутентификацию. "
                    "Программа должна поддерживать ролевую модель. "
                    "Должно быть обеспечено хранение журнала. "
                    "Должна вестись регистрация действий пользователя. "
                    "Система должна выполнять резервное копирование. "
                    "Программа должна обеспечивать поиск по метаданным. "
                    "Должна быть реализована обработка ошибок. "
                    "Система должна поддерживать workflow публикаций. "
                    "Необходимо обеспечить доступ через REST API. "
                    "Требуется поддержка форматов JSON и XML. "
                    "Должны быть реализованы права доступа. "
                    "Система должна обеспечивать защиту данных."
                ),
            },
        ],
    }
    requirements = [_req("Только одно требование")]
    diag = _build_extraction_diagnostics(artifact, requirements)
    assert diag["marker_count"] >= 10, diag
    assert diag["extracted_count"] == 1
    assert diag["low_extraction_coverage"] is True
    assert diag["suspected_reason"]
    # When only one section exists and it's not requirement-bearing,
    # the diagnostic should mention the heading-style hypothesis.
    assert (
        "заголовки" in diag["suspected_reason"].lower()
        or "requirement-bearing" in diag["suspected_reason"].lower()
    )


def test_diagnostics_no_low_coverage_when_extraction_is_proportional():
    """30 markers, 12 extracted → ratio ≈40%, NOT flagged."""
    artifact = {
        "sections": [
            {
                "section_id": "4.1",
                "category": "requirements",
                "text": (
                    "Система должна обеспечивать аутентификацию. " * 6
                    + "Программа должна поддерживать роли. " * 6
                    + "Должно быть обеспечено хранение журналов. " * 6
                ),
            },
        ],
    }
    requirements = [_req(f"req-{i}") for i in range(12)]
    diag = _build_extraction_diagnostics(artifact, requirements)
    assert diag["marker_count"] >= 18, diag
    assert diag["extracted_count"] == 12
    assert diag["low_extraction_coverage"] is False


def test_diagnostics_no_markers_no_warning():
    """Empty source → no markers, no warning (clean signal not noise)."""
    artifact = {"sections": [{"section_id": "1", "text": "Lorem ipsum dolor sit."}]}
    diag = _build_extraction_diagnostics(artifact, [])
    assert diag["marker_count"] == 0
    assert diag["low_extraction_coverage"] is False


def test_diagnostics_per_section_distribution():
    """sections_per_extracted_req groups requirement counts by source_section_id."""
    artifact = {"sections": []}
    requirements = [
        _req("a", sid="4.1"),
        _req("b", sid="4.1"),
        _req("c", sid="4.2"),
        _req("d", sid="preamble"),
    ]
    diag = _build_extraction_diagnostics(artifact, requirements)
    assert diag["sections_per_extracted_req"] == {"4.1": 2, "4.2": 1, "preamble": 1}


def test_diagnostics_counts_requirement_sections():
    artifact = {
        "sections": [
            {"section_id": "1", "category": "other"},
            {"section_id": "4", "category": "requirements"},
            {"section_id": "5", "category": "test_methods"},
            {"section_id": "9", "category": "bibliography"},
        ],
    }
    diag = _build_extraction_diagnostics(artifact, [])
    assert diag["sections_seen"] == 4
    assert diag["requirement_sections_seen"] == 2  # 4 + 5


def test_diagnostics_falls_back_to_fragments_when_no_sections():
    artifact = {
        "fragments": [
            {"text": "Система должна обеспечивать поиск." * 3},
            {"text": "Должна вестись регистрация." * 4},
        ],
    }
    diag = _build_extraction_diagnostics(artifact, [_req("x")])
    assert diag["marker_count"] >= 5
