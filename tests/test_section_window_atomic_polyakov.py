"""
Polyakov-regression (2026-05-10): SECTION_WINDOW units bundled four
distinct self-contained normative sentences («Система должна …») into
one PMI evidence card. The reviewer reads it as «слишком много
требований в одном кандидате» — multiple requirements per evidence card
is exactly the noise pattern atomic candidates were supposed to
prevent.

Concrete bad unit observed on the demo:
  «Система должна обеспечивать работу с научными проектами…
   Система должна обеспечивать загрузку, хранение и управление
   публикациями… Система должна обеспечивать реализацию рабочего
   процесса… Система должна обеспечивать возможность регистрации…»

Each of these is already its own paragraph unit, so the SECTION_WINDOW
provides no extra coverage signal — it just glues four requirements
into one card.

The fix changes the window builder from fixed-size groups of
`_SECTION_WINDOW_SIZE` to a greedy walk that closes the window before
the second self-contained normative fragment. Sections with bullet-
style fragments (no normative subject head) are unaffected — windowing
still helps there.
"""
from __future__ import annotations

from app.application.use_cases.build_coverage_units import (
    CoverageUnitBuilder,
    _is_self_contained_normative,
)
from app.domain.c_quality_enums import CoverageUnitType


def _frag(section_id: str, frag_id: str, text: str) -> dict:
    return {
        "section_id": section_id,
        "fragment_id": frag_id,
        "kind": "paragraph",
        "text": text,
    }


# ── _is_self_contained_normative ────────────────────────────────────


def test_self_contained_detects_subject_modal_head() -> None:
    assert _is_self_contained_normative(
        "Система должна обеспечивать поиск по публикациям."
    )
    assert _is_self_contained_normative(
        "Программа должна предоставлять REST API."
    )
    assert _is_self_contained_normative(
        "Клиентская часть должна отображать форму входа."
    )
    assert _is_self_contained_normative(
        "Сервис должен возвращать ответ в формате JSON."
    )


def test_self_contained_rejects_bullet_fragments() -> None:
    # Bullet-item bodies don't carry the «Subject должн…» head and need
    # windowing to combine with siblings into a coherent unit.
    assert not _is_self_contained_normative("создание проектов;")
    assert not _is_self_contained_normative("редактирование метаданных,")
    assert not _is_self_contained_normative("REST API в формате JSON.")
    # Empty / whitespace.
    assert not _is_self_contained_normative("")
    assert not _is_self_contained_normative("   ")


# ── End-to-end: window builder splits at normative boundaries ───────


def test_polyakov_four_normative_sentences_split_into_separate_windows() -> None:
    """Four sequential «Система должна …» sentences must NOT end up in
    one SECTION_WINDOW unit. Each closes its own window."""
    artifact = {
        "document_id": "doc-pmi",
        "doc_role": "pmi",
        "sections": [
            {"section_id": "3.1",
             "title": "Требования к функциональным характеристикам"}
        ],
        "fragments": [
            _frag("3.1", "p1",
                  "Система должна обеспечивать работу с научными проектами, "
                  "включая создание проектов, их редактирование и управление "
                  "метаданными."),
            _frag("3.1", "p2",
                  "Система должна обеспечивать загрузку, хранение и управление "
                  "публикациями, включая добавление метаданных, прикрепление "
                  "файлов и изменение статуса публикации в рамках жизненного "
                  "цикла."),
            _frag("3.1", "p3",
                  "Система должна обеспечивать реализацию рабочего процесса, "
                  "связанного с публикациями, включающего в себя этапы "
                  "загрузки, модерации, публикации, редактирования и "
                  "деактивации материалов."),
            _frag("3.1", "p4",
                  "Система должна обеспечивать возможность регистрации и "
                  "авторизации пользователей."),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    windows = [u for u in units if u.unit_type == CoverageUnitType.SECTION_WINDOW]

    # No window may contain more than one «Система должна» occurrence.
    for w in windows:
        # Count how many distinct self-contained normative sentences
        # are present by counting subject+modal heads.
        n_normative = w.text.count("Система должна")
        assert n_normative <= 1, (
            f"window glues {n_normative} normative sentences: {w.text!r}"
        )


def test_bullet_section_still_windows_normally() -> None:
    """A section whose fragments are bullet-item bodies (no normative
    subject head) must still produce a windowed unit so the LLM judge
    can see the combined context. Greedy windowing must not regress
    this case."""
    artifact = {
        "document_id": "doc-pmi",
        "doc_role": "pmi",
        "sections": [{"section_id": "2.1", "title": "Состав поставки"}],
        "fragments": [
            _frag("2.1", "p1",
                  "исходные коды клиентской части на TypeScript;"),
            _frag("2.1", "p2",
                  "конфигурационные файлы сборки и развёртывания;"),
            _frag("2.1", "p3",
                  "набор автоматизированных тестов;"),
            _frag("2.1", "p4",
                  "руководство пользователя в формате PDF."),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    windows = [u for u in units if u.unit_type == CoverageUnitType.SECTION_WINDOW]
    assert windows, "bullet section must still produce a window unit"
    # The single window should aggregate all four bullet bodies.
    assert "TypeScript" in windows[0].text
    assert "PDF" in windows[0].text


def test_mixed_intro_plus_normative_keeps_intro_with_first() -> None:
    """A non-normative intro paragraph followed by a normative sentence
    may share a window (the intro carries no normative head, so it
    doesn't trigger a split). The next normative sentence opens a new
    window."""
    artifact = {
        "document_id": "doc-pmi",
        "doc_role": "pmi",
        "sections": [{"section_id": "3.2", "title": "Описание подсистемы"}],
        "fragments": [
            _frag("3.2", "p1",
                  "Подсистема публикаций отвечает за хранение материалов."),
            _frag("3.2", "p2",
                  "Система должна поддерживать загрузку файлов размером "
                  "до 100 МБ."),
            _frag("3.2", "p3",
                  "Система должна индексировать публикации по ключевым "
                  "словам."),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    windows = [u for u in units if u.unit_type == CoverageUnitType.SECTION_WINDOW]
    for w in windows:
        assert w.text.count("Система должна") <= 1, (
            f"window bundles two requirements: {w.text!r}"
        )
