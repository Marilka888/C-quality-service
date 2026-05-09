"""
Polyakov-regression: PMI coverage units must be free of list-intro
stubs and section-heading-only paragraphs, and SECTION_WINDOW
truncation must respect sentence boundaries.

Concrete bad unit observed on the Polyakov demo:
  «Требования к функциональным характеристикам. Клиентская часть
   приложения должна обеспечивать возможность выполнения следующих
   функций:. Система должна обеспечивать работу с научными
   проектами, включая создание проектов, их редактирование и
   управление метаданными. Система должна обеспечивать загрузку»

Three issues compound:
  1. The section heading «Требования к функциональным характеристикам.»
     is prepended to the window text — duplicates topic on the wire,
     dilutes Jaccard against the requirement.
  2. The list-intro stub «…следующих функций:.» is included as a
     fragment — adds «:.» noise that reads like a truncated sentence.
  3. The window is truncated at the last space («должна обеспечивать
     загрузку») rather than at a sentence boundary — looks like a
     half-finished requirement to the LLM judge.

The fix combines three filters in build_coverage_units:
  * `_is_list_intro_stub` filters list-intro fragments before they
    enter paragraph or window units.
  * `_is_section_heading_only` filters paragraphs that are merely the
    section heading text (some DOCX exports re-emit headings as body
    paragraphs).
  * `_truncate_at_sentence_boundary` truncates window text at the
    last «.», «!», «?», «»» within budget instead of at the last
    space.
  * SECTION_WINDOW units no longer prepend the section title — the
    title lives in metadata.section_title and the section-prior
    boost in retrieve_candidates already uses it.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.build_coverage_units import (
    CoverageUnitBuilder,
    _is_list_intro_stub,
    _is_section_heading_only,
    _truncate_at_sentence_boundary,
)
from app.domain.c_quality_enums import CoverageUnitType


# ── Helpers ──────────────────────────────────────────────────────────


def _frag(section_id: str, frag_id: str, text: str) -> dict:
    return {
        "section_id": section_id,
        "fragment_id": frag_id,
        "kind": "paragraph",
        "text": text,
    }


# ── _is_list_intro_stub ─────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    # Polyakov pattern: post-period normalisation tail.
    "Система должна предоставить пользователю следующий набор функций:.",
    "Клиентская часть должна обеспечивать выполнение следующих функций:.",
    # Plain trailing colon + cue noun.
    "Перечень обязательных функций:",
    "Указанных ниже параметров:",
    "Приведённых далее условий:",
    # Tail-only «:.» without explicit cue noun.
    "Эти параметры заданы:.",
])
def test_polyakov_list_intro_stub_filtered(text: str) -> None:
    assert _is_list_intro_stub(text)


@pytest.mark.parametrize("text", [
    # Real requirement sentences must NOT match.
    "Система должна обеспечивать поиск по публикациям.",
    "Программный интерфейс должен быть представлен в виде REST API.",
    # Plain trailing colon WITHOUT a list-cue noun — kept (PDF block
    # detector elsewhere needs that signal).
    "Параметры подключения к базе данных заданы:",
])
def test_real_requirement_not_filtered_as_list_intro(text: str) -> None:
    assert not _is_list_intro_stub(text)


# ── _is_section_heading_only ────────────────────────────────────────


def test_polyakov_section_heading_only_filtered() -> None:
    # Some DOCX exports re-emit the section heading as a body paragraph.
    assert _is_section_heading_only(
        "Требования к функциональным характеристикам.",
        "Требования к функциональным характеристикам",
    )
    assert _is_section_heading_only(
        "ТРЕБОВАНИЯ К ПРОГРАММЕ",
        "Требования к программе",
    )


def test_real_paragraph_not_filtered_as_heading() -> None:
    # A genuine requirement paragraph that happens to mention the
    # section topic must NOT be filtered.
    assert not _is_section_heading_only(
        "Система должна обеспечивать выполнение следующих функций.",
        "Требования к функциональным характеристикам",
    )


def test_no_title_means_no_filter() -> None:
    assert not _is_section_heading_only("Любой текст.", "")


# ── _truncate_at_sentence_boundary ──────────────────────────────────


def test_truncate_prefers_sentence_boundary() -> None:
    text = (
        "Система должна выполнять регистрацию. "
        "Система должна выполнять авторизацию. "
        "Система должна выполнять восстановление пароля."
    )
    truncated = _truncate_at_sentence_boundary(text, 70)
    # Must end at a sentence boundary, not mid-word or mid-sentence.
    assert truncated.endswith(".")
    # Must not include the third sentence (out of budget).
    assert "восстановление" not in truncated


def test_truncate_falls_back_to_space_when_no_boundary() -> None:
    # Worst case — single long sentence with no boundary in the budget.
    text = "Очень длинное непрерывное предложение " * 30
    truncated = _truncate_at_sentence_boundary(text, 50)
    # Must respect the budget.
    assert len(truncated) <= 50
    # Must NOT end mid-word — output must be space-aligned, so the
    # last word is complete (whichever it happens to be).
    last_token = truncated.rsplit(" ", 1)[-1].strip()
    # In our repeating-word text, every token is a known word.
    known_tokens = {"Очень", "длинное", "непрерывное", "предложение"}
    assert last_token in known_tokens, (
        f"truncate produced mid-word output: {last_token!r} from {truncated!r}"
    )


# ── End-to-end: section-window unit cleanliness ─────────────────────


def test_polyakov_section_window_excludes_heading_and_list_intro_stub() -> None:
    """The exact Polyakov scenario reconstructed: section heading +
    list-intro + bullet bodies. The resulting window unit must NOT
    contain the heading text and must NOT contain the «:.» tail."""
    artifact = {
        "document_id": "doc-pmi",
        "doc_role": "pmi",
        "sections": [
            {"section_id": "3.1", "title": "Требования к функциональным характеристикам"}
        ],
        "fragments": [
            # Section heading re-emitted as body paragraph.
            _frag("3.1", "p1", "Требования к функциональным характеристикам."),
            # List-intro stub.
            _frag("3.1", "p2",
                  "Клиентская часть приложения должна обеспечивать "
                  "возможность выполнения следующих функций:."),
            # Real coverage paragraphs (bullet-body shape — no «Система
            # должна» head — so the atomic-window rule still groups
            # them; bullet bodies need windowing to make sense as
            # evidence).
            _frag("3.1", "p3",
                  "работа с научными проектами, включая создание проектов, "
                  "их редактирование и управление метаданными;"),
            _frag("3.1", "p4",
                  "загрузка, хранение и управление публикациями, включая "
                  "добавление метаданных, прикрепление файлов и изменение "
                  "статуса."),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)

    # Window unit must exist.
    windows = [u for u in units if u.unit_type == CoverageUnitType.SECTION_WINDOW]
    assert windows, "section-window unit should be produced"
    win_text = windows[0].text

    # MUST NOT contain the section heading.
    assert "Требования к функциональным характеристикам" not in win_text, (
        f"window text leaks section heading: {win_text!r}"
    )
    # MUST NOT contain the list-intro stub tail.
    assert "следующих функций:" not in win_text, (
        f"window text leaks list-intro stub: {win_text!r}"
    )
    # MUST contain the real coverage content.
    assert "научными проектами" in win_text or "загрузку" in win_text

    # Title still travels via metadata.
    assert (windows[0].metadata or {}).get("section_title") == \
        "Требования к функциональным характеристикам"

    # Paragraph units for the noise fragments must NOT be created.
    paragraph_texts = [
        u.text for u in units if u.unit_type == CoverageUnitType.PARAGRAPH
    ]
    for pt in paragraph_texts:
        assert "Требования к функциональным характеристикам." != pt
        assert not pt.endswith("следующих функций:.")


def test_polyakov_section_window_text_ends_at_sentence_boundary() -> None:
    """Force the truncation path: build a section with text longer than
    _MAX_WINDOW_CHARS (1800) and verify the unit text ends at a real
    sentence boundary, not mid-word like «должна обеспечивать загрузку».
    Uses MULTIPLE fragments — single-fragment sections produce a
    PARAGRAPH unit whose text is the duplicate of any window, so dedup
    would suppress the window. The window-building code joins multiple
    fragments and only THEN truncates."""
    # Bullet-body shape (no «Subject должн…» head) so the atomic-window
    # rule keeps grouping fragments — the test exercises the truncation
    # path on the resulting concatenated window text.
    sentence = (
        "корректная обработка запросов от клиентов и формирование "
        "ответов на стороне сервиса. "
    )
    # Three different long paragraphs in the same section so the window
    # builder concatenates them into one window > _MAX_WINDOW_CHARS.
    artifact = {
        "document_id": "doc-pmi",
        "doc_role": "pmi",
        "sections": [{"section_id": "3.1", "title": "Test section"}],
        "fragments": [
            _frag("3.1", "p1", (sentence + "Регистрация. ") * 10),
            _frag("3.1", "p2", (sentence + "Авторизация. ") * 10),
            _frag("3.1", "p3", (sentence + "Аутентификация. ") * 10),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    windows = [u for u in units if u.unit_type == CoverageUnitType.SECTION_WINDOW]
    assert windows, "window unit must be produced when section has ≥2 fragments"
    text = windows[0].text
    # Must terminate at a sentence boundary (period).
    assert text.endswith("."), f"window text not terminated cleanly: ...{text[-60:]!r}"
    # Must respect the budget (with small margin for rstrip).
    assert len(text) <= 1800, f"window text exceeds budget: {len(text)} chars"
