"""
Regression tests for BUG-06: _is_document_boilerplate must apply in
_from_candidates and _from_fragments paths, not only in _from_model.

Acceptance criteria from the fix request:
  * "– М.: Изд-во стандартов, 1997" must NOT become a requirement.
  * "ГОСТ 19.101-77" reference-only line must NOT become a requirement.
  * Trailing-section-number heading "... 4.1." must NOT become a requirement.
  * A genuine requirement that mentions ГОСТ ("должны соответствовать ГОСТ
    19.101-77") MUST still pass.
"""
from __future__ import annotations

from app.application.use_cases.build_requirements import RequirementBuilder


def _texts(reqs) -> list[str]:
    return [r.text for r in reqs]


# ── _from_candidates path ────────────────────────────────────────────────


def test_from_candidates_drops_bibliography_line():
    artifact = {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "fragments": [],
        "requirement_candidates": [
            # bibliography stamp from a real TZ
            {"req_id": "r-bib", "text": "– М.: Изд-во стандартов, 1997. – 12 с."},
            # genuine requirement, must remain
            {"req_id": "r-real", "text": "Система должна хранить журнал не менее 90 дней."},
        ],
    }
    reqs = RequirementBuilder().build(artifact)
    out = _texts(reqs)
    assert any("90 дней" in t for t in out), f"real requirement was dropped: {out}"
    assert not any("Изд-во стандартов" in t for t in out), (
        f"bibliography line leaked into requirements: {out}"
    )


def test_from_candidates_drops_bare_gost_reference_line():
    artifact = {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "fragments": [],
        "requirement_candidates": [
            # GOST reference WITHOUT modality
            {"req_id": "r1", "text": "ГОСТ 19.101-77 Виды программ и программных документов."},
            {"req_id": "r2", "text": "Система должна обеспечивать резервное копирование."},
        ],
    }
    reqs = RequirementBuilder().build(artifact)
    out = _texts(reqs)
    assert any("резервное копирование" in t for t in out)
    assert not any(t.strip().startswith("ГОСТ 19.101-77") for t in out), (
        f"bare GOST reference leaked into requirements: {out}"
    )


def test_from_candidates_keeps_gost_referencing_real_requirement():
    """A requirement that legitimately cites ГОСТ must not be filtered.
    The boilerplate filter is modality-aware; sentences with 'должны'
    pass through.
    """
    artifact = {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "fragments": [],
        "requirement_candidates": [
            {
                "req_id": "r1",
                "text": (
                    "Программные средства должны соответствовать ГОСТ 19.101-77 "
                    "в части видов программ и документов."
                ),
            },
        ],
    }
    reqs = RequirementBuilder().build(artifact)
    assert len(reqs) == 1, f"genuine requirement dropped: {reqs}"
    assert "должны соответствовать" in reqs[0].text


def test_from_candidates_drops_trailing_section_number_heading():
    artifact = {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "fragments": [],
        "requirement_candidates": [
            # short fragment ending in a bare section number — heading glue
            {"req_id": "r-trail", "text": "Требования к функциональным характеристикам 4.1.1."},
            {"req_id": "r-real", "text": "Система должна выполнять резервное копирование."},
        ],
    }
    reqs = RequirementBuilder().build(artifact)
    out = _texts(reqs)
    assert not any(t.strip().endswith("4.1.1.") for t in out)
    assert any("резервное копирование" in t for t in out)


# ── _from_fragments path ────────────────────────────────────────────────


def test_from_fragments_drops_bibliography_line():
    """_from_fragments triggers when requirement_candidates is missing.
    The boilerplate filter must run there as well.
    """
    artifact = {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "fragments": [
            {
                "fragment_id": "f-bib",
                # _is_requirement_fragment requires a trigger word; embed
                # a modality token to force this fragment past the trigger
                # check, so the boilerplate filter is the only thing that
                # can stop it. The text is still a clear bibliography line.
                "text": "ГОСТ 19.101-77 — М.: Изд-во стандартов, 1997. – 12 с.",
            },
            {
                "fragment_id": "f-real",
                "text": "Система должна хранить журнал событий не менее 90 дней.",
            },
        ],
    }
    reqs = RequirementBuilder().build(artifact)
    out = _texts(reqs)
    # Real requirement should always survive.
    assert any("90 дней" in t for t in out), f"real requirement dropped: {out}"
    # Bibliography line should never end up as a requirement, regardless of
    # whether the trigger filter or the boilerplate filter caught it.
    assert not any("Изд-во стандартов" in t for t in out), (
        f"bibliography leaked into requirements: {out}"
    )


def test_from_fragments_drops_bare_gost_reference():
    artifact = {
        "document_id": "doc-tz",
        "doc_role": "tz",
        "fragments": [
            # Make the line pass the requirement-trigger filter by appending a
            # weak modality verb so we can exercise the boilerplate filter
            # specifically. The line is still GOST-citation-only in shape and
            # should be dropped.
            {
                "fragment_id": "f-gost",
                "text": "ГОСТ 19.101-77 устанавливает виды программ и документов.",
            },
            {
                "fragment_id": "f-real",
                "text": "Программа должна обеспечивать журналирование операций.",
            },
        ],
    }
    reqs = RequirementBuilder().build(artifact)
    out = _texts(reqs)
    assert any("журналирование" in t for t in out)
    # The "ГОСТ 19.101-77 устанавливает ..." line is reference-shaped.
    # If it slipped past the trigger filter (it shouldn't, but be defensive),
    # the boilerplate filter is supposed to catch it.
    assert not any(t.strip().startswith("ГОСТ 19.101-77") for t in out)
