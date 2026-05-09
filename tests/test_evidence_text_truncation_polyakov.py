"""
Polyakov-regression: EvidenceItem.text in CoverageReportBuilder must
not cut mid-word.

Concrete user-visible bug on the Polyakov demo (2026-05-10): the
«Требования» UI tab rendered an evidence card with text
  «… изменение статуса публикации в рамках жизненн»
— the «жизненного цикла» got chopped at character 300 by a hard
`unit.text[:300]` slice in CoverageAggregator.aggregate. Reads as a
garbled / unfinished requirement to the reviewer.

The fix replaces the hard slice with the sentence-boundary truncator
already used by SECTION_WINDOW unit construction, and bumps the
budget to 600 chars so a typical PMI restatement paragraph fits
without truncation at all.
"""
from __future__ import annotations

from app.application.use_cases.aggregate_coverage import CoverageAggregator
from app.domain.c_quality_enums import (
    CoverageUnitType,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)


def _req() -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Клиентская часть должна предоставлять интерфейс.",
        normalized_text="клиентская часть должна предоставлять интерфейс.",
        requirement_type=RequirementType.FUNCTIONAL,
    )


def _unit(text: str) -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        unit_type=CoverageUnitType.SECTION_WINDOW,
        text=text,
        normalized_text=text.lower(),
    )


def test_polyakov_evidence_text_does_not_cut_midword() -> None:
    """The Polyakov demo unit text reconstructed: a long PMI window
    that previously got cut mid-word. Now the truncator picks the last
    sentence boundary within the 600-char budget instead — output ends
    in «.», not in mid-word."""
    long_text = (
        "Система должна обеспечивать работу с научными проектами, "
        "включая создание проектов, их редактирование и управление "
        "метаданными. "
        "Система должна обеспечивать загрузку, хранение и управление "
        "публикациями, включая добавление метаданных, прикрепление "
        "файлов и изменение статуса публикации в рамках жизненного цикла. "
        "Система должна обеспечивать реализацию рабочего процесса, "
        "связанного с публикациями, включающего в себя этапы загрузки, "
        "модерации, публикации, редактирования и деактивации "
        "материалов. "
        "Система должна обеспечивать поиск по публикациям и проектам "
        "с фильтрацией по авторам, темам и ключевым словам."
    )
    req = _req()
    unit = _unit(long_text)
    j = PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=0.7,
    )
    cand = RetrievedCandidate(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        retrieval_score=0.5,
    )
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=[j],
        candidates_by_unit_id={"u1": cand},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )

    assert result.evidence, "evidence list must be populated"
    ev_text = result.evidence[0].text
    # Must respect the 600-char wire budget.
    assert len(ev_text) <= 600
    # MUST end at a sentence boundary, not mid-word.
    assert ev_text.endswith((".", "!", "?", "»")), (
        f"evidence text not terminated cleanly: ...{ev_text[-60:]!r}"
    )
    # MUST NOT contain the «жизненн» mid-word artefact.
    assert "жизненн " not in ev_text and not ev_text.endswith("жизненн")


def test_evidence_text_short_unit_passed_through_intact() -> None:
    """When unit.text fits in budget, no truncation happens."""
    short_text = "Система должна обеспечивать поиск."
    unit = _unit(short_text)
    j = PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=0.7,
    )
    cand = RetrievedCandidate(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        retrieval_score=0.5,
    )
    result = CoverageAggregator().aggregate(
        requirement=_req(),
        judgments=[j],
        candidates_by_unit_id={"u1": cand},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.evidence[0].text == short_text
