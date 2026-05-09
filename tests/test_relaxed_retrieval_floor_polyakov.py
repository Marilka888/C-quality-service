"""
Polyakov-regression (2026-05-10, R-floor): SECURITY / PERFORMANCE /
RELIABILITY requirements use specialised vocabulary («атаки типа
Внедрение кода», «время отклика», «отказоустойчивость») that rarely
shares lexical mass with the surrounding PMI/PZ narrative. Their
retrieval scores cap at ~0.20-0.30 even when the LLM judge correctly
identifies partial coverage. The 0.30 medium-retrieval floor was
rejecting those judgments wholesale → MISSING_LOW_CONFIDENCE →
criticalCount inflation for what the LLM read as legitimate partial
coverage.

Two real Polyakov rows hit this:
  * 0.14::sent1 PMI «Время отклика 3 сек»: judge PARTIAL,
    retrieval=0.28 → rejected by 0.30 floor.
  * 0.18::sent2 PMI «атаки типа Внедрение кода»: judge PARTIAL conf
    0.7, retrieval=0.20 → rejected.

The fix relaxes the medium retrieval floor to 0.20 for these three
specialised-vocabulary types. The LLM confidence and grounding gates
still apply unchanged, so we're not admitting hallucinations — just
loosening the lex-density gate that hurts narrow-domain requirements.
"""
from __future__ import annotations

from app.application.use_cases.aggregate_coverage import (
    CoverageAggregator,
    SUBCODE_PARTIAL,
    SUBCODE_MISSING_LOW_CONFIDENCE,
)
from app.domain.c_quality_enums import (
    CoverageStatus,
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


def _req(req_type: RequirementType) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Test requirement.",
        normalized_text="test requirement.",
        requirement_type=req_type,
    )


def _unit(unit_id: str = "u1", text: str = "Test evidence.") -> CoverageUnit:
    return CoverageUnit(
        unit_id=unit_id,
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        unit_type=CoverageUnitType.PARAGRAPH,
        text=text,
        normalized_text=text.lower(),
    )


def _cand(unit_id: str, score: float) -> RetrievedCandidate:
    return RetrievedCandidate(
        req_id="r1", unit_id=unit_id, target_document_id="doc-pmi",
        retrieval_score=score,
    )


def _partial_judgment(unit_id: str = "u1", conf: float = 0.7) -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id=unit_id, target_document_id="doc-pmi",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=conf,
        cited_phrases=["test"],
        explanation="Test rationale.",
    )


def _aggregate(req: RequirementUnit, retrieval_score: float) -> object:
    unit = _unit(text="Test evidence with content matching test.")
    judgment = _partial_judgment(conf=0.7)
    return CoverageAggregator().aggregate(
        requirement=req,
        judgments=[judgment],
        candidates_by_unit_id={"u1": _cand("u1", retrieval_score)},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )


# ── Polyakov repro: relaxed types cross the 0.20 floor ──────────────


def test_polyakov_security_partial_at_0_20_retrieval_accepted() -> None:
    """0.18::sent2 PMI shape: SECURITY req, judge PARTIAL conf 0.7,
    retrieval 0.20. Old: rejected by 0.30 floor → MISSING_LOW_CONFIDENCE.
    New: accepted via the 0.20 SECURITY floor → PARTIAL."""
    result = _aggregate(_req(RequirementType.SECURITY), retrieval_score=0.20)
    assert result.status == CoverageStatus.PARTIAL, (
        f"SECURITY+0.20 must be accepted as PARTIAL after R-floor; "
        f"got {result.status} ({result.status_subcode})"
    )
    assert result.status_subcode == SUBCODE_PARTIAL


def test_polyakov_performance_partial_at_0_28_retrieval_accepted() -> None:
    """0.14::sent1 PMI shape: PERFORMANCE req, retrieval 0.28. Old:
    rejected by 0.30 floor. New: accepted via the 0.20 PERFORMANCE
    floor → PARTIAL."""
    result = _aggregate(_req(RequirementType.PERFORMANCE), retrieval_score=0.28)
    assert result.status == CoverageStatus.PARTIAL
    assert result.status_subcode == SUBCODE_PARTIAL


def test_polyakov_reliability_partial_at_0_22_accepted() -> None:
    """RELIABILITY also relaxed (Polyakov has reliability rows too)."""
    result = _aggregate(_req(RequirementType.RELIABILITY), retrieval_score=0.22)
    assert result.status == CoverageStatus.PARTIAL


# ── Non-relaxed types still respect the 0.30 floor ─────────────────


def test_functional_partial_at_0_25_still_rejected() -> None:
    """FUNCTIONAL is NOT in the relaxed-floor set — wide-vocab type
    where lex retrieval works well. Floor stays at 0.30 → 0.25
    rejected → MISSING_LOW_CONFIDENCE."""
    result = _aggregate(_req(RequirementType.FUNCTIONAL), retrieval_score=0.25)
    assert result.status == CoverageStatus.MISSING
    assert result.status_subcode == SUBCODE_MISSING_LOW_CONFIDENCE


def test_other_partial_at_0_25_still_rejected() -> None:
    """OTHER also keeps the 0.30 floor — defensive, applies to most
    non-specialised requirement types."""
    result = _aggregate(_req(RequirementType.OTHER), retrieval_score=0.25)
    assert result.status == CoverageStatus.MISSING
    assert result.status_subcode == SUBCODE_MISSING_LOW_CONFIDENCE


# ── Sanity: high retrieval still accepted for all types ─────────────


def test_security_high_retrieval_obviously_accepted() -> None:
    """Sanity: at retrieval 0.50 (well above any floor), SECURITY
    PARTIAL is accepted. Confirms the relaxed path doesn't break
    the high-retrieval happy path."""
    result = _aggregate(_req(RequirementType.SECURITY), retrieval_score=0.50)
    assert result.status == CoverageStatus.PARTIAL


def test_security_below_relaxed_floor_still_rejected() -> None:
    """Sanity: even with relaxation, retrieval 0.10 (below the 0.20
    relaxed floor) is genuinely too low — must reject. Otherwise
    we'd let through pure noise."""
    result = _aggregate(_req(RequirementType.SECURITY), retrieval_score=0.10)
    assert result.status == CoverageStatus.MISSING
    assert result.status_subcode == SUBCODE_MISSING_LOW_CONFIDENCE
