"""
Polyakov-regression Step 4 (2026-05-11): per-row visibility into
LLM-judge unavailability.

The May-11 Polyakov re-run reported «judge backend errored on 5
pair(s)» in the warning, but `cByStatus.UNKNOWN` stayed 0. The
prior C1 fix (commit 6e98e21) only surfaces UNKNOWN when ALL
judgments for a row were sentinels — most real-package runs have
mixed shortlists (some pairs timed out, others succeeded). Those
"partial-shortlist" rows shipped a real verdict computed on
incomplete evidence with no per-row trace.

This step adds `RequirementCoverageResult.unjudged_pair_count` —
the count of `make_unknown_judgment` sentinels that were silently
filtered before Branch C aggregation. The orchestrator/UI can now
show an "incomplete shortlist" badge so the reviewer knows the row
would have benefited from a re-run after LLM restoration.

Counted in two places:
  * Branch B' (all-sentinel UNKNOWN row) — count == total judgments.
  * Branch C with mixed shortlist — count == filtered sentinels;
    the row also gets a real status (COVERED/PARTIAL/MISSING) from
    the surviving real judgments.
"""
from __future__ import annotations

from app.application.use_cases.aggregate_coverage import (
    CoverageAggregator,
    SUBCODE_UNKNOWN_LLM_UNAVAILABLE,
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
from app.infrastructure.llm.coverage_judge import make_unknown_judgment


def _req() -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Система должна обеспечивать поиск.",
        normalized_text="система должна обеспечивать поиск.",
        requirement_type=RequirementType.FUNCTIONAL,
    )


def _unit(unit_id: str = "u1", text: str = "test") -> CoverageUnit:
    return CoverageUnit(
        unit_id=unit_id,
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        unit_type=CoverageUnitType.PARAGRAPH,
        text=text,
        normalized_text=text.lower(),
    )


def _cand(unit_id: str, score: float = 0.5) -> RetrievedCandidate:
    return RetrievedCandidate(
        req_id="r1", unit_id=unit_id, target_document_id="doc-pmi",
        retrieval_score=score,
    )


def _real_partial(unit_id: str = "u1") -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id=unit_id, target_document_id="doc-pmi",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=0.7,
        cited_phrases=["test"],
        explanation="Real PARTIAL.",
    )


# ── All-sentinel branch (B'): full count ───────────────────────────


def test_all_sentinels_unknown_row_carries_full_count() -> None:
    """When all 3 candidate pairs timed out, the row's
    `unjudged_pair_count` equals 3 — the count IS the shortlist
    size."""
    req = _req()
    units = [_unit(f"u{i}") for i in range(3)]
    sentinels = [
        make_unknown_judgment(req, u, "timeout after 240s") for u in units
    ]
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=sentinels,
        candidates_by_unit_id={u.unit_id: _cand(u.unit_id) for u in units},
        units_by_id={u.unit_id: u for u in units},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.UNKNOWN
    assert result.status_subcode == SUBCODE_UNKNOWN_LLM_UNAVAILABLE
    assert result.unjudged_pair_count == 3, (
        f"all-sentinel row must report unjudged_pair_count=3; "
        f"got {result.unjudged_pair_count}"
    )


# ── Mixed shortlist (Branch C): partial count ──────────────────────


def test_mixed_shortlist_carries_partial_count() -> None:
    """Polyakov reproduction: 1 real PARTIAL judgment + 2 sentinels.
    Row gets real status from the PARTIAL but `unjudged_pair_count=2`
    so the UI can flag the partial-shortlist."""
    req = _req()
    real_unit = _unit("u1")
    sent_a = _unit("u2", "sent a")
    sent_b = _unit("u3", "sent b")

    judgments = [
        _real_partial("u1"),
        make_unknown_judgment(req, sent_a, "timeout after 240s"),
        make_unknown_judgment(req, sent_b, "ConnectionError: refused"),
    ]
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=judgments,
        candidates_by_unit_id={
            "u1": _cand("u1", 0.5),
            "u2": _cand("u2", 0.4),
            "u3": _cand("u3", 0.3),
        },
        units_by_id={"u1": real_unit, "u2": sent_a, "u3": sent_b},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.PARTIAL
    assert result.unjudged_pair_count == 2, (
        f"mixed-shortlist row must report unjudged_pair_count=2; "
        f"got {result.unjudged_pair_count}"
    )


def test_no_sentinels_count_is_zero() -> None:
    """Sanity: shortlist with no timeouts → count = 0 (default)."""
    req = _req()
    unit = _unit()
    judgments = [_real_partial("u1")]
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=judgments,
        candidates_by_unit_id={"u1": _cand("u1", 0.5)},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.PARTIAL
    assert result.unjudged_pair_count == 0


def test_field_default_zero_on_legacy_branches() -> None:
    """Sanity: branches that don't touch the count (NOT_APPLICABLE,
    no-judgments) leave it at the default 0."""
    req = RequirementUnit(
        req_id="r1", source_document_id="tz",
        text="Test.", normalized_text="test.",
        # ARCHITECTURE_IMPLEMENTATION is NOT_APPLICABLE for PMI →
        # Branch A returns immediately.
        requirement_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
    )
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=[],
        candidates_by_unit_id={},
        units_by_id={},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.unjudged_pair_count == 0
