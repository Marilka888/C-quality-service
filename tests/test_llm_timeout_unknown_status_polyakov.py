"""
Polyakov-regression (2026-05-10): when the LLM judge backend errors at
request time (timeout / connection / HTTP / parse-exhausted /
unexpected exception), the affected pair must NOT collapse into the
ordinary CoverageStatus.MISSING bucket. That conflates an
infrastructure failure (Ollama overloaded / VRAM exhausted / network
blip) with a documentation defect, inflates criticalCount, and turned
a perfectly-fine package into a CRITICAL one on every transient blip.

The fix:
  * `make_unknown_judgment(...)` returns a sentinel PairJudgment with
    LLMLabel.NOT_JUDGED and a `verifier_actions` tag starting with
    `llm_unavailable:`.
  * Aggregator detects the sentinel via `is_unknown_judgment(...)`.
  * If EVERY judgment for a requirement is a sentinel → status
    UNKNOWN, subcode UNKNOWN_LLM_UNAVAILABLE, severity=low,
    should_affect_critical=False, should_affect_grade=False.
  * If only SOME judgments are sentinels → they are filtered out
    before Branch C aggregation; the remaining real judgments win.
  * Old behaviour (DisabledCoverageJudge fallback → most pairs
    IRRELEVANT → MISSING → false CRITICAL) is gone.
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
from app.infrastructure.llm.coverage_judge import (
    is_unknown_judgment,
    make_unknown_judgment,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _req(req_id: str = "r1", req_type: RequirementType = RequirementType.FUNCTIONAL) -> RequirementUnit:
    return RequirementUnit(
        req_id=req_id,
        source_document_id="tz",
        text="Система должна обеспечивать поиск.",
        normalized_text="система должна обеспечивать поиск.",
        requirement_type=req_type,
    )


def _unit(unit_id: str = "u1", text: str = "Система обеспечивает поиск.") -> CoverageUnit:
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


def _real_partial_judgment(unit_id: str = "u1") -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id=unit_id, target_document_id="doc-pmi",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=0.7,
        explanation="Real LLM verdict.",
    )


# ── make_unknown_judgment / is_unknown_judgment ────────────────────


def test_make_unknown_judgment_carries_sentinel_label_and_tag() -> None:
    j = make_unknown_judgment(_req(), _unit(), "timeout after 120s")
    assert j.llm_label == LLMLabel.NOT_JUDGED
    assert j.rule_adjusted_label == LLMLabel.NOT_JUDGED
    assert j.llm_confidence == 0.0
    # verifier_actions tag the aggregator looks for.
    assert any(
        a.startswith("llm_unavailable:") for a in j.verifier_actions
    ), j.verifier_actions
    assert "timeout after 120s" in j.verifier_actions[0]


def test_is_unknown_judgment_detects_sentinel() -> None:
    sentinel = make_unknown_judgment(_req(), _unit(), "ConnectionError: ...")
    assert is_unknown_judgment(sentinel)


def test_is_unknown_judgment_rejects_real_judgments() -> None:
    real = _real_partial_judgment()
    assert not is_unknown_judgment(real)
    # Even an IRRELEVANT real judgment is not the sentinel.
    real.llm_label = LLMLabel.IRRELEVANT
    real.rule_adjusted_label = LLMLabel.IRRELEVANT
    assert not is_unknown_judgment(real)


def test_is_unknown_judgment_requires_both_label_and_tag() -> None:
    # NOT_JUDGED label without the verifier_actions tag is NOT the
    # sentinel — defensive: don't surface random NOT_JUDGED rows as
    # UNKNOWN unless they came from the runtime-failure factory.
    fake = PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.NOT_JUDGED,
        rule_adjusted_label=LLMLabel.NOT_JUDGED,
    )
    assert not is_unknown_judgment(fake)


# ── Aggregator: all-sentinel → UNKNOWN ─────────────────────────────


def test_all_unknown_judgments_yield_unknown_status() -> None:
    """The Polyakov pain point: Ollama timed out for every candidate
    in the shortlist. Old behaviour produced MISSING + criticalCount++.
    New: status=UNKNOWN, no critical impact, no grade impact."""
    req = _req()
    unit = _unit()
    sentinels = [make_unknown_judgment(req, unit, "timeout after 120s")]
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=sentinels,
        candidates_by_unit_id={"u1": _cand("u1", 0.5)},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.UNKNOWN
    assert result.status_subcode == SUBCODE_UNKNOWN_LLM_UNAVAILABLE
    assert result.should_affect_critical is False, (
        "infrastructure failure must NOT inflate criticalCount"
    )
    assert result.should_affect_grade is False, (
        "infrastructure failure must NOT pull C-grade down"
    )
    assert result.severity == "low"
    # The sentinel's explanation should be preserved on the row so
    # the reviewer can see WHY (timeout vs HTTP vs ConnectionError).
    assert "timeout after 120s" in (result.rationale or ""), result.rationale
    assert "UNKNOWN" in (result.aggregation_reason or "")


def test_partial_unknown_judgments_filtered_real_one_wins() -> None:
    """Mixed shortlist: 1 real PARTIAL judgment + 2 LLM-timeout
    sentinels. The real PARTIAL must drive the verdict; sentinels
    are filtered out, do not poison the aggregation."""
    req = _req()
    real_unit = _unit("u1", "Система обеспечивает поиск.")
    sent_unit_a = _unit("u2", "Дополнительный фрагмент A.")
    sent_unit_b = _unit("u3", "Дополнительный фрагмент B.")

    judgments = [
        _real_partial_judgment("u1"),
        make_unknown_judgment(req, sent_unit_a, "timeout after 120s"),
        make_unknown_judgment(req, sent_unit_b, "ConnectionError: refused"),
    ]
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=judgments,
        candidates_by_unit_id={
            "u1": _cand("u1", 0.5),
            "u2": _cand("u2", 0.4),
            "u3": _cand("u3", 0.3),
        },
        units_by_id={"u1": real_unit, "u2": sent_unit_a, "u3": sent_unit_b},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.PARTIAL, (
        f"real PARTIAL must win, got {result.status}; sentinels poisoned the aggregation"
    )
    assert result.status_subcode != SUBCODE_UNKNOWN_LLM_UNAVAILABLE


def test_unknown_does_not_inflate_critical_for_high_severity_type() -> None:
    """Even a SECURITY-typed requirement (which would be HIGH-priority
    and critical on a real MISSING) must NOT trip critical when the
    sole reason for the verdict is LLM unavailability."""
    req = _req(req_type=RequirementType.SECURITY)
    unit = _unit()
    sentinels = [make_unknown_judgment(req, unit, "timeout after 120s")]
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=sentinels,
        candidates_by_unit_id={"u1": _cand("u1", 0.5)},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.UNKNOWN
    assert result.should_affect_critical is False, (
        "SECURITY MISSING normally goes critical, but UNKNOWN must not"
    )


def test_label_to_status_maps_not_judged_to_unknown() -> None:
    """Lookup table sanity check — used by other aggregator paths if
    NOT_JUDGED ever flows through there."""
    from app.application.use_cases.aggregate_coverage import _label_to_status
    assert _label_to_status(LLMLabel.NOT_JUDGED) == CoverageStatus.UNKNOWN
    # Sanity: existing mappings unchanged.
    assert _label_to_status(LLMLabel.COVERED) == CoverageStatus.COVERED
    assert _label_to_status(LLMLabel.IRRELEVANT) == CoverageStatus.MISSING
