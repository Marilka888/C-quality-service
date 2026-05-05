"""
PR-K regression tests:

  * AdaptiveCandidateSelector — k decision tree (no candidates,
    NO_EVIDENCE, STRONG + wide margin, MEDIUM/narrow margin, WEAK,
    critical / numeric-constraints).
  * Conditional reranker — fires only when first-stage signals are
    weak.
  * EvidenceBasedCoverageAggregator — confident COVERED, ungrounded
    COVERED → MISSING_LOW_GROUNDING, low-conf COVERED →
    MISSING_LOW_CONFIDENCE, unverified CONFLICT → PARTIAL,
    NOT_APPLICABLE / OUT_OF_SCOPE rows skip LLM.
  * Evidence trace — populated when debug.enabled.
  * Score reason / evidence_strength — populated by retriever.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.adaptive_candidate_selector import (
    SelectionResult,
    select_candidates,
)
from app.application.use_cases.aggregate_coverage import (
    SUBCODE_COVERED,
    SUBCODE_CONFLICT_UNVERIFIED,
    SUBCODE_CONFLICT_VERIFIED,
    SUBCODE_MISSING_LOW_CONFIDENCE,
    SUBCODE_MISSING_LOW_GROUNDING,
    SUBCODE_MISSING_NO_EVIDENCE,
    SUBCODE_NOT_APPLICABLE,
    SUBCODE_OPTIONAL_NOT_FOUND,
    SUBCODE_OUT_OF_SCOPE,
    SUBCODE_PARTIAL,
    CoverageAggregator,
)
from app.application.use_cases.applicability import (
    coverage_requirement_level_for,
    evidence_strength_from_score,
)
from app.application.use_cases.retrieve_candidates import (
    CandidateRetriever,
    _conditional_should_rerank,
)
from app.core.config import (
    CoverageConfig,
    CoverageAggregatorConfig,
    CoverageDebugConfig,
    CoverageRerankerConfig,
    CoverageRetrievalConfig,
)
from app.domain.c_quality_enums import (
    Applicability,
    CoverageRequirementLevel,
    CoverageStatus,
    EvidenceStrength,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.embeddings.simple import BagOfWordsEmbeddingBackend


class TestCoverageConfigOptions:
    def test_from_options_exposes_evidence_and_partial_thresholds(self):
        cfg = CoverageConfig.from_options(
            {
                "evidence_floor": 0.42,
                "skip_llm_below_floor": False,
                "partial_confidence_threshold": 0.72,
            }
        )

        assert cfg.retrieval.evidence_floor == pytest.approx(0.42)
        assert cfg.retrieval.skip_llm_below_floor is False
        assert cfg.aggregator.partial_confidence_threshold == pytest.approx(0.72)

    def test_skip_llm_below_floor_default_enabled(self):
        cfg = CoverageRetrievalConfig()
        assert cfg.skip_llm_below_floor is True

    def test_floor_skip_keeps_critical_and_numeric_on_llm_path(self):
        from app.application.use_cases.run_coverage_analysis import (
            _can_skip_llm_below_floor,
        )

        assert _can_skip_llm_below_floor(
            _req(req_type=RequirementType.FUNCTIONAL)
        )
        assert not _can_skip_llm_below_floor(
            _req(req_type=RequirementType.SECURITY)
        )
        assert not _can_skip_llm_below_floor(
            _req(
                req_type=RequirementType.FUNCTIONAL,
                constraints=[Constraint(kind="time", operator="<=", value=3, unit="seconds")],
            )
        )


# ── fixtures ────────────────────────────────────────────────────────────


def _req(
    text: str = "Система должна обеспечивать аутентификацию.",
    req_type: RequirementType = RequirementType.FUNCTIONAL,
    constraints=None,
) -> RequirementUnit:
    return RequirementUnit(
        req_id="req-1",
        source_document_id="doc-tz",
        text=text,
        normalized_text=text.lower(),
        requirement_type=req_type,
        constraints=constraints or [],
    )


def _unit(text: str = "x", unit_id: str = "u1", role: str = "pmi") -> CoverageUnit:
    return CoverageUnit(
        unit_id=unit_id,
        target_document_id="doc-pmi",
        target_doc_role=role,
        text=text,
        normalized_text=text.lower(),
    )


def _candidate(unit_id: str, score: float, lex: float = 0.0, sem: float = 0.0) -> RetrievedCandidate:
    return RetrievedCandidate(
        req_id="req-1", unit_id=unit_id, target_document_id="doc-pmi",
        retrieval_score=score, lexical_score=lex, semantic_score=sem,
    )


def _judgment(
    unit_id: str,
    label: LLMLabel,
    conf: float = 0.85,
    low_confidence: bool = False,
    grounding_failed: bool = False,
    verifier_actions=None,
    explanation: str = "ok",
) -> PairJudgment:
    return PairJudgment(
        req_id="req-1", unit_id=unit_id, target_document_id="doc-pmi",
        llm_label=label, rule_adjusted_label=label, llm_confidence=conf,
        grounding_failed=grounding_failed,
        low_confidence=low_confidence,
        verifier_actions=list(verifier_actions or []),
        explanation=explanation,
    )


# ── AdaptiveCandidateSelector ───────────────────────────────────────────


class TestAdaptiveCandidateSelector:
    """The selector decides per-pair k based on retrieval signals."""

    def setup_method(self):
        self.cfg = CoverageRetrievalConfig()

    def test_empty_shortlist_skips_llm(self):
        result = select_candidates(_req(), [], self.cfg)
        assert result.skip_llm is True
        assert result.selected == []
        assert result.selected_k == 0

    def test_no_evidence_top1_skips_llm(self):
        # All candidates below evidence_strength_weak_threshold (0.12)
        cands = [_candidate("u1", 0.05), _candidate("u2", 0.04)]
        result = select_candidates(_req(), cands, self.cfg)
        assert result.skip_llm is True
        assert result.skip_reason == "all candidates NO_EVIDENCE"
        assert all(c.evidence_strength == EvidenceStrength.NO_EVIDENCE for c in cands)

    def test_strong_top1_with_wide_margin_picks_one(self):
        # Top-1 STRONG (>=0.45), margin >=0.08
        cands = [_candidate("u1", 0.85), _candidate("u2", 0.30), _candidate("u3", 0.25)]
        result = select_candidates(_req(), cands, self.cfg)
        assert result.skip_llm is False
        assert result.selected_k == 1
        assert result.selected[0].unit_id == "u1"
        assert len(result.discarded) == 2
        # Selected candidate is mutated as a side effect
        assert cands[0].selected_for_llm is True
        assert cands[1].selected_for_llm is False

    def test_strong_top1_with_narrow_margin_broadens_to_three(self):
        # Top-1 STRONG, top-2 also STRONG, margin <0.08
        cands = [
            _candidate("u1", 0.50), _candidate("u2", 0.48),
            _candidate("u3", 0.46), _candidate("u4", 0.40),
        ]
        result = select_candidates(_req(), cands, self.cfg)
        assert result.selected_k == 3
        assert [c.unit_id for c in result.selected] == ["u1", "u2", "u3"]

    def test_medium_top1_broadens_to_three(self):
        # Top-1 MEDIUM (>=0.25, <0.45)
        cands = [_candidate("u1", 0.30), _candidate("u2", 0.27), _candidate("u3", 0.20)]
        result = select_candidates(_req(), cands, self.cfg)
        assert result.selected_k == 3

    def test_weak_top1_broadens_to_three(self):
        # Top-1 WEAK (>=0.12, <0.25). With skip_llm_below_floor=True
        # (default) the selector skips the LLM here for non-critical reqs;
        # disable the flag so this test continues exercising the WEAK
        # broadening branch it was written for.
        cfg = CoverageRetrievalConfig(skip_llm_below_floor=False)
        cands = [_candidate("u1", 0.20), _candidate("u2", 0.18), _candidate("u3", 0.15)]
        result = select_candidates(_req(), cands, cfg)
        assert result.selected_k == 3
        assert "WEAK" in result.selection_reason or "weak" in result.selection_reason.lower()

    def test_skip_llm_below_floor_skips_non_critical(self):
        """Audit fix: top-1 in [weak_threshold, evidence_floor) for a
        non-critical requirement should skip the LLM. The aggregator
        would have rejected any positive verdict via
        medium_retrieval_threshold anyway — saves a call per pair."""
        cfg = CoverageRetrievalConfig()  # skip_llm_below_floor=True default
        cands = [_candidate("u1", 0.20), _candidate("u2", 0.18)]
        req = _req(req_type=RequirementType.FUNCTIONAL)
        result = select_candidates(req, cands, cfg)
        assert result.skip_llm is True
        assert result.selected == []
        assert "evidence_floor" in result.selection_reason

    def test_skip_llm_below_floor_keeps_critical(self):
        """Critical requirements still go to the LLM at low retrieval —
        paraphrase risk justifies the call."""
        cfg = CoverageRetrievalConfig()
        cands = [_candidate("u1", 0.20), _candidate("u2", 0.18), _candidate("u3", 0.15)]
        req = _req(req_type=RequirementType.SECURITY)
        result = select_candidates(req, cands, cfg)
        assert result.skip_llm is False
        assert result.selected_k > 0

    def test_skip_llm_below_floor_keeps_numeric_constraints(self):
        cfg = CoverageRetrievalConfig()
        cands = [_candidate("u1", 0.20), _candidate("u2", 0.18)]
        req = _req(
            req_type=RequirementType.FUNCTIONAL,
            constraints=[Constraint(kind="timeout", operator="<=", value=2, unit="seconds")],
        )
        result = select_candidates(req, cands, cfg)
        assert result.skip_llm is False
        assert result.selected_k > 0

    def test_critical_type_uses_max_k(self):
        # SECURITY → critical → broad sweep
        req = _req(req_type=RequirementType.SECURITY)
        cands = [
            _candidate("u1", 0.60), _candidate("u2", 0.55),
            _candidate("u3", 0.50), _candidate("u4", 0.45),
            _candidate("u5", 0.40), _candidate("u6", 0.35),
        ]
        result = select_candidates(req, cands, self.cfg)
        assert result.selected_k == self.cfg.selector_max_k  # 5

    def test_numeric_constraints_force_broad_sweep(self):
        # FUNCTIONAL is not critical, but constraints make it critical
        req = _req(
            req_type=RequirementType.FUNCTIONAL,
            constraints=[Constraint(kind="retention_period", operator=">=", value=90, unit="days")],
        )
        cands = [
            _candidate("u1", 0.50), _candidate("u2", 0.45),
            _candidate("u3", 0.40), _candidate("u4", 0.35),
        ]
        result = select_candidates(req, cands, self.cfg)
        # Selector picks min(selector_max_k, len) = 4
        assert result.selected_k == 4


# ── ConditionalReranker decision ────────────────────────────────────────


class TestConditionalReranker:
    """The reranker should fire only when first-stage signals are weak."""

    def _rrcfg(self) -> CoverageRerankerConfig:
        return CoverageRerankerConfig(
            mode="conditional",
            conditional_top1_threshold=0.45,
            conditional_min_margin=0.08,
        )

    def test_strong_top1_wide_margin_skips_rerank(self):
        cands = [_candidate("u1", 0.80), _candidate("u2", 0.50)]
        should, reason = _conditional_should_rerank(_req(), cands, self._rrcfg())
        assert should is False
        assert "skipped" in reason

    def test_weak_top1_triggers_rerank(self):
        cands = [_candidate("u1", 0.30), _candidate("u2", 0.20)]
        should, reason = _conditional_should_rerank(_req(), cands, self._rrcfg())
        assert should is True
        assert "top1 score" in reason

    def test_narrow_margin_triggers_rerank(self):
        cands = [_candidate("u1", 0.60), _candidate("u2", 0.58)]
        should, reason = _conditional_should_rerank(_req(), cands, self._rrcfg())
        assert should is True
        assert "margin" in reason

    def test_critical_type_triggers_rerank(self):
        cands = [_candidate("u1", 0.80), _candidate("u2", 0.40)]
        req = _req(req_type=RequirementType.SECURITY)
        should, reason = _conditional_should_rerank(req, cands, self._rrcfg())
        assert should is True
        assert "critical" in reason.lower()

    def test_paraphrase_signal_triggers_rerank(self):
        # high semantic, low lexical on top-1
        cands = [
            _candidate("u1", 0.80, lex=0.10, sem=0.70),
            _candidate("u2", 0.40),
        ]
        should, reason = _conditional_should_rerank(_req(), cands, self._rrcfg())
        assert should is True
        assert "paraphrase" in reason.lower()


# ── EvidenceBasedCoverageAggregator decision tree ───────────────────────


class TestEvidenceBasedAggregator:
    def setup_method(self):
        self.agg = CoverageAggregator()
        self.aggcfg = CoverageAggregatorConfig()
        self.dbgcfg = CoverageDebugConfig(enabled=True)

    def _aggregate(self, judgments, candidates):
        req = _req()
        units = {f"u{i}": _unit(f"text {i}", unit_id=f"u{i}") for i in range(len(judgments) + 1)}
        for j in judgments:
            units.setdefault(j.unit_id, _unit("x", unit_id=j.unit_id))
        return self.agg.aggregate(
            requirement=req,
            judgments=judgments,
            candidates_by_unit_id=candidates,
            units_by_id=units,
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            aggregator_cfg=self.aggcfg,
            debug_cfg=self.dbgcfg,
        )

    def test_confident_grounded_covered_accepted(self):
        j = _judgment("u1", LLMLabel.COVERED, conf=0.85, low_confidence=False)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.COVERED
        assert res.status_subcode == SUBCODE_COVERED
        assert res.winning_candidate_id == "u1"
        assert res.final_confidence == pytest.approx(0.85)
        assert res.aggregation_reason and "COVERED accepted" in res.aggregation_reason

    def test_ungrounded_covered_demoted_to_missing_low_grounding(self):
        # PR-K P0: ungrounded means citation didn't substring-match evidence,
        # which is `grounding_failed=True` (not just `low_confidence=True`).
        # `low_confidence` alone now means below-evidence-floor — that's a
        # retrieval-quality flag, NOT a grounding violation.
        j = _judgment("u1", LLMLabel.COVERED, conf=0.85, grounding_failed=True)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.MISSING
        assert res.status_subcode == SUBCODE_MISSING_LOW_GROUNDING

    def test_below_floor_with_grounded_covered_stays_covered(self):
        """PR-K P0 contract: when retrieval is below evidence_floor BUT the
        citation is grounded, the aggregator must still accept COVERED.
        The row carries low_confidence=True for UI dimming, status remains
        COVERED. Real-package symptom (Polyakov 0.20::sent1)."""
        j = _judgment(
            "u1", LLMLabel.COVERED, conf=1.0,
            low_confidence=True,           # below evidence_floor
            grounding_failed=False,        # but properly grounded
        )
        cands = {"u1": _candidate("u1", score=0.45)}  # > medium_threshold (0.30)
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.COVERED, (
            f"grounded COVERED with below-floor retrieval must stay COVERED; "
            f"got {res.status}, subcode={res.status_subcode}"
        )
        assert res.low_confidence is True  # UI dim flag preserved

    def test_low_conf_covered_demoted_to_missing_low_confidence(self):
        j = _judgment("u1", LLMLabel.COVERED, conf=0.4, low_confidence=False)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.MISSING
        assert res.status_subcode == SUBCODE_MISSING_LOW_CONFIDENCE

    def test_below_medium_retrieval_covered_demoted(self):
        # conf is high, grounded, but retrieval below medium_threshold (0.30)
        j = _judgment("u1", LLMLabel.COVERED, conf=0.85, low_confidence=False)
        cands = {"u1": _candidate("u1", score=0.20)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.MISSING
        assert res.status_subcode == SUBCODE_MISSING_LOW_CONFIDENCE

    def test_unverified_conflict_produces_missing_not_partial(self):
        # Regression: unconfirmed CONFLICT used to produce PARTIAL, which
        # inflated the PARTIAL count with off-topic pairs (Поляков runs).
        # Now it must produce MISSING with subcode CONFLICT_UNVERIFIED.
        j = _judgment(
            "u1", LLMLabel.CONFLICT, conf=0.85,
            verifier_actions=["no_op_llm_conflict_unverified"],
        )
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.MISSING, (
            f"unconfirmed CONFLICT must produce MISSING, not {res.status}"
        )
        assert res.status_subcode == SUBCODE_CONFLICT_UNVERIFIED, (
            f"unconfirmed CONFLICT must use CONFLICT_UNVERIFIED subcode, "
            f"got {res.status_subcode!r}"
        )

    def test_unverified_conflict_yields_to_genuine_partial(self):
        # When an unconfirmed CONFLICT exists alongside a genuine PARTIAL
        # judgment from a different candidate, the PARTIAL must win.
        j_conflict = _judgment(
            "u1", LLMLabel.CONFLICT, conf=0.85,
            verifier_actions=["no_op_llm_conflict_unverified"],
        )
        j_partial = _judgment(
            "u2", LLMLabel.PARTIAL, conf=0.75,
        )
        cands = {
            "u1": _candidate("u1", score=0.5),
            "u2": _candidate("u2", score=0.4),
        }
        res = self._aggregate([j_conflict, j_partial], cands)
        assert res.status == CoverageStatus.PARTIAL, (
            f"genuine PARTIAL candidate must win over unconfirmed CONFLICT; "
            f"got {res.status}"
        )

    def test_unverified_conflict_suppressed_no_topical_link(self):
        # Specific Поляков regression: verifier fires
        # suppress_negation_no_topical_link → judgment stays CONFLICT with
        # unverified tag → must produce MISSING, not PARTIAL.
        j = _judgment(
            "u1", LLMLabel.CONFLICT, conf=1.0,
            verifier_actions=["suppress_negation_no_topical_link",
                              "no_op_llm_conflict_unverified"],
        )
        cands = {"u1": _candidate("u1", score=0.47)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.MISSING, (
            f"off-topic unconfirmed CONFLICT must be MISSING; got {res.status}"
        )

    def test_verified_conflict_accepted(self):
        j = _judgment(
            "u1", LLMLabel.CONFLICT, conf=0.85,
            verifier_actions=["conflict_confirmed_numeric"],
        )
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.CONFLICT
        assert res.status_subcode == SUBCODE_CONFLICT_VERIFIED

    def test_not_applicable_skips_llm_and_marks_subcode(self):
        # ARCHITECTURE_IMPLEMENTATION on PMI is NOT_APPLICABLE.
        req = _req(req_type=RequirementType.ARCHITECTURE_IMPLEMENTATION)
        res = self.agg.aggregate(
            requirement=req,
            judgments=[],
            candidates_by_unit_id={},
            units_by_id={},
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            aggregator_cfg=self.aggcfg,
            debug_cfg=self.dbgcfg,
        )
        assert res.applicability == Applicability.NOT_APPLICABLE
        assert res.status == CoverageStatus.MISSING
        assert res.status_subcode == SUBCODE_NOT_APPLICABLE
        assert res.should_affect_critical is False
        assert res.should_affect_grade is False

    def test_out_of_scope_skips_llm_and_marks_subcode(self):
        # DELIVERY_REQUIREMENT is OUT_OF_SCOPE everywhere.
        req = _req(req_type=RequirementType.DELIVERY_REQUIREMENT)
        res = self.agg.aggregate(
            requirement=req,
            judgments=[],
            candidates_by_unit_id={},
            units_by_id={},
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            aggregator_cfg=self.aggcfg,
            debug_cfg=self.dbgcfg,
        )
        assert res.applicability == Applicability.OUT_OF_SCOPE
        assert res.status_subcode == SUBCODE_OUT_OF_SCOPE
        assert res.should_affect_critical is False
        assert res.should_affect_grade is False

    def test_optional_not_found_when_no_evidence_and_optional_level(self):
        # INTERFACE on PZ is APPLICABLE + OPTIONAL.
        req = _req(req_type=RequirementType.INTERFACE)
        res = self.agg.aggregate(
            requirement=req,
            judgments=[],
            candidates_by_unit_id={},
            units_by_id={},
            target_document_id="doc-pz",
            target_doc_role="pz",
            aggregator_cfg=self.aggcfg,
            debug_cfg=self.dbgcfg,
        )
        assert res.coverage_requirement_level == CoverageRequirementLevel.OPTIONAL
        assert res.status_subcode == SUBCODE_OPTIONAL_NOT_FOUND
        assert res.should_affect_critical is False

    def test_evidence_trace_populated_when_debug_enabled(self):
        j = _judgment("u1", LLMLabel.COVERED, conf=0.85)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.evidence_trace is not None
        assert "decision_log" in res.evidence_trace
        assert "candidates" in res.evidence_trace
        assert res.evidence_trace["candidates"][0]["unit_id"] == "u1"

    def test_evidence_trace_omitted_when_debug_disabled(self):
        self.dbgcfg = CoverageDebugConfig(enabled=False)
        j = _judgment("u1", LLMLabel.COVERED, conf=0.85)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.evidence_trace is None

    # ── Fix B: partial_confidence_threshold ──────────────────────────────

    def test_partial_below_confidence_threshold_becomes_missing(self):
        """PARTIAL with conf below partial_confidence_threshold must be
        rejected and reported as MISSING_LOW_CONFIDENCE."""
        self.aggcfg = CoverageAggregatorConfig(partial_confidence_threshold=0.60)
        j = _judgment("u1", LLMLabel.PARTIAL, conf=0.45)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.MISSING, (
            f"PARTIAL conf=0.45 below threshold=0.60 must become MISSING; "
            f"got {res.status}"
        )
        assert res.status_subcode == SUBCODE_MISSING_LOW_CONFIDENCE

    def test_partial_at_or_above_confidence_threshold_accepted(self):
        """PARTIAL with conf at partial_confidence_threshold must be accepted."""
        self.aggcfg = CoverageAggregatorConfig(partial_confidence_threshold=0.60)
        j = _judgment("u1", LLMLabel.PARTIAL, conf=0.60)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.PARTIAL, (
            f"PARTIAL conf=0.60 at threshold=0.60 must be accepted; "
            f"got {res.status}"
        )
        assert res.status_subcode == SUBCODE_PARTIAL

    def test_partial_threshold_zero_accepts_any_conf(self):
        """partial_confidence_threshold=0.0 disables the gate entirely."""
        self.aggcfg = CoverageAggregatorConfig(partial_confidence_threshold=0.0)
        j = _judgment("u1", LLMLabel.PARTIAL, conf=0.10)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j], cands)
        assert res.status == CoverageStatus.PARTIAL, (
            f"threshold=0.0 must accept any conf; got {res.status}"
        )

    def test_partial_threshold_default_is_conservative(self):
        """Default partial_confidence_threshold (0.65) should accept
        conf=0.66 (above) and reject conf=0.55 (below)."""
        # Above default
        self.aggcfg = CoverageAggregatorConfig()
        j_above = _judgment("u1", LLMLabel.PARTIAL, conf=0.66)
        cands = {"u1": _candidate("u1", score=0.5)}
        res = self._aggregate([j_above], cands)
        assert res.status == CoverageStatus.PARTIAL

        # Below default
        j_below = _judgment("u1", LLMLabel.PARTIAL, conf=0.55)
        res2 = self._aggregate([j_below], cands)
        assert res2.status == CoverageStatus.MISSING


# ── score_reason / evidence_strength on candidates ──────────────────────


class TestRetrieverDiagnostics:
    def test_score_reason_and_evidence_strength_populated(self):
        cfg = CoverageRetrievalConfig()
        cfg.min_retrieval_score = 0.0
        retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())

        req = RequirementUnit(
            req_id="r1", source_document_id="doc-tz",
            text="Хранить журнал событий не менее 90 дней.",
            normalized_text="хранить журнал событий не менее 90 дней",
            requirement_type=RequirementType.STORAGE,
        )
        unit_a = CoverageUnit(
            unit_id="a", target_document_id="doc-pmi", target_doc_role="pmi",
            text="Журнал событий хранится 90 дней.",
            normalized_text="журнал событий хранится 90 дней",
        )
        unit_b = CoverageUnit(
            unit_id="b", target_document_id="doc-pmi", target_doc_role="pmi",
            text="Совершенно несвязанный фрагмент.",
            normalized_text="совершенно несвязанный фрагмент",
        )
        results = retriever.retrieve(req, [unit_a, unit_b])
        assert results, "retrieval should produce candidates"
        for c in results:
            # PR-K invariants: every returned candidate carries diagnostic
            # fields.
            assert c.score_reason, f"expected score_reason on {c.unit_id}"
            assert c.evidence_strength is not None

    def test_evidence_strength_bins(self):
        # Direct bin function check against the default thresholds.
        assert evidence_strength_from_score(0.50) == EvidenceStrength.STRONG
        assert evidence_strength_from_score(0.30) == EvidenceStrength.MEDIUM
        assert evidence_strength_from_score(0.20) == EvidenceStrength.WEAK
        assert evidence_strength_from_score(0.05) == EvidenceStrength.NO_EVIDENCE


# ── Coverage requirement level mapping ──────────────────────────────────


class TestVerifierIRRELEVANTOverride:
    """PR-K P0: rule-verifier must promote IRRELEVANT-LLM-verdict to
    CONFLICT when a numeric mismatch with a topical link is detected.

    Smoke-time symptom: qwen2.5:3b labels "журнал 90 дней" vs "журнал 30
    суток" as IRRELEVANT, missing the obvious 30/90 contradiction.
    Before the fix, the verifier returned IRRELEVANT immediately and the
    aggregator scored the pair MISSING_NO_EVIDENCE. After the fix, the
    deterministic numeric-rule promotes to CONFLICT with the topical-link
    guard intact (so unrelated pairs sharing a number coincidence are
    NOT promoted).
    """

    def setup_method(self):
        from app.application.use_cases.verify_pairs import PairVerifier
        self.verifier = PairVerifier()

    def _req(self, text: str, constraints):
        return RequirementUnit(
            req_id="r1",
            source_document_id="doc-tz",
            text=text,
            normalized_text=text.lower(),
            requirement_type=RequirementType.LOGGING,
            constraints=constraints,
        )

    def _unit(self, text: str, constraints):
        return CoverageUnit(
            unit_id="u1",
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text=text,
            normalized_text=text.lower(),
            constraints=constraints,
        )

    def test_llm_irrelevant_with_numeric_mismatch_and_topical_link_promoted_to_conflict(self):
        """Reproduces the qwen2.5:3b smoke-time bug end-to-end."""
        req = self._req(
            "Журнал событий безопасности должен храниться не менее 90 дней с момента записи.",
            constraints=[Constraint(kind="retention_period", operator=">=", value=90, unit="days")],
        )
        unit = self._unit(
            "Проверить, что журнал событий хранится за последние 30 суток.",
            constraints=[Constraint(kind="retention_period", operator="=", value=30, unit="days")],
        )
        # LLM said IRRELEVANT — that's the bug we're fixing.
        j = PairJudgment(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            llm_label=LLMLabel.IRRELEVANT,
            rule_adjusted_label=LLMLabel.IRRELEVANT,
            llm_confidence=0.6,
        )
        out = self.verifier.verify(j, req, unit)
        # Verifier must override.
        assert out.rule_adjusted_label == LLMLabel.CONFLICT, (
            f"expected CONFLICT, got {out.rule_adjusted_label}; "
            f"actions={out.verifier_actions}"
        )
        assert any(a.startswith("conflict_confirmed") for a in out.verifier_actions), (
            f"expected conflict_confirmed_* action; got {out.verifier_actions}"
        )
        # PR-K P0: confidence must be raised so the aggregator's
        # CONFLICT confidence gate (0.70 default) doesn't reject the
        # rule-confirmed verdict.
        assert out.llm_confidence >= 0.85, (
            f"verifier-promoted CONFLICT must carry rule confidence ≥0.85; "
            f"got {out.llm_confidence}"
        )

    def test_llm_irrelevant_without_topical_link_stays_irrelevant(self):
        """Sanity guard: a pair with the same constraint UNIT class but
        zero topical link must NOT be promoted. Random number coincidences
        ("отклик 90 мс" vs "хранить 30 дней" — both time-class) should
        stay IRRELEVANT."""
        req = self._req(
            "Время отклика интерфейса должно быть не более 90 миллисекунд.",
            constraints=[Constraint(kind="response_time", operator="<=", value=90, unit="ms")],
        )
        unit = self._unit(
            "Документация поставляется в формате PDF.",
            constraints=[Constraint(kind="generic", operator="=", value=30, unit="days")],
        )
        j = PairJudgment(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            llm_label=LLMLabel.IRRELEVANT,
            rule_adjusted_label=LLMLabel.IRRELEVANT,
            llm_confidence=0.1,
        )
        out = self.verifier.verify(j, req, unit)
        # Different constraint kinds + zero entity overlap + few shared tokens.
        # Topical-link guard must keep this IRRELEVANT.
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
            f"random number coincidence promoted to {out.rule_adjusted_label}; "
            f"actions={out.verifier_actions}"
        )

    def test_llm_irrelevant_with_no_numeric_constraints_stays_irrelevant(self):
        """When neither side has constraints, the IRRELEVANT verdict
        passes through untouched."""
        req = self._req(
            "Система должна предоставлять интерфейс администратора.",
            constraints=[],
        )
        unit = self._unit(
            "Подпись передаётся в виде PNG-изображения.",
            constraints=[],
        )
        j = PairJudgment(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            llm_label=LLMLabel.IRRELEVANT,
            rule_adjusted_label=LLMLabel.IRRELEVANT,
            llm_confidence=0.05,
        )
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT
        assert "no_op_irrelevant" in out.verifier_actions


class TestCoverageRequirementLevel:
    def test_required_for_functional_on_pmi(self):
        assert (
            coverage_requirement_level_for(RequirementType.FUNCTIONAL, "pmi")
            == CoverageRequirementLevel.REQUIRED
        )

    def test_optional_for_interface_on_pz(self):
        assert (
            coverage_requirement_level_for(RequirementType.INTERFACE, "pz")
            == CoverageRequirementLevel.OPTIONAL
        )

    def test_not_applicable_for_architecture_on_pmi(self):
        assert (
            coverage_requirement_level_for(RequirementType.ARCHITECTURE_IMPLEMENTATION, "pmi")
            == CoverageRequirementLevel.NOT_APPLICABLE
        )

    def test_not_applicable_for_delivery_anywhere(self):
        assert (
            coverage_requirement_level_for(RequirementType.DELIVERY_REQUIREMENT, "pmi")
            == CoverageRequirementLevel.NOT_APPLICABLE
        )
        assert (
            coverage_requirement_level_for(RequirementType.DELIVERY_REQUIREMENT, "pz")
            == CoverageRequirementLevel.NOT_APPLICABLE
        )
