"""
PR-K: AdaptiveCandidateSelector — chooses how many candidates from the
retrieval shortlist actually go to the LLM judge.

The point: a fixed `top_k` is wrong in two directions at once.
  * `top_k=1` is brittle — if the right unit was rank-2, the LLM never
    sees it and the row becomes a false MISSING.
  * `top_k=5` always wastes calls — most pairs have a clear top-1 with
    a wide margin to top-2; sending all 5 to the LLM produces 4 noise
    judgments that the aggregator has to filter out anyway.

Adaptive selection uses cheap signals from the hybrid retriever
(score, evidence_strength, top1-top2 margin, requirement criticality)
to pick a per-pair `selected_k`. Bandwidth-cheap (no LLM calls), and
the savings on free-tier APIs (Groq 100K TPD, Cerebras throttled
flagship) are large.

Outputs `SelectionResult` carrying the chosen list, a list of
discarded candidates, and a one-line `selection_reason` so the
evidence_trace block can render WHY only N candidates were judged.

This module has no LLM dependency and no I/O — pure logic over the
candidates already produced by the retriever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from app.application.use_cases.applicability import evidence_strength_from_score
from app.core.config import CoverageRetrievalConfig
from app.domain.c_quality_enums import CoverageUnitType, EvidenceStrength
from app.domain.c_quality_models import RequirementUnit, RetrievedCandidate


@dataclass
class SelectionResult:
    """Output of AdaptiveCandidateSelector. selected: candidates the
    pipeline should send to the LLM; discarded: the rest, kept for the
    evidence_trace debug block."""
    selected: List[RetrievedCandidate] = field(default_factory=list)
    discarded: List[RetrievedCandidate] = field(default_factory=list)
    selection_reason: str = ""
    selected_k: int = 0
    skip_llm: bool = False
    skip_reason: str = ""


def _is_critical(requirement: RequirementUnit) -> bool:
    """A requirement is "critical" if its kind tends to drive
    correctness/safety decisions: SECURITY, PERFORMANCE, RELIABILITY,
    or any requirement carrying explicit numeric constraints (a
    measurable threshold to verify)."""
    from app.domain.c_quality_enums import RequirementType
    if requirement.requirement_type in {
        RequirementType.SECURITY,
        RequirementType.PERFORMANCE,
        RequirementType.RELIABILITY,
    }:
        return True
    return bool(requirement.constraints)


def select_candidates(
    requirement: RequirementUnit,
    candidates: List[RetrievedCandidate],
    config: CoverageRetrievalConfig,
) -> SelectionResult:
    """Pick the per-pair k. Pure function over the already-scored shortlist.

    Decision tree (tightest-fit rule wins, evaluated top-down):

      0. No candidates                   → skip LLM, skip_reason set.
      1. Top-1 NO_EVIDENCE               → skip LLM (waste).
      2. Critical or numeric constraints → k = min(selector_max_k, len)
                                           (broad sweep so the verifier
                                           and aggregator have material
                                           to confirm/refute conflicts).
      3. Top-1 STRONG and margin wide    → k = 1.
      4. Top-1 MEDIUM, or margin narrow  → k = 3.
      5. Top-1 WEAK                       → k = 3, all marked weak.
      6. Default                          → k = 1.

    The selector never grows above `selector_max_k`. Selected candidates
    have `selected_for_llm = True` mutated on them as a side effect so
    the trace block can mark them.
    """
    if not candidates:
        return SelectionResult(
            selection_reason="no candidates above min_retrieval_score",
            skip_llm=True,
            skip_reason="no candidates in shortlist",
        )

    # SECTION_WINDOW units are retrieval helpers: they aggregate several
    # adjacent paragraphs to help BoW find the right section. Prefer atomic
    # evidence, but keep very strong windows near the top: in real PZ docs
    # a requirement is often covered across neighboring paragraphs
    # ("REST API" in one, "JSON" in the next).
    primary = [
        c for c in candidates
        if c.unit_type != CoverageUnitType.SECTION_WINDOW
    ]
    window_only = [
        c for c in candidates
        if c.unit_type == CoverageUnitType.SECTION_WINDOW
    ]
    effective_candidates = primary if primary else list(candidates)
    window_discarded: List[RetrievedCandidate] = []
    if primary and window_only:
        best_primary = primary[0].retrieval_score
        promoted_windows = [
            c for c in window_only
            if c.retrieval_score >= max(
                config.evidence_strength_strong_threshold,
                best_primary - config.selector_strong_margin,
            )
        ]
        if promoted_windows:
            effective_candidates = sorted(
                primary + promoted_windows,
                key=lambda c: c.retrieval_score,
                reverse=True,
            )
            promoted_ids = {c.unit_id for c in promoted_windows}
            window_discarded = [c for c in window_only if c.unit_id not in promoted_ids]
        else:
            window_discarded = window_only

    # Compute evidence_strength for each candidate using config-driven
    # thresholds (so re-tuning doesn't require code changes).
    strong = config.evidence_strength_strong_threshold
    medium = config.evidence_strength_medium_threshold
    weak = config.evidence_strength_weak_threshold
    for c in effective_candidates:
        c.evidence_strength = evidence_strength_from_score(
            c.retrieval_score, strong=strong, medium=medium, weak=weak
        )

    top1 = effective_candidates[0]
    top2 = effective_candidates[1] if len(effective_candidates) > 1 else None
    margin = (top1.retrieval_score - top2.retrieval_score) if top2 is not None else top1.retrieval_score

    max_k = min(config.selector_max_k, len(effective_candidates))

    # Rule 1: NO_EVIDENCE — skip LLM entirely.
    if top1.evidence_strength == EvidenceStrength.NO_EVIDENCE:
        return SelectionResult(
            selected=[],
            discarded=list(effective_candidates) + window_discarded,
            selection_reason=(
                f"top1 retrieval_score={top1.retrieval_score:.3f} below "
                f"weak threshold ({weak:.2f}); skipping LLM."
            ),
            selected_k=0,
            skip_llm=True,
            skip_reason="all candidates NO_EVIDENCE",
        )

    # Rule 1b: skip_llm_below_floor — when top-1 retrieval is in the
    # [weak_threshold, evidence_floor) band for a non-critical requirement,
    # the aggregator's medium_retrieval_threshold would reject any positive
    # verdict the LLM produces. The LLM call is pure waste. Critical
    # requirements (SECURITY / PERFORMANCE / RELIABILITY / numeric
    # constraints) still go to the LLM — paraphrase risk justifies the call.
    if (
        getattr(config, "skip_llm_below_floor", False)
        and top1.retrieval_score < config.evidence_floor
        and not _is_critical(requirement)
    ):
        return SelectionResult(
            selected=[],
            discarded=list(effective_candidates) + window_discarded,
            selection_reason=(
                f"top1 retrieval_score={top1.retrieval_score:.3f} below "
                f"evidence_floor ({config.evidence_floor:.2f}); "
                f"skip_llm_below_floor enabled and requirement is non-critical."
            ),
            selected_k=0,
            skip_llm=True,
            skip_reason=(
                f"top1 retrieval below evidence_floor "
                f"({config.evidence_floor:.2f}); non-critical requirement"
            ),
        )

    # Rule 2: critical / has-numeric-constraints — broad sweep.
    if _is_critical(requirement):
        selected = effective_candidates[:max_k]
        for c in selected:
            c.selected_for_llm = True
        return SelectionResult(
            selected=selected,
            discarded=list(effective_candidates[max_k:]) + window_discarded,
            selection_reason=(
                f"critical requirement ({requirement.requirement_type.value} or "
                f"has_numeric_constraints={bool(requirement.constraints)}); "
                f"sending top {len(selected)} for thorough verification."
            ),
            selected_k=len(selected),
        )

    # Rule 3: STRONG top-1 with wide margin — single-candidate.
    if (
        top1.evidence_strength == EvidenceStrength.STRONG
        and margin >= config.selector_strong_margin
    ):
        top1.selected_for_llm = True
        return SelectionResult(
            selected=[top1],
            discarded=list(effective_candidates[1:]) + window_discarded,
            selection_reason=(
                f"top1 STRONG (score={top1.retrieval_score:.3f}) and margin "
                f"{margin:.3f} ≥ {config.selector_strong_margin:.2f}; one LLM "
                f"call is sufficient."
            ),
            selected_k=1,
        )

    # Rule 4-5: MEDIUM, narrow margin, or WEAK top-1 — broaden to 3.
    k = min(3, max_k)
    selected = effective_candidates[:k]
    for c in selected:
        c.selected_for_llm = True
    if top1.evidence_strength == EvidenceStrength.WEAK:
        reason = (
            f"top1 WEAK (score={top1.retrieval_score:.3f}); broadening to {k} "
            f"candidates so the LLM can disambiguate weak signal."
        )
    elif top1.evidence_strength == EvidenceStrength.STRONG:
        reason = (
            f"top1 STRONG but margin {margin:.3f} < "
            f"{config.selector_strong_margin:.2f}; broadening to {k} to break "
            f"the tie."
        )
    else:
        reason = (
            f"top1 MEDIUM (score={top1.retrieval_score:.3f}, margin={margin:.3f}); "
            f"sending {k} candidates."
        )
    return SelectionResult(
        selected=selected,
        discarded=list(effective_candidates[k:]) + window_discarded,
        selection_reason=reason,
        selected_k=k,
    )
