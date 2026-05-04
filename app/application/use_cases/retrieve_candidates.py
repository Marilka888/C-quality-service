"""
Stage 3 + 4: hybrid retrieval → top-N shortlist per RequirementUnit.

Score = w_lex * lexical + w_sem * semantic + w_con * constraint_overlap + w_sec * section_prior
All weights come from CoverageRetrievalConfig.

PR-K additions (additive, no contract breaks):
  * Each RetrievedCandidate carries `score_reason` (one-liner explaining
    which signal drove the score) and `evidence_strength` (STRONG /
    MEDIUM / WEAK / NO_EVIDENCE bin).
  * `initial_top_n` controls how many candidates are returned to the
    caller; the AdaptiveCandidateSelector trims further before LLM.
  * Reranker can run unconditionally (mode="always", legacy) or only
    when first-stage signals are weak (mode="conditional"): top1 below
    a threshold, narrow top1-top2 margin, requirement carries numeric
    constraints, or paraphrase indicated by high semantic / low lexical.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from app.application.use_cases.applicability import evidence_strength_from_score
from app.core.config import CoverageRerankerConfig, CoverageRetrievalConfig
from app.core.logging import get_logger
from app.core.text import tokenize_content
from app.domain.c_quality_enums import RequirementType
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.embeddings.base import EmbeddingBackend
from app.infrastructure.reranker.base import NoopReranker, Reranker

logger = get_logger(__name__)

# RequirementTypes that get a section_prior bonus on PMI / PZ docs
_PMI_PREFERRED: Set[RequirementType] = {
    RequirementType.PERFORMANCE,
    RequirementType.LOGGING,
}
_PZ_PREFERRED: Set[RequirementType] = {
    RequirementType.FUNCTIONAL,
    RequirementType.SECURITY,
    RequirementType.INTERFACE,
    RequirementType.STORAGE,
    RequirementType.ARCHITECTURE_IMPLEMENTATION,
    RequirementType.RELIABILITY,
}

# Critical types that always justify the more expensive reranker
# (in conditional mode). Mirrors `_is_critical` in the adaptive selector.
_CRITICAL_TYPES: Set[RequirementType] = {
    RequirementType.SECURITY,
    RequirementType.PERFORMANCE,
    RequirementType.RELIABILITY,
}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _constraint_overlap(req_constraints: List[Constraint], unit_constraints: List[Constraint]) -> float:
    """
    Score in [0, 1]:
      1.0 per constraint pair with matching unit class AND matching value
          (comparing canonical values — "90 дней" matches "7776000 секунд"
           if we ever saw such a conversion)
      0.5 per constraint pair with matching unit class only
          (same topic, different value → possible conflict, keep as retrieval signal)
    Normalized by number of req constraints.
    """
    if not req_constraints:
        return 0.0
    if not unit_constraints:
        return 0.0

    from app.application.use_cases.verify_pairs import (
        _canonical_value,
        _same_unit_class,
    )

    total = 0.0
    for rc in req_constraints:
        for uc in unit_constraints:
            if not _same_unit_class(rc.unit, uc.unit):
                continue
            # Compare on canonical scale when both units are convertible;
            # this makes the retriever treat "2 сек" and "2000 мс" as an
            # exact match instead of a possible-conflict.
            rc_canon = _canonical_value(rc.value, rc.unit)
            uc_canon = _canonical_value(uc.value, uc.unit)
            if rc_canon is not None and uc_canon is not None:
                tol = max(abs(rc_canon), abs(uc_canon)) * 1e-3
                if abs(rc_canon - uc_canon) <= max(tol, 1e-6):
                    total += 1.0
                else:
                    total += 0.5
            else:
                if abs(rc.value - uc.value) < 1e-6:
                    total += 1.0
                else:
                    total += 0.5
    return min(total / len(req_constraints), 1.0)


def _section_prior(req: RequirementUnit, unit: CoverageUnit) -> float:
    role = unit.target_doc_role.lower()
    if role == "pmi" and req.requirement_type in _PMI_PREFERRED:
        return 1.0
    if role == "pz" and req.requirement_type in _PZ_PREFERRED:
        return 1.0
    return 0.0


def _build_score_reason(
    lex: float, sem: float, con: float, sec: float, total: float,
) -> str:
    """One-line, human-readable explanation of which component drove
    the score. Rendered in evidence_trace and also handy for log
    inspection. Pure formatting — no thresholds outside what the
    reader of a score-breakdown intuitively expects."""
    parts: List[str] = []
    # Identify dominant component(s). Treat anything within 70% of the
    # max as "co-leading".
    components = {
        "lex": lex,
        "sem": sem,
        "con": con,
        "sec": sec,
    }
    max_v = max(components.values()) if components else 0.0
    if max_v <= 0.0:
        return f"all signals near zero (score={total:.2f})"
    leaders = [k for k, v in components.items() if v >= 0.7 * max_v and v > 0.0]

    label = {
        "lex": "lexical",
        "sem": "semantic",
        "con": "constraint",
        "sec": "section",
    }
    if len(leaders) == 1:
        parts.append(f"{label[leaders[0]]} dominant")
    else:
        parts.append("+".join(label[k] for k in leaders) + " co-leading")

    parts.append(
        f"lex={lex:.2f} sem={sem:.2f} con={con:.2f} sec={sec:.2f} ⇒ {total:.2f}"
    )
    return " | ".join(parts)


def _conditional_should_rerank(
    requirement: RequirementUnit,
    sorted_candidates: List[RetrievedCandidate],
    rr_cfg: CoverageRerankerConfig,
) -> tuple[bool, str]:
    """Return (should_rerank, reason). Pure decision over the first-stage
    shortlist. The thresholds live in CoverageRerankerConfig so they
    can be retuned without code changes.

    Rules (any one fires):
      * Top-1 score below `conditional_top1_threshold` — first stage is weak.
      * Top-1 minus top-2 below `conditional_min_margin` — close call.
      * Requirement is critical (SECURITY / PERFORMANCE / RELIABILITY).
      * Requirement carries numeric constraints — verify mismatch risk.
      * Top-1 has high semantic but low lexical (paraphrase) — bi-encoder
        risks a false positive without cross-encoder confirmation.
    """
    if not sorted_candidates:
        return False, "empty shortlist"
    top1 = sorted_candidates[0]
    top2 = sorted_candidates[1] if len(sorted_candidates) > 1 else None
    margin = (top1.retrieval_score - top2.retrieval_score) if top2 else top1.retrieval_score

    if top1.retrieval_score < rr_cfg.conditional_top1_threshold:
        return True, (
            f"top1 score {top1.retrieval_score:.3f} < "
            f"{rr_cfg.conditional_top1_threshold:.2f}"
        )
    if top2 is not None and margin < rr_cfg.conditional_min_margin:
        return True, (
            f"top1-top2 margin {margin:.3f} < "
            f"{rr_cfg.conditional_min_margin:.2f}"
        )
    if requirement.requirement_type in _CRITICAL_TYPES:
        return True, (
            f"critical type {requirement.requirement_type.value}"
        )
    if requirement.constraints:
        return True, (
            f"requirement carries {len(requirement.constraints)} numeric constraint(s)"
        )
    # Paraphrase signal: high semantic, low lexical on top-1.
    if top1.semantic_score >= 0.55 and top1.lexical_score <= 0.20:
        return True, (
            f"top1 paraphrase-like (sem={top1.semantic_score:.2f}, "
            f"lex={top1.lexical_score:.2f})"
        )
    return False, "first-stage signals strong; reranker skipped"


class CandidateRetriever:
    """Hybrid first-stage scoring + optional cross-encoder rerank.

    Pipeline:
        1. Score every unit with the hybrid formula
           (lex + semantic + constraint + section).
        2. Keep top-N (`top_k_before_rerank`) above `min_retrieval_score`.
        3. Reranker decision:
             - `mode == "always"` → run on the top-N, overwrite scores.
             - `mode == "conditional"` → run only when first-stage signals
               are weak (see `_conditional_should_rerank`).
             - `enabled == False` → skip.
        4. Fill `score_reason` and `evidence_strength` on every
           returned candidate, then trim to `initial_top_n` (PR-K) or
           `top_k` (legacy fallback).
    """

    def __init__(
        self,
        config: CoverageRetrievalConfig,
        embedding_backend: EmbeddingBackend,
        reranker: Reranker | None = None,
        reranker_config: Optional[CoverageRerankerConfig] = None,
    ) -> None:
        self._cfg = config
        self._emb = embedding_backend
        self._reranker: Reranker = reranker or NoopReranker()
        # `reranker_config` is optional so existing callers keep working.
        # When omitted we default to "always" (legacy behaviour) so a
        # wired reranker is exercised on every shortlist as before.
        self._rr_cfg: CoverageRerankerConfig = (
            reranker_config or CoverageRerankerConfig(mode="always")
        )

    # ------------------------------------------------------------------

    def retrieve(
        self,
        requirement: RequirementUnit,
        coverage_units: List[CoverageUnit],
    ) -> List[RetrievedCandidate]:
        """Return up to `initial_top_n` candidates above
        `min_retrieval_score`, sorted descending. Falls back to `top_k`
        when `initial_top_n` is unset (older configs)."""
        if not coverage_units:
            return []

        req_tokens: Set[str] = tokenize_content(requirement.normalized_text)
        candidate_texts = [u.normalized_text for u in coverage_units]

        semantic_scores = self._emb.similarity(requirement.normalized_text, candidate_texts)

        results: List[RetrievedCandidate] = []
        for i, unit in enumerate(coverage_units):
            unit_tokens: Set[str] = tokenize_content(unit.normalized_text)

            lex = _jaccard(req_tokens, unit_tokens)
            sem = float(semantic_scores[i]) if i < len(semantic_scores) else 0.0
            con = _constraint_overlap(requirement.constraints, unit.constraints)
            sec = _section_prior(requirement, unit)

            score = (
                self._cfg.lexical_weight * lex
                + self._cfg.semantic_weight * sem
                + self._cfg.constraint_weight * con
                + self._cfg.section_prior_weight * sec
            )

            if score < self._cfg.min_retrieval_score:
                continue

            results.append(
                RetrievedCandidate(
                    req_id=requirement.req_id,
                    unit_id=unit.unit_id,
                    target_document_id=unit.target_document_id,
                    lexical_score=round(lex, 4),
                    semantic_score=round(sem, 4),
                    constraint_overlap_score=round(con, 4),
                    section_prior_score=round(sec, 4),
                    retrieval_score=round(score, 4),
                )
            )

        results.sort(key=lambda c: c.retrieval_score, reverse=True)

        # --- Second stage: cross-encoder rerank of the top-N -----------
        #
        # PR-K change: the decision to rerank is now driven by
        # CoverageRerankerConfig.mode. "always" preserves legacy
        # behaviour; "conditional" runs only when first-stage signals
        # are weak.
        if not isinstance(self._reranker, NoopReranker):
            shortlist = results[: self._cfg.top_k_before_rerank]
            mode = (self._rr_cfg.mode or "always").lower()
            should_rerank = True
            rerank_reason = "mode=always"
            if mode == "conditional":
                should_rerank, rerank_reason = _conditional_should_rerank(
                    requirement, shortlist, self._rr_cfg,
                )

            if shortlist and should_rerank:
                unit_text_by_id = {u.unit_id: u.normalized_text for u in coverage_units}
                texts = [unit_text_by_id.get(c.unit_id, "") for c in shortlist]
                try:
                    rr_scores = self._reranker.score(requirement.normalized_text, texts)
                except Exception as exc:
                    logger.warning(
                        "Reranker failed (%s); falling back to hybrid order", exc,
                    )
                    rr_scores = None
                if rr_scores is not None and len(rr_scores) == len(shortlist):
                    # Overwrite retrieval_score with rerank score so the
                    # rest of the pipeline (and the final report) reflects
                    # what actually determined the top-K order.
                    for c, s in zip(shortlist, rr_scores):
                        c.retrieval_score = round(float(s), 4)
                        c.reranker_used = True
                        c.reranker_score = round(float(s), 4)
                    shortlist.sort(key=lambda c: c.retrieval_score, reverse=True)
                    results = shortlist
            elif shortlist and not should_rerank:
                logger.debug(
                    "Reranker skipped for req=%s: %s",
                    requirement.req_id[:12], rerank_reason,
                )

        # PR-K: populate explainability fields. Done AFTER any reranking
        # so the score_reason reflects the final retrieval_score and the
        # evidence_strength binning matches what downstream sees.
        strong = self._cfg.evidence_strength_strong_threshold
        medium = self._cfg.evidence_strength_medium_threshold
        weak = self._cfg.evidence_strength_weak_threshold
        for c in results:
            if c.reranker_used:
                # When the cross-encoder rewrote the score, the original
                # component breakdown is no longer the determinant — flag
                # that explicitly.
                c.score_reason = (
                    f"reranker score {c.retrieval_score:.2f} "
                    f"(first-stage lex={c.lexical_score:.2f} "
                    f"sem={c.semantic_score:.2f} con={c.constraint_overlap_score:.2f})"
                )
            else:
                c.score_reason = _build_score_reason(
                    c.lexical_score, c.semantic_score,
                    c.constraint_overlap_score, c.section_prior_score,
                    c.retrieval_score,
                )
            c.evidence_strength = evidence_strength_from_score(
                c.retrieval_score, strong=strong, medium=medium, weak=weak,
            )

        # PR-K: return up to initial_top_n. Falls back to top_k when the
        # config predates PR-K (e.g. unit tests that explicitly set top_k
        # to a small value and never touch initial_top_n).
        cap = max(
            getattr(self._cfg, "initial_top_n", 0) or 0,
            self._cfg.top_k,
        )
        if cap <= 0:
            cap = self._cfg.top_k
        return results[:cap]
