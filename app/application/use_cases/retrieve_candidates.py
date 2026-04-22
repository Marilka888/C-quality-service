"""
Stage 3 + 4: hybrid retrieval → top-K shortlist per RequirementUnit.

Score = w_lex * lexical + w_sem * semantic + w_con * constraint_overlap + w_sec * section_prior
All weights come from CoverageRetrievalConfig.
"""
from __future__ import annotations

from typing import Dict, List, Set

from app.core.config import CoverageRetrievalConfig
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


class CandidateRetriever:
    """Hybrid first-stage scoring + optional cross-encoder rerank.

    Pipeline:
        1. Score every unit with the hybrid formula
           (lex + semantic + constraint + section).
        2. Keep top-N (`top_k_before_rerank`) above `min_retrieval_score`.
        3. If a reranker is attached, rerank those N pairs jointly and
           keep the top-K. The final `retrieval_score` is overwritten
           with the reranker's score — downstream scoring logic keeps
           interpreting "higher = more relevant".
        4. Otherwise keep the hybrid top-K directly.
    """

    def __init__(
        self,
        config: CoverageRetrievalConfig,
        embedding_backend: EmbeddingBackend,
        reranker: Reranker | None = None,
    ) -> None:
        self._cfg = config
        self._emb = embedding_backend
        self._reranker: Reranker = reranker or NoopReranker()

    # ------------------------------------------------------------------

    def retrieve(
        self,
        requirement: RequirementUnit,
        coverage_units: List[CoverageUnit],
    ) -> List[RetrievedCandidate]:
        """Return top-K candidates above min_retrieval_score, sorted descending."""
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

        # --- Second stage: cross-encoder rerank of the top-N ----------
        #
        # The reranker sees (query, passage) pairs jointly and is much
        # better than the bi-encoder at distinguishing near-miss from
        # real coverage. Skipped when NoopReranker is in place so the
        # zero-dependency path stays fast.
        if not isinstance(self._reranker, NoopReranker):
            shortlist = results[: self._cfg.top_k_before_rerank]
            if shortlist:
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
                    shortlist.sort(key=lambda c: c.retrieval_score, reverse=True)
                    results = shortlist

        return results[: self._cfg.top_k]
