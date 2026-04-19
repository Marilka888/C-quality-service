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
      1.0 per constraint pair with matching value AND unit class  (exact match)
      0.5 per constraint pair with matching unit class only       (same topic, different value)
    Normalized by number of req constraints.
    """
    if not req_constraints:
        return 0.0
    if not unit_constraints:
        return 0.0

    from app.application.use_cases.verify_pairs import _same_unit_class

    total = 0.0
    for rc in req_constraints:
        for uc in unit_constraints:
            if _same_unit_class(rc.unit, uc.unit):
                if abs(rc.value - uc.value) < 1e-6:
                    total += 1.0  # exact match
                else:
                    total += 0.5  # same quantity, different value (possible conflict)
    return min(total / len(req_constraints), 1.0)


def _section_prior(req: RequirementUnit, unit: CoverageUnit) -> float:
    role = unit.target_doc_role.lower()
    if role == "pmi" and req.requirement_type in _PMI_PREFERRED:
        return 1.0
    if role == "pz" and req.requirement_type in _PZ_PREFERRED:
        return 1.0
    return 0.0


class CandidateRetriever:
    """Computes retrieval scores and returns top-K candidates per requirement."""

    def __init__(
        self,
        config: CoverageRetrievalConfig,
        embedding_backend: EmbeddingBackend,
    ) -> None:
        self._cfg = config
        self._emb = embedding_backend

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
        return results[: self._cfg.top_k]
