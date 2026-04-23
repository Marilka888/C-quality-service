"""
Zero-shot coverage judge backed by a multilingual cross-encoder.

Why this instead of a fine-tuned judge:
    We tried two rounds of fine-tuning a (req, unit) → match classifier on
    the pre-labelled `c_pairs_variant_a.csv`. Both rounds collapsed to
    "text is almost duplicate → MATCH, otherwise NO MATCH" — the positive
    signal in that CSV is fuzzy-matched duplicates, not coverage. The
    training objective never saw real "does this test cover that
    requirement?" examples.

    BGE-reranker-v2-m3, on the other hand, was trained on billions of
    MSMarco-style pair relevance judgments across 100+ languages. It
    already knows what "passage answers this query" looks like, including
    Russian, and handles paraphrases and partial coverage out of the box.
    Zero training required.

Contract:
    Produces a PairJudgment with the same schema as
    OllamaCoverageJudge / DisabledCoverageJudge so the pipeline stays
    oblivious to which backend scored the pair.

    Score → label mapping is configurable; defaults are calibrated for
    BGE-reranker-v2-m3 sigmoid output:
        p ≥ covered_threshold       → COVERED
        partial_threshold ≤ p < ct  → PARTIAL
        p < partial_threshold       → IRRELEVANT

    matched_aspects / conflict_aspects are NOT filled by the cross-encoder
    — the downstream rule verifier (PairVerifier) already fills those with
    numeric-conflict details, which is what we need. Leaving them empty
    keeps the post-validation logic in OllamaCoverageJudge (which downgrades
    CONFLICT-without-conflict_aspects to PARTIAL) working correctly when
    callers mix backends.
"""
from __future__ import annotations

import math
from typing import List

from app.core.logging import get_logger
from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.coverage_judge import CoverageJudge

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    # Clamp to avoid overflow for very large magnitudes (BGE logits can
    # reach ±15; sigmoid saturates cleanly within float range).
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderCoverageJudge(CoverageJudge):
    """Cross-encoder judge (default: BAAI/bge-reranker-v2-m3).

    Parameters
    ----------
    reranker :
        Any object with a `score(query, candidates: List[str]) -> List[float]`
        interface (the `Reranker` abstract class from
        `app.infrastructure.reranker.base`). Caller owns lifetime — the
        judge only calls `score`, doesn't load the model itself. This keeps
        the reranker and the judge sharing one in-memory copy of BGE.
    covered_threshold :
        Probability at or above which the pair is labelled COVERED.
    partial_threshold :
        Probability at or above which the pair is labelled PARTIAL (below
        COVERED). Below this → IRRELEVANT.
    use_sigmoid :
        BGE outputs raw logits. Set True to apply a sigmoid before
        thresholding (recommended — thresholds are easier to reason about
        on 0..1).
    """

    def __init__(
        self,
        reranker,
        covered_threshold: float = 0.8,
        partial_threshold: float = 0.3,
        use_sigmoid: bool = True,
    ) -> None:
        if covered_threshold < partial_threshold:
            raise ValueError(
                f"covered_threshold ({covered_threshold}) must be >= "
                f"partial_threshold ({partial_threshold})"
            )
        self._reranker = reranker
        self._covered = float(covered_threshold)
        self._partial = float(partial_threshold)
        self._use_sigmoid = bool(use_sigmoid)

    # ------------------------------------------------------------------

    def _score_to_label(self, p: float) -> LLMLabel:
        if p >= self._covered:
            return LLMLabel.COVERED
        if p >= self._partial:
            return LLMLabel.PARTIAL
        return LLMLabel.IRRELEVANT

    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        # Single-pair call — inefficient on GPU but keeps the interface
        # identical to the LLM judge. A future optimisation is batch_judge
        # which the pipeline could call once per shortlist.
        scores = self._reranker.score(req.text, [unit.text])
        raw = float(scores[0]) if scores else 0.0
        p = _sigmoid(raw) if self._use_sigmoid else raw
        label = self._score_to_label(p)
        explanation = (
            f"[cross-encoder] score={raw:+.3f} (p={p:.3f}) → {label.value}"
        )
        return PairJudgment(
            req_id=req.req_id,
            unit_id=unit.unit_id,
            target_document_id=unit.target_document_id,
            llm_label=label,
            llm_confidence=p,
            rule_adjusted_label=label,
            matched_aspects=[],
            missing_aspects=[],
            conflict_aspects=[],
            explanation=explanation,
        )

    # ------------------------------------------------------------------

    def judge_batch(
        self, req: RequirementUnit, units: List[CoverageUnit]
    ) -> List[PairJudgment]:
        """Score all (req, unit) pairs in one cross-encoder call.

        Pipeline orchestrator may prefer this over the per-pair `judge`
        because BGE is ~10× faster in batched mode.
        """
        if not units:
            return []
        scores = self._reranker.score(req.text, [u.text for u in units])
        out: List[PairJudgment] = []
        for unit, raw in zip(units, scores):
            p = _sigmoid(raw) if self._use_sigmoid else raw
            label = self._score_to_label(p)
            out.append(
                PairJudgment(
                    req_id=req.req_id,
                    unit_id=unit.unit_id,
                    target_document_id=unit.target_document_id,
                    llm_label=label,
                    llm_confidence=p,
                    rule_adjusted_label=label,
                    matched_aspects=[],
                    missing_aspects=[],
                    conflict_aspects=[],
                    explanation=f"[cross-encoder] score={raw:+.3f} (p={p:.3f}) → {label.value}",
                )
            )
        return out
