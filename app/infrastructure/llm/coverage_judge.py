"""Abstract interface for pair-level LLM judge."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit


class CoverageJudge(ABC):
    @abstractmethod
    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        """Return a PairJudgment for the (req, unit) pair."""


# Polyakov-regression (2026-05-10): runtime-failure sentinel.
#
# The LLM-judge wrappers (Ollama, LiteLLM) USED to fall back to a
# rule-based DisabledCoverageJudge whenever the backend errored at
# request time. That conflated two different failure modes:
#
#   1. User intentionally disabled LLM (config.llm.enabled == False)
#      — DisabledCoverageJudge is the sole judge; its verdicts are
#      authoritative.
#   2. LLM was supposed to answer but errored (timeout / connection /
#      HTTP / parse-exhausted / unexpected exception). Returning a
#      rule-based verdict here means an infrastructure failure shows
#      up as a documentation defect on the package report, inflating
#      criticalCount and turning a CRITICAL package status on a
#      perfectly good submission whose reviewer just had Ollama timeout.
#
# `make_unknown_judgment(req, unit, reason)` is the runtime-failure
# sentinel: it carries the special LLMLabel.NOT_JUDGED label and a
# verifier_actions tag the aggregator looks for to surface the pair
# as CoverageStatus.UNKNOWN — distinct from MISSING, excluded from
# criticalCount, excluded from C-grade denominator.
_UNKNOWN_REASON_TAG = "llm_unavailable"


def make_unknown_judgment(
    req: RequirementUnit,
    unit: CoverageUnit,
    reason: str,
) -> PairJudgment:
    """Build a sentinel PairJudgment marking that the LLM judge could
    not produce a verdict for this pair. The aggregator detects this
    sentinel via `verifier_actions` (entry starting with
    `llm_unavailable:`) AND `llm_label == LLMLabel.NOT_JUDGED`, and
    surfaces the row as CoverageStatus.UNKNOWN."""
    return PairJudgment(
        req_id=req.req_id,
        unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        llm_label=LLMLabel.NOT_JUDGED,
        rule_adjusted_label=LLMLabel.NOT_JUDGED,
        llm_confidence=0.0,
        explanation=(
            f"LLM judge unavailable: {reason}. Pair not judged — "
            f"reported as UNKNOWN, excluded from criticalCount and "
            f"C-grade denominator."
        ),
        verifier_actions=[f"{_UNKNOWN_REASON_TAG}:{reason}"],
    )


def is_unknown_judgment(judgment: PairJudgment) -> bool:
    """True when this judgment is the runtime-failure sentinel from
    `make_unknown_judgment`. Aggregator uses this to detect rows that
    must be surfaced as UNKNOWN rather than MISSING."""
    if judgment.llm_label != LLMLabel.NOT_JUDGED:
        return False
    return any(
        (a or "").startswith(f"{_UNKNOWN_REASON_TAG}:")
        for a in (judgment.verifier_actions or [])
    )
