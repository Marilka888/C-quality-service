"""
Stage 5: run the LLM judge over the top-K shortlist for one requirement.

PR-K post-fix (A): per-pair judge calls run in parallel when
CQUALITY_JUDGE_CONCURRENCY > 1. Real-package symptom: on qwen2.5:3b
each LLM call is ~5-15s on a long prompt; for a 50-100-req TZ × top_k
fan-out that's 20-30 minutes serial. Ollama happily serves several
concurrent /api/generate calls on a single GPU (it queues them and
keeps the model resident), so a 2-3× speedup is free for the cost of
threading.

Default concurrency is 3. Override via env var, e.g.:
  CQUALITY_JUDGE_CONCURRENCY=1   # force serial (low-VRAM machines)
  CQUALITY_JUDGE_CONCURRENCY=5   # more aggressive parallel
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from app.core.logging import get_logger
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.llm.coverage_judge import CoverageJudge

logger = get_logger(__name__)

# Hard cap to prevent OOM / Ollama queue collapse if the env var is set
# to something silly. 8 is well above any single-GPU sweet spot.
_CONCURRENCY_HARD_CAP = 8


def _resolve_concurrency() -> int:
    """Read CQUALITY_JUDGE_CONCURRENCY, validate, clamp to [1, cap]."""
    raw = os.environ.get("CQUALITY_JUDGE_CONCURRENCY", "3").strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "CQUALITY_JUDGE_CONCURRENCY=%r is not an int; using 1", raw,
        )
        return 1
    if n < 1:
        return 1
    if n > _CONCURRENCY_HARD_CAP:
        logger.warning(
            "CQUALITY_JUDGE_CONCURRENCY=%d > hard cap %d; clamping",
            n, _CONCURRENCY_HARD_CAP,
        )
        return _CONCURRENCY_HARD_CAP
    return n


class PairJudgeService:
    def __init__(self, judge: CoverageJudge) -> None:
        self._judge = judge

    def judge_shortlist(
        self,
        requirement: RequirementUnit,
        shortlist: List[RetrievedCandidate],
        units_by_id: Dict[str, CoverageUnit],
    ) -> List[PairJudgment]:
        # Collect existing units once so we can dispatch to a batch judge
        # when the backend supports it (~10× faster for cross-encoders).
        pairs: List[CoverageUnit] = []
        for candidate in shortlist:
            unit = units_by_id.get(candidate.unit_id)
            if unit is None:
                logger.warning("unit_id=%s not found in index; skipping", candidate.unit_id)
                continue
            pairs.append(unit)

        if not pairs:
            return []

        # Backend that has its own batch path (e.g. CrossEncoderCoverageJudge
        # which does GPU-batched scoring) — let it do its thing. The numeric
        # speedup from per-pair parallelism is tiny next to GPU batching.
        judge_batch = getattr(self._judge, "judge_batch", None)
        if callable(judge_batch):
            try:
                return list(judge_batch(requirement, pairs))
            except Exception as exc:
                # If the batch path errors (e.g. network blip), fall back
                # to per-pair — we'd rather be slow than lose the request.
                logger.warning(
                    "judge.judge_batch failed (%s); falling back to per-pair", exc,
                )

        concurrency = _resolve_concurrency()
        # Single-pair shortlists or concurrency=1 → simple sequential path.
        if concurrency <= 1 or len(pairs) <= 1:
            return [self._judge.judge(requirement, u) for u in pairs]

        return self._judge_concurrent(requirement, pairs, concurrency)

    def _judge_concurrent(
        self,
        requirement: RequirementUnit,
        pairs: List[CoverageUnit],
        concurrency: int,
    ) -> List[PairJudgment]:
        """Fan-out the per-pair judge calls across `concurrency` worker
        threads. Order of returned judgments matches the order of `pairs`
        (not the order of completion) so downstream aggregation behaves
        identically to the sequential path.

        Errors in a single pair are logged and skipped — they would have
        bubbled up in the sequential path too (the OllamaCoverageJudge
        falls back to DisabledCoverageJudge internally on any HTTP error,
        so a None result here is unusual but possible if the underlying
        judge raises before catching). The pipeline already handles
        per-pair shortcomings via low_confidence flags downstream.
        """
        # Effective workers = min(concurrency, pair count) — no point
        # spinning more threads than we have work for.
        workers = min(concurrency, len(pairs))
        out: List[PairJudgment | None] = [None] * len(pairs)
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="pair-judge"
        ) as pool:
            future_to_idx = {
                pool.submit(self._judge.judge, requirement, unit): i
                for i, unit in enumerate(pairs)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    out[idx] = fut.result()
                except Exception as exc:
                    logger.warning(
                        "judge.judge raised on pair %d (req=%s unit=%s): %s",
                        idx, requirement.req_id[:8],
                        pairs[idx].unit_id[:8], exc,
                    )
                    # `out[idx]` stays None; filtered out below.
        return [j for j in out if j is not None]
