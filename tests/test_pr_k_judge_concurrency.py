"""
PR-K post-fix (A): tests for parallel per-pair judge calls in
PairJudgeService.

Concurrency is controlled by CQUALITY_JUDGE_CONCURRENCY env var:
  1 (default) — sequential, identical to pre-A behaviour
  N > 1       — fan-out across N worker threads
  N <= 0      — clamped to 1
  > cap       — clamped to _CONCURRENCY_HARD_CAP

Contract verified:
  * concurrency=1 → behaviour identical to before
  * concurrency=3 → all pairs judged, output order matches input order
  * order is preserved even when judges complete in arbitrary order
    (slow / fast / staggered)
  * one judge raising does not poison the others
  * judge_batch path (cross_encoder) takes precedence over the
    concurrent fan-out — that's a GPU-batched fast path
  * pair count < concurrency → workers spun = pair count
  * concurrency env var is parsed: empty / non-int / negative → 1
  * concurrency above hard cap → clamped
"""
from __future__ import annotations

import threading
import time

import pytest

from app.application.use_cases.judge_pairs import (
    PairJudgeService,
    _CONCURRENCY_HARD_CAP,
    _resolve_concurrency,
)
from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.llm.coverage_judge import CoverageJudge


# ── Fixtures ─────────────────────────────────────────────────────────────


def _req(req_id: str = "r1", text: str = "Система должна работать.") -> RequirementUnit:
    return RequirementUnit(
        req_id=req_id, source_document_id="doc-tz",
        text=text, normalized_text=text.lower(),
    )


def _unit(unit_id: str, text: str = "Работает.") -> CoverageUnit:
    return CoverageUnit(
        unit_id=unit_id, target_document_id="doc-pmi", target_doc_role="pmi",
        text=text, normalized_text=text.lower(),
    )


def _candidate(unit_id: str, score: float = 0.5) -> RetrievedCandidate:
    return RetrievedCandidate(
        req_id="r1", unit_id=unit_id, target_document_id="doc-pmi",
        retrieval_score=score,
    )


class _RecordingJudge(CoverageJudge):
    """Mock judge that records (req, unit) calls and the calling thread.
    Returns a deterministic PairJudgment whose label encodes the unit_id
    so output ordering can be verified."""

    def __init__(self, delay_per_unit: dict[str, float] | None = None,
                 raise_on: set[str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.threads: set[str] = set()
        self.lock = threading.Lock()
        self.delay_per_unit = delay_per_unit or {}
        self.raise_on = raise_on or set()

    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        with self.lock:
            self.calls.append((req.req_id, unit.unit_id))
            self.threads.add(threading.current_thread().name)
        delay = self.delay_per_unit.get(unit.unit_id, 0.0)
        if delay:
            time.sleep(delay)
        if unit.unit_id in self.raise_on:
            raise RuntimeError(f"simulated failure on {unit.unit_id}")
        # Encode unit_id into explanation so the test can verify ordering.
        return PairJudgment(
            req_id=req.req_id, unit_id=unit.unit_id,
            target_document_id=unit.target_document_id,
            llm_label=LLMLabel.IRRELEVANT, rule_adjusted_label=LLMLabel.IRRELEVANT,
            llm_confidence=0.5,
            explanation=f"judged-{unit.unit_id}",
        )


def _build_service_with_pairs(judge: CoverageJudge, n: int):
    units = {f"u{i}": _unit(f"u{i}") for i in range(n)}
    shortlist = [_candidate(f"u{i}") for i in range(n)]
    return PairJudgeService(judge), shortlist, units


# ── _resolve_concurrency unit tests ─────────────────────────────────────


class TestResolveConcurrency:
    @pytest.mark.parametrize("env_val, expected", [
        (None, 1),
        ("", 1),
        ("1", 1),
        ("3", 3),
        ("8", 8),
        # Negative / zero → clamped to 1
        ("0", 1),
        ("-1", 1),
        # Above hard cap → clamped
        (str(_CONCURRENCY_HARD_CAP + 1), _CONCURRENCY_HARD_CAP),
        ("100", _CONCURRENCY_HARD_CAP),
        # Non-int → fallback 1
        ("abc", 1),
        ("3.14", 1),
    ])
    def test_resolve(self, monkeypatch, env_val, expected):
        if env_val is None:
            monkeypatch.delenv("CQUALITY_JUDGE_CONCURRENCY", raising=False)
        else:
            monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", env_val)
        assert _resolve_concurrency() == expected


# ── Concurrency = 1 → sequential equivalence ────────────────────────────


class TestSequentialEquivalence:
    def test_concurrency_1_uses_main_thread(self, monkeypatch):
        monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", "1")
        judge = _RecordingJudge()
        service, shortlist, units = _build_service_with_pairs(judge, 3)
        out = service.judge_shortlist(_req(), shortlist, units)
        assert len(out) == 3
        # Sequential path runs in the calling thread only.
        assert "MainThread" in judge.threads or len(judge.threads) == 1
        # No "pair-judge" workers spun up.
        assert not any(t.startswith("pair-judge") for t in judge.threads)

    def test_concurrency_unset_is_sequential(self, monkeypatch):
        monkeypatch.delenv("CQUALITY_JUDGE_CONCURRENCY", raising=False)
        judge = _RecordingJudge()
        service, shortlist, units = _build_service_with_pairs(judge, 3)
        out = service.judge_shortlist(_req(), shortlist, units)
        assert len(out) == 3
        assert not any(t.startswith("pair-judge") for t in judge.threads)


# ── Concurrency > 1 → fan-out + order-independence ──────────────────────


class TestConcurrentFanOut:
    def test_concurrency_3_spawns_workers(self, monkeypatch):
        monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", "3")
        judge = _RecordingJudge()
        service, shortlist, units = _build_service_with_pairs(judge, 5)
        out = service.judge_shortlist(_req(), shortlist, units)
        assert len(out) == 5
        # At least one worker thread participated.
        worker_threads = {t for t in judge.threads if t.startswith("pair-judge")}
        assert worker_threads, f"no worker threads spawned; saw: {judge.threads}"

    def test_concurrent_output_preserves_input_order(self, monkeypatch):
        """Input order: u0, u1, u2, u3, u4. Even though u0 takes longest,
        output index 0 must be u0's judgment."""
        monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", "5")
        judge = _RecordingJudge(delay_per_unit={
            "u0": 0.10,  # slowest
            "u1": 0.04,
            "u2": 0.01,  # fastest, will complete first
            "u3": 0.06,
            "u4": 0.02,
        })
        service, shortlist, units = _build_service_with_pairs(judge, 5)
        out = service.judge_shortlist(_req(), shortlist, units)
        assert len(out) == 5
        # Output[i] must correspond to input pair[i] regardless of completion order.
        for i in range(5):
            assert out[i].unit_id == f"u{i}", (
                f"order broken at idx {i}: expected u{i}, got {out[i].unit_id}"
            )

    def test_workers_clamped_to_pair_count(self, monkeypatch):
        """Concurrency=8 but only 2 pairs → at most 2 workers spawned."""
        monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", "8")
        judge = _RecordingJudge(delay_per_unit={"u0": 0.05, "u1": 0.05})
        service, shortlist, units = _build_service_with_pairs(judge, 2)
        out = service.judge_shortlist(_req(), shortlist, units)
        assert len(out) == 2
        worker_threads = {t for t in judge.threads if t.startswith("pair-judge")}
        # 2 pairs → at most 2 workers (could be 1 if one finishes before the other starts).
        assert len(worker_threads) <= 2

    def test_single_pair_uses_sequential_even_with_high_concurrency(
        self, monkeypatch,
    ):
        """One pair shortlist → no point spawning workers; sequential path."""
        monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", "5")
        judge = _RecordingJudge()
        service, shortlist, units = _build_service_with_pairs(judge, 1)
        out = service.judge_shortlist(_req(), shortlist, units)
        assert len(out) == 1
        worker_threads = {t for t in judge.threads if t.startswith("pair-judge")}
        assert not worker_threads, (
            f"single-pair shortlist spawned workers: {worker_threads}"
        )


# ── Exception isolation ─────────────────────────────────────────────────


class TestExceptionIsolation:
    def test_one_judge_failure_does_not_break_others(self, monkeypatch):
        """If pair u2 raises, u0/u1/u3/u4 must still be returned."""
        monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", "5")
        judge = _RecordingJudge(raise_on={"u2"})
        service, shortlist, units = _build_service_with_pairs(judge, 5)
        out = service.judge_shortlist(_req(), shortlist, units)
        # 4 successful judgments returned (u2 dropped).
        assert len(out) == 4
        unit_ids = [j.unit_id for j in out]
        assert "u2" not in unit_ids
        # And the order of survivors is still u0, u1, u3, u4.
        assert unit_ids == ["u0", "u1", "u3", "u4"]


# ── judge_batch precedence (cross_encoder) ──────────────────────────────


class _BatchJudge(CoverageJudge):
    """Mock cross-encoder-style judge with a custom judge_batch."""
    def __init__(self):
        self.judge_calls = 0
        self.judge_batch_calls = 0

    def judge(self, req, unit):
        self.judge_calls += 1
        return PairJudgment(
            req_id=req.req_id, unit_id=unit.unit_id,
            target_document_id=unit.target_document_id,
            llm_label=LLMLabel.IRRELEVANT, rule_adjusted_label=LLMLabel.IRRELEVANT,
        )

    def judge_batch(self, req, units):
        self.judge_batch_calls += 1
        return [
            PairJudgment(
                req_id=req.req_id, unit_id=u.unit_id,
                target_document_id=u.target_document_id,
                llm_label=LLMLabel.IRRELEVANT, rule_adjusted_label=LLMLabel.IRRELEVANT,
                explanation=f"batched-{u.unit_id}",
            )
            for u in units
        ]


class TestJudgeBatchPrecedence:
    def test_judge_batch_takes_precedence_over_concurrent(self, monkeypatch):
        """When the judge has judge_batch, concurrent fan-out is NOT
        used — judge_batch is faster (GPU-batched)."""
        monkeypatch.setenv("CQUALITY_JUDGE_CONCURRENCY", "5")
        judge = _BatchJudge()
        service, shortlist, units = _build_service_with_pairs(judge, 5)
        out = service.judge_shortlist(_req(), shortlist, units)
        assert len(out) == 5
        # Per-pair judge() must NOT have been called when judge_batch exists.
        assert judge.judge_calls == 0
        assert judge.judge_batch_calls == 1
        # All judgments came from the batch path.
        for j in out:
            assert "batched-" in j.explanation
