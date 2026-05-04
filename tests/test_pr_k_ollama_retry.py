"""
PR-K post-fix (E): retry on transient JSON-parse failures in OllamaCoverageJudge.

Design contract:
  * Empty response or un-parseable JSON → retry up to CQUALITY_JUDGE_RETRIES-1
    additional times with exponential backoff (1s, 2s, …).
  * Parse failure on the last attempt → LLM_UNAVAILABLE fallback (same as today).
  * Timeout / ConnectionError / HTTPError → immediate fallback, NO retry
    (retrying into an overloaded/unreachable server makes things worse).
  * Successful parse on any attempt → result returned, cache populated.
  * CQUALITY_JUDGE_RETRIES=1 → original no-retry behaviour.
  * Default CQUALITY_JUDGE_RETRIES=2 (1 original + 1 retry).

Thread-safety: each call is independent — retry state is stack-local.
"""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.ollama_coverage_judge import (
    OllamaCoverageJudge,
    _resolve_max_attempts,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _req(text: str = "Система должна хранить журнал не менее 90 дней.") -> RequirementUnit:
    return RequirementUnit(
        req_id="r-retry-1",
        source_document_id="doc-tz",
        text=text,
        normalized_text=text.lower(),
    )


def _unit(text: str = "Журнал хранится 90 дней в защищённом хранилище.") -> CoverageUnit:
    return CoverageUnit(
        unit_id="u-retry-1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        text=text,
        normalized_text=text.lower(),
    )


def _good_response_json(unit: CoverageUnit) -> str:
    """Build a valid LLM JSON response whose cited_phrase IS a substring of
    unit.text so the grounding gate does not demote the verdict."""
    # Take a 4-word slice from unit.text as the cited phrase.
    words = unit.text.split()
    cited = " ".join(words[:4]) if len(words) >= 4 else unit.text
    return json.dumps({
        "label": "COVERED",
        "confidence": 0.9,
        "cited_phrases": [cited],
        "matched_aspects": ["хранение журнала"],
        "missing_aspects": [],
        "conflict_aspects": [],
        "explanation": "Журнал хранится достаточно долго.",
    })


class _FakeResp:
    """Minimal requests.Response mimic."""
    def __init__(self, body: str, status: int = 200):
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(response=self)
            err.response = self
            raise err

    def json(self):
        return {"response": self._body}


# ── _resolve_max_attempts unit tests ─────────────────────────────────────────


class TestResolveMaxAttempts:
    @pytest.mark.parametrize("env_val, expected", [
        (None, 2),
        ("", 2),        # empty → default
        ("1", 1),
        ("2", 2),
        ("5", 5),
        ("0", 1),       # clamped to 1
        ("-3", 1),      # clamped to 1
        ("6", 5),       # clamped to cap
        ("abc", 2),     # non-int → default
    ])
    def test_resolve(self, monkeypatch, env_val, expected):
        if env_val is None:
            monkeypatch.delenv("CQUALITY_JUDGE_RETRIES", raising=False)
        else:
            monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", env_val)
        assert _resolve_max_attempts() == expected


# ── Retry on parse failures ───────────────────────────────────────────────────


class TestRetryOnParseFailure:
    """Parse failures (empty / garbled JSON) trigger retries; hard errors don't."""

    def test_bad_json_first_then_success(self, monkeypatch):
        """First call returns un-parseable text; second returns valid JSON.
        Result must be a real (non-fallback) judgment, unavailable_count=0."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "2")
        monkeypatch.setattr("time.sleep", lambda _: None)  # don't actually wait
        u = _unit()
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            n = call_n["n"]
            call_n["n"] += 1
            if n == 0:
                return _FakeResp("THIS IS NOT JSON AT ALL")
            return _FakeResp(_good_response_json(u))

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        result = judge.judge(_req(), u)

        assert isinstance(result, PairJudgment)
        assert result.llm_label == LLMLabel.COVERED
        assert judge.unavailable_count == 0
        assert call_n["n"] == 2  # two HTTP calls made

    def test_empty_response_then_success(self, monkeypatch):
        """Empty string response retried; second attempt valid."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "2")
        monkeypatch.setattr("time.sleep", lambda _: None)
        u = _unit()
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            n = call_n["n"]
            call_n["n"] += 1
            return _FakeResp("" if n == 0 else _good_response_json(u))

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        result = judge.judge(_req(), u)
        assert result.llm_label == LLMLabel.COVERED
        assert judge.unavailable_count == 0

    def test_all_parse_failures_exhaust_to_fallback(self, monkeypatch):
        """All attempts return bad JSON → fallback, unavailable_count=1."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
        monkeypatch.setattr("time.sleep", lambda _: None)
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            call_n["n"] += 1
            return _FakeResp("not json ever")

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        result = judge.judge(_req(), _unit())

        # Falls back — result is still a PairJudgment (DisabledCoverageJudge).
        assert isinstance(result, PairJudgment)
        assert judge.unavailable_count == 1
        assert "parse" in judge.last_error.lower() or "json" in judge.last_error.lower()
        assert call_n["n"] == 3  # all 3 attempts were made

    def test_retries_1_means_no_retry(self, monkeypatch):
        """CQUALITY_JUDGE_RETRIES=1 → original no-retry behaviour.
        Single bad parse → immediate fallback after 1 attempt."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "1")
        monkeypatch.setattr("time.sleep", lambda _: None)
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            call_n["n"] += 1
            return _FakeResp("garbage")

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        judge.judge(_req(), _unit())
        assert call_n["n"] == 1  # only 1 attempt


# ── Hard errors: no retry ─────────────────────────────────────────────────────


class TestNoRetryOnHardErrors:
    """Timeout / ConnectionError / HTTPError must break out immediately."""

    def test_timeout_no_retry(self, monkeypatch):
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
        monkeypatch.setattr("time.sleep", lambda _: None)
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            call_n["n"] += 1
            raise requests.Timeout("simulated timeout")

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b", timeout=1)
        judge.judge(_req(), _unit())
        assert call_n["n"] == 1
        assert judge.unavailable_count == 1
        assert "timeout" in judge.last_error.lower()

    def test_connection_error_no_retry(self, monkeypatch):
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
        monkeypatch.setattr("time.sleep", lambda _: None)
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            call_n["n"] += 1
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge()
        judge.judge(_req(), _unit())
        assert call_n["n"] == 1
        assert judge.unavailable_count == 1
        assert "ConnectionError" in judge.last_error

    def test_http_error_no_retry(self, monkeypatch):
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
        monkeypatch.setattr("time.sleep", lambda _: None)
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            call_n["n"] += 1
            return _FakeResp("Internal Server Error", status=500)

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge()
        judge.judge(_req(), _unit())
        assert call_n["n"] == 1
        assert judge.unavailable_count == 1
        assert "500" in judge.last_error or "HTTP" in judge.last_error


# ── Backoff timing ────────────────────────────────────────────────────────────


class TestRetryBackoff:
    def test_sleep_called_between_parse_retries(self, monkeypatch):
        """time.sleep must be called with a positive delay between retries."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
        sleep_calls: list[float] = []

        def _track_sleep(secs: float):
            sleep_calls.append(secs)

        monkeypatch.setattr("time.sleep", _track_sleep)
        u = _unit()
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            n = call_n["n"]
            call_n["n"] += 1
            if n < 2:
                return _FakeResp("not json")
            return _FakeResp(_good_response_json(u))

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        judge.judge(_req(), u)

        # 3 attempts → 2 retries → 2 sleep calls
        assert len(sleep_calls) == 2
        assert all(s > 0 for s in sleep_calls)
        # Second sleep >= first (exponential)
        assert sleep_calls[1] >= sleep_calls[0]

    def test_no_sleep_on_success_first_attempt(self, monkeypatch):
        """No sleep called when first attempt succeeds."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "2")
        sleep_calls: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))
        u = _unit()
        monkeypatch.setattr("requests.post", lambda *a, **k: _FakeResp(_good_response_json(u)))

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        judge.judge(_req(), u)
        assert sleep_calls == []


# ── Cache interaction ─────────────────────────────────────────────────────────


class TestCacheInteractionWithRetry:
    def test_cache_hit_skips_http_entirely(self, monkeypatch):
        """A cache hit must bypass the retry loop completely — no HTTP call."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
        http_calls = {"n": 0}

        def _should_not_be_called(*args, **kwargs):
            http_calls["n"] += 1
            return _FakeResp("not json")

        monkeypatch.setattr("requests.post", _should_not_be_called)

        r = _req()
        u = _unit()
        cached_judgment = PairJudgment(
            req_id=r.req_id, unit_id=u.unit_id,
            target_document_id=u.target_document_id,
            llm_label=LLMLabel.COVERED,
            rule_adjusted_label=LLMLabel.COVERED,
            llm_confidence=0.9,
        )

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        # Inject a fake cache that always returns the pre-built judgment.
        fake_cache = MagicMock()
        fake_cache.get.return_value = cached_judgment
        judge._cache = fake_cache

        result = judge.judge(r, u)
        assert result.llm_label == LLMLabel.COVERED
        assert http_calls["n"] == 0

    def test_successful_retry_populates_cache(self, monkeypatch):
        """When the first attempt fails but the second succeeds, the result
        must be written to the cache exactly once."""
        monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "2")
        monkeypatch.setattr("time.sleep", lambda _: None)
        u = _unit()
        call_n = {"n": 0}

        def _mock_post(*args, **kwargs):
            n = call_n["n"]
            call_n["n"] += 1
            return _FakeResp("bad" if n == 0 else _good_response_json(u))

        monkeypatch.setattr("requests.post", _mock_post)

        judge = OllamaCoverageJudge(model_name="qwen2.5:3b")
        fake_cache = MagicMock()
        fake_cache.get.return_value = None  # no hit
        judge._cache = fake_cache

        judge.judge(_req(), u)
        # put() must have been called exactly once (on success, not on retry)
        assert fake_cache.put.call_count == 1
