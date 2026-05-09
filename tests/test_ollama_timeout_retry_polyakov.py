"""
Polyakov-regression (2026-05-10): on the May-10 re-run, 57 of ~60 LLM
pairs hit `timeout after 120s`. Original code returned the
NOT_JUDGED sentinel immediately on first timeout — correct behaviour
(don't morph infra failure into MISSING) but it left a lot of pairs
unjudged when a second attempt would have succeeded.

Two fixes shipped together:
  1. Default Ollama timeout 120 → 240 s; CQUALITY_JUDGE_TIMEOUT env
     override.
  2. Timeouts now share the parse-retry budget — on the first
     timeout the loop continues to the next attempt (with backoff)
     instead of sentinelling out immediately.

If the SECOND attempt also times out (or any other terminal failure
hits and the budget is exhausted), the sentinel path still kicks in —
so the UNKNOWN-status guarantee from the prior commit is preserved.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import requests

from app.core.config import CoverageLLMConfig
from app.domain.c_quality_enums import CoverageUnitType, LLMLabel, RequirementType
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)
from app.infrastructure.llm.coverage_judge import is_unknown_judgment
from app.infrastructure.llm.ollama_coverage_judge import OllamaCoverageJudge


def _req() -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Система должна обеспечивать поиск.",
        normalized_text="система должна обеспечивать поиск.",
        requirement_type=RequirementType.FUNCTIONAL,
    )


def _unit() -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        unit_type=CoverageUnitType.PARAGRAPH,
        text="Система обеспечивает поиск.",
        normalized_text="система обеспечивает поиск.",
    )


def _ok_response_json(label: str = "PARTIAL") -> dict:
    return {
        "response": (
            '{"label": "' + label + '", "confidence": 0.7, '
            '"matched_aspects": [], "missing_aspects": [], '
            '"conflict_aspects": [], "cited_phrases": ["обеспечивает поиск"], '
            '"explanation": "test"}'
        )
    }


# ── Config defaults ────────────────────────────────────────────────


def test_config_default_timeout_is_240() -> None:
    """Polyakov-regression: default timeout bumped from 120 → 240 so
    qwen2.5:7b's long-prompt tail no longer routinely sentinels out."""
    cfg = CoverageLLMConfig()
    assert cfg.timeout == 240


def test_env_override_takes_effect(monkeypatch) -> None:
    """CQUALITY_JUDGE_TIMEOUT env var must override the constructor
    default — operators need this lever to tune at runtime without
    code changes."""
    monkeypatch.setenv("CQUALITY_JUDGE_TIMEOUT", "60")
    judge = OllamaCoverageJudge(model_name="test", timeout=240)
    assert judge._timeout == 60


def test_env_override_clamps_to_min(monkeypatch) -> None:
    """Defensive: comically-low env values get clamped to 30 s so a
    typo doesn't disable the timeout entirely."""
    monkeypatch.setenv("CQUALITY_JUDGE_TIMEOUT", "5")
    judge = OllamaCoverageJudge(model_name="test", timeout=240)
    assert judge._timeout == 30


def test_env_override_invalid_string_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("CQUALITY_JUDGE_TIMEOUT", "abc")
    judge = OllamaCoverageJudge(model_name="test", timeout=200)
    assert judge._timeout == 200


# ── Timeout retry within budget ────────────────────────────────────


def test_timeout_then_success_returns_real_judgment(monkeypatch) -> None:
    """Polyakov core case: first attempt times out (Ollama queue cold),
    second attempt succeeds. Result must be the real PARTIAL judgment,
    NOT the unknown sentinel — pair is judged, not abandoned."""
    monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "2")
    # Speed up the retry's backoff sleep so the test is fast.
    monkeypatch.setattr(
        "app.infrastructure.llm.ollama_coverage_judge.time.sleep",
        lambda s: None,
    )
    judge = OllamaCoverageJudge(model_name="test", timeout=240)

    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = _ok_response_json("PARTIAL")

    with patch(
        "app.infrastructure.llm.ollama_coverage_judge.requests.post",
        side_effect=[requests.Timeout("first attempt timed out"), fake_resp],
    ):
        result = judge.judge(_req(), _unit())

    assert isinstance(result, PairJudgment)
    assert not is_unknown_judgment(result), (
        "second attempt succeeded — must NOT be sentinel"
    )
    assert result.llm_label == LLMLabel.PARTIAL


def test_two_timeouts_exhaust_budget_returns_sentinel(monkeypatch) -> None:
    """If BOTH attempts time out, the budget is exhausted and the
    sentinel must fire. Status UNKNOWN is preserved."""
    monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "2")
    monkeypatch.setattr(
        "app.infrastructure.llm.ollama_coverage_judge.time.sleep",
        lambda s: None,
    )
    judge = OllamaCoverageJudge(model_name="test", timeout=240)

    with patch(
        "app.infrastructure.llm.ollama_coverage_judge.requests.post",
        side_effect=[
            requests.Timeout("first timeout"),
            requests.Timeout("second timeout"),
        ],
    ):
        result = judge.judge(_req(), _unit())

    assert is_unknown_judgment(result), (
        "exhausted budget must produce UNKNOWN sentinel, not real verdict"
    )
    # Reason tag should reference timeout, not parse failure.
    assert any(
        "timeout" in (a or "").lower() for a in result.verifier_actions
    ), result.verifier_actions


def test_timeout_then_parse_error_then_success(monkeypatch) -> None:
    """Mixed-failure path: budget=2 means 1 retry. Configure budget=3
    to exercise mixed timeout+parse-error+success. Verifies retries
    work irrespective of failure category."""
    monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
    monkeypatch.setattr(
        "app.infrastructure.llm.ollama_coverage_judge.time.sleep",
        lambda s: None,
    )
    judge = OllamaCoverageJudge(model_name="test", timeout=240)

    parse_fail_resp = MagicMock()
    parse_fail_resp.raise_for_status.return_value = None
    parse_fail_resp.json.return_value = {"response": "garbage not json"}

    ok_resp = MagicMock()
    ok_resp.raise_for_status.return_value = None
    ok_resp.json.return_value = _ok_response_json("COVERED")

    with patch(
        "app.infrastructure.llm.ollama_coverage_judge.requests.post",
        side_effect=[requests.Timeout("first"), parse_fail_resp, ok_resp],
    ):
        result = judge.judge(_req(), _unit())

    assert not is_unknown_judgment(result)
    assert result.llm_label == LLMLabel.COVERED


def test_connection_error_does_not_retry(monkeypatch) -> None:
    """ConnectionError is systemic (Ollama daemon down / network
    unreachable). Per design we DO NOT retry — sentinel out
    immediately. Old behaviour preserved."""
    monkeypatch.setenv("CQUALITY_JUDGE_RETRIES", "3")
    judge = OllamaCoverageJudge(model_name="test", timeout=240)

    with patch(
        "app.infrastructure.llm.ollama_coverage_judge.requests.post",
        side_effect=requests.ConnectionError("daemon refused"),
    ) as mocked:
        result = judge.judge(_req(), _unit())

    assert is_unknown_judgment(result)
    # Single attempt: no retry.
    assert mocked.call_count == 1, (
        f"connection errors must NOT retry; got {mocked.call_count} calls"
    )
