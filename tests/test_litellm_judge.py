"""
Tests for LiteLLMCoverageJudge.

Mocks `litellm.completion` so the test runs without network / API
keys and verifies:
  * Happy path: a parsed JSON verdict propagates through the
    grounding gate identically to the Ollama path.
  * Empty/malformed response → fallback to DisabledCoverageJudge,
    unavailability counter increments.
  * Rate-limit error retries with backoff, then succeeds on the
    second attempt.
  * Rate-limit error exhausting retries → fallback.
  * `consume_unavailability` returns and resets state (same contract
    as OllamaCoverageJudge so the pipeline's LLM_UNAVAILABLE warning
    works unchanged).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from app.domain.c_quality_enums import LLMLabel, RequirementType
from app.domain.c_quality_models import CoverageUnit, RequirementUnit
from app.infrastructure.llm.litellm_coverage_judge import LiteLLMCoverageJudge


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def req() -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="doc-tz",
        text="Система должна обеспечивать аутентификацию через Keycloak.",
        normalized_text="система должна обеспечивать аутентификацию через keycloak",
        requirement_type=RequirementType.SECURITY,
    )


@pytest.fixture
def unit() -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="Аутентификация через Keycloak реализована средствами OIDC.",
        normalized_text="аутентификация через keycloak реализована средствами oidc",
    )


def _fake_response(content: str):
    """Build an OpenAI-shape response object that LiteLLM normalises to."""
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


# ── Happy path ──────────────────────────────────────────────────────────


def test_judge_parses_valid_json_and_passes_grounding_gate(req, unit):
    raw = (
        '{"label": "COVERED", "confidence": 0.9, '
        '"matched_aspects": ["аутентификация"], '
        '"missing_aspects": [], "conflict_aspects": [], '
        '"cited_phrases": ["Аутентификация через Keycloak"], '
        '"explanation": "Покрыто."}'
    )
    judge = LiteLLMCoverageJudge(model_name="groq/llama-3.3-70b-versatile")

    with patch("litellm.completion", return_value=_fake_response(raw)) as m:
        out = judge.judge(req, unit)

    m.assert_called_once()
    call_kwargs = m.call_args.kwargs
    assert call_kwargs["model"] == "groq/llama-3.3-70b-versatile"
    # System + user messages built from the same prompts module.
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"

    assert out.llm_label == LLMLabel.COVERED
    assert out.low_confidence is False
    assert out.cited_phrases == ["Аутентификация через Keycloak"]


def test_judge_grounding_gate_demotes_ungrounded_verdict(req, unit):
    raw = (
        '{"label": "COVERED", "confidence": 0.95, '
        '"matched_aspects": ["x"], "missing_aspects": [], '
        '"conflict_aspects": [], '
        '"cited_phrases": ["биометрический сканер сетчатки"], '
        '"explanation": "x"}'
    )
    judge = LiteLLMCoverageJudge(model_name="gemini/gemini-2.0-flash")
    with patch("litellm.completion", return_value=_fake_response(raw)):
        out = judge.judge(req, unit)
    assert out.llm_label == LLMLabel.IRRELEVANT
    assert out.low_confidence is True
    assert "[ungrounded]" in out.explanation


# ── Failure modes ──────────────────────────────────────────────────────


def test_empty_response_falls_back_to_disabled(req, unit):
    judge = LiteLLMCoverageJudge(model_name="openai/gpt-4o-mini")
    with patch("litellm.completion", return_value=_fake_response("")):
        out = judge.judge(req, unit)
    # Disabled judge produces a real verdict (likely PARTIAL given matching tokens).
    assert out.llm_label in {LLMLabel.IRRELEVANT, LLMLabel.PARTIAL, LLMLabel.COVERED}
    count, err = judge.consume_unavailability()
    assert count == 1
    assert "Empty completion text" in err


def test_unparseable_json_falls_back_to_disabled(req, unit):
    judge = LiteLLMCoverageJudge(model_name="anthropic/claude-3-5-haiku-latest")
    with patch("litellm.completion", return_value=_fake_response("This is plain text, not JSON.")):
        out = judge.judge(req, unit)
    count, err = judge.consume_unavailability()
    assert count == 1
    assert "Could not parse JSON" in err
    assert out.llm_label in set(LLMLabel)


def test_litellm_import_failure_falls_back(req, unit):
    """If litellm isn't installed, judge gracefully degrades to
    DisabledCoverageJudge and records the unavailability."""
    judge = LiteLLMCoverageJudge(model_name="groq/llama-3.3-70b-versatile")
    # Hide the module to simulate ImportError on first access.
    with patch.dict(sys.modules, {"litellm": None}):
        out = judge.judge(req, unit)
    count, err = judge.consume_unavailability()
    assert count == 1
    assert "litellm not installed" in err
    assert out.llm_label in set(LLMLabel)


# ── Rate-limit retry ────────────────────────────────────────────────────


def test_rate_limit_retries_with_backoff(req, unit):
    """First call raises a rate-limit-shaped exception, second call succeeds."""
    judge = LiteLLMCoverageJudge(
        model_name="groq/llama-3.3-70b-versatile",
        max_retries=2, retry_backoff_seconds=0.01,  # near-instant for test
    )

    call_count = {"n": 0}
    raw_ok = (
        '{"label": "PARTIAL", "confidence": 0.7, '
        '"matched_aspects": ["a"], "missing_aspects": [], '
        '"conflict_aspects": [], '
        '"cited_phrases": ["Аутентификация через Keycloak"], '
        '"explanation": "x"}'
    )

    def _flaky(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("RateLimitError: 429 too many requests")
        return _fake_response(raw_ok)

    with patch("litellm.completion", side_effect=_flaky), \
         patch("time.sleep") as m_sleep:
        out = judge.judge(req, unit)

    assert call_count["n"] == 2
    assert m_sleep.call_count >= 1
    assert out.llm_label == LLMLabel.PARTIAL
    # No unavailability recorded — recovery succeeded.
    assert judge.consume_unavailability() == (0, "")


def test_rate_limit_exhausts_retries_then_falls_back(req, unit):
    judge = LiteLLMCoverageJudge(
        model_name="groq/llama-3.3-70b-versatile",
        max_retries=1, retry_backoff_seconds=0.01,
    )

    def _always_429(*args, **kwargs):
        raise RuntimeError("RateLimitError: 429 too many requests")

    with patch("litellm.completion", side_effect=_always_429), \
         patch("time.sleep"):
        judge.judge(req, unit)

    count, err = judge.consume_unavailability()
    assert count == 1
    assert "rate-limit retries exhausted" in err or "429" in err


# ── Telemetry contract ─────────────────────────────────────────────────


def test_consume_unavailability_resets_state(req, unit):
    judge = LiteLLMCoverageJudge(model_name="openai/gpt-4o-mini")
    with patch("litellm.completion", return_value=_fake_response("")):
        judge.judge(req, unit)
    assert judge.unavailable_count == 1
    n, e = judge.consume_unavailability()
    assert n == 1 and e
    # After consume: state cleared.
    assert judge.unavailable_count == 0
    assert judge.last_error == ""
    n2, e2 = judge.consume_unavailability()
    assert n2 == 0 and e2 == ""
