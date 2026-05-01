"""
LiteLLM-backed coverage judge.

A single judge class that routes through 100+ providers (OpenAI, Anthropic,
Groq, Gemini, Mistral, Cerebras, Together, Ollama, …) via the LiteLLM
unified API. Uses the SAME `build_judge_prompt(req, unit)` and the SAME
`_parse_response(...)` as `OllamaCoverageJudge`, so the wire format —
including the BUG-3 grounding gate — works identically across backends.

Configuration:
  config.llm.backend     = "litellm"
  config.llm.model_name  = "groq/llama-3.3-70b-versatile"
                          | "gemini/gemini-2.0-flash"
                          | "openai/gpt-4o-mini"
                          | "anthropic/claude-3-5-haiku-latest"
                          | "cerebras/llama-3.1-70b"
                          | "ollama/qwen2.5:7b"
                          | …any other LiteLLM-supported model name

API keys:
  GROQ_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), OPENAI_API_KEY,
  ANTHROPIC_API_KEY, CEREBRAS_API_KEY, MISTRAL_API_KEY, …
  Set via environment. LiteLLM picks up the right one for each provider
  prefix automatically.

Why a separate class (not replacing OllamaCoverageJudge):
  * `OllamaCoverageJudge` is a thin requests-based path; replacing it
    introduces a heavy new dependency for legacy single-provider deploys.
  * Here we keep both — operators choose Ollama-only when they want a
    minimal local stack, and LiteLLM when they need cross-provider
    calibration / production with cloud models.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from app.core.logging import get_logger
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.coverage_judge import CoverageJudge
from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
from app.infrastructure.llm.ollama_coverage_judge import _extract_json, _parse_response
from app.infrastructure.llm.prompts import build_judge_prompt

logger = get_logger(__name__)
_FALLBACK = DisabledCoverageJudge()


class LiteLLMCoverageJudge(CoverageJudge):
    """Multi-provider coverage judge backed by LiteLLM.

    Behaviour mirrors OllamaCoverageJudge:
      * Build the same system + user prompt.
      * Parse the JSON response through the shared `_parse_response`,
        which applies the BUG-3 grounding gate (cited_phrases must be
        substrings of the evidence text).
      * Track silent unavailability so the pipeline can surface
        LLM_UNAVAILABLE warnings to the user.

    Differences:
      * Uses `litellm.completion(model=..., messages=[...])` so any
        provider supported by LiteLLM works through the same call.
      * Passes a temperature of 0.1 (matches Ollama path) for
        deterministic-ish output.
      * Uses `max_tokens` instead of Ollama's `num_predict` (LiteLLM
        translates per-provider).
      * Sleeps on `RateLimitError` and retries up to `max_retries` times
        before giving up — needed for free-tier providers (Groq,
        Gemini, Cerebras) with 15-30 RPM limits.
    """

    def __init__(
        self,
        model_name: str,
        timeout: int = 120,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_backoff_seconds: float = 4.0,
    ) -> None:
        self._model = model_name
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = max(0.0, float(retry_backoff_seconds))
        # Same telemetry contract as OllamaCoverageJudge so the
        # pipeline's LLM_UNAVAILABLE-warning logic works unchanged.
        self.unavailable_count: int = 0
        self.last_error: str = ""

    # ── unavailability telemetry (same contract as OllamaCoverageJudge) ─

    def _record_unavailable(self, reason: str) -> None:
        self.unavailable_count += 1
        self.last_error = reason

    def consume_unavailability(self) -> tuple[int, str]:
        count, err = self.unavailable_count, self.last_error
        self.unavailable_count = 0
        self.last_error = ""
        return count, err

    # ── core call ───────────────────────────────────────────────────────

    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        try:
            import litellm
        except ImportError:
            self._record_unavailable("litellm not installed")
            return _FALLBACK.judge(req, unit)

        system_prompt, user_prompt = build_judge_prompt(req, unit)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self._max_retries:
            try:
                resp = litellm.completion(
                    model=self._model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=self._max_tokens,
                    timeout=self._timeout,
                )
                raw_text = self._extract_text(resp)
                if not raw_text:
                    raise ValueError(f"Empty completion text from {self._model}")
                parsed = _extract_json(raw_text)
                if not parsed:
                    raise ValueError(
                        f"Could not parse JSON from {self._model}: {raw_text[:200]}"
                    )
                judgment = _parse_response(
                    parsed,
                    req.req_id,
                    unit.unit_id,
                    unit.target_document_id,
                    evidence_text=unit.text,
                )
                logger.debug(
                    "LiteLLM judge: model=%s req=%s unit=%s → %s",
                    self._model, req.req_id[:8], unit.unit_id[:8], judgment.llm_label,
                )
                return judgment

            except Exception as exc:
                last_exc = exc
                msg = f"{type(exc).__name__}: {exc}"
                # Rate-limit-aware retry. LiteLLM normalises rate-limit
                # exceptions to litellm.RateLimitError; we also catch
                # by string sniff in case the underlying exception
                # leaks through (some providers wrap differently).
                is_rate_limit = (
                    type(exc).__name__ == "RateLimitError"
                    or "rate" in msg.lower() and "limit" in msg.lower()
                    or "429" in msg
                )
                if is_rate_limit and attempt < self._max_retries:
                    wait = self._retry_backoff * (2 ** attempt)
                    logger.warning(
                        "LiteLLM rate-limited (%s); sleeping %.1fs before retry %d/%d",
                        msg[:120], wait, attempt + 1, self._max_retries,
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                # Non-rate-limit error or out of retries → fall through.
                logger.warning("LiteLLM judge error (model=%s): %s", self._model, msg)
                self._record_unavailable(msg)
                return _FALLBACK.judge(req, unit)

        # All retries exhausted on rate-limit.
        self._record_unavailable(f"rate-limit retries exhausted: {last_exc}")
        return _FALLBACK.judge(req, unit)

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """LiteLLM normalises responses to OpenAI-shape:
        `resp.choices[0].message.content`. Handle both attribute and
        dict-like access defensively in case the wrapper changes."""
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            pass
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""
