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
import collections
import os
import re
import threading
import time
from typing import Any, Deque, Optional, Tuple

from app.core.logging import get_logger
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.coverage_judge import CoverageJudge
from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
from app.infrastructure.llm.ollama_coverage_judge import _extract_json, _parse_response
from app.infrastructure.llm.prompts import build_judge_prompt

# Audit (Polyakov re-run with Groq free tier llama-3.3-70b): the API
# error message includes a "try again in 3.07s" hint that tells us
# exactly when the next call will succeed. The previous fixed-backoff
# retry (4/8/16s) ignored this hint, often retrying inside the same
# saturated minute window and burning all retries before the TPM
# resets. This regex extracts the hinted seconds so we can honor it.
_RETRY_AFTER_HINT_RE = re.compile(
    r"(?:try\s+again\s+in|retry\s+after|please\s+wait)\s+"
    r"(\d+(?:\.\d+)?)\s*(?:seconds?|sec|s)\b",
    re.IGNORECASE,
)


def _parse_retry_after_hint(msg: str) -> Optional[float]:
    """Extract a 'retry after X seconds' hint from a rate-limit error
    message. Groq, OpenAI and most LiteLLM-routed providers include
    such a hint in the body. Returns the suggested wait in seconds, or
    None when no hint is present."""
    if not msg:
        return None
    m = _RETRY_AFTER_HINT_RE.search(msg)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None

logger = get_logger(__name__)
_FALLBACK = DisabledCoverageJudge()


# ── Token-bucket throttle (rate limit prevention, not just recovery) ────
#
# Audit (Polyakov re-run with Groq free tier): even with rate-limit-aware
# retry honoring the 'try again in Xs' hint, 33/41 LLM calls failed
# because the pipeline issues calls back-to-back. The first 4 calls burn
# 6K TPM in <2 seconds; subsequent calls all hit RateLimitError and the
# 5 retries × 14s wait isn't enough — multiple retries wake up at the
# same time and pound the API in a thundering herd.
#
# Solution: BEFORE issuing the call, consult a process-wide token bucket
# that tracks tokens used in the last 60 seconds. If adding this call
# would exceed the limit, sleep until the oldest entry rolls off — this
# spreads requests evenly across the window.
#
# Activation:
#   CQUALITY_LITELLM_TPM=5500     # set just below your provider's TPM
#                                 # (e.g. Groq free tier: 6000 → use 5500)
#   CQUALITY_LITELLM_TOKEN_EST=1500  # average tokens per pair-judgment
#                                    # (default 1500 — calibrated for the
#                                    # pair-judgment prompt + JSON response)
#
# When CQUALITY_LITELLM_TPM is unset or 0, throttling is OFF (legacy
# behaviour — useful when running ollama locally or on a paid tier with
# generous limits).


class _TokenBucket:
    """Thread-safe sliding-window token bucket.

    Tracks (timestamp, tokens) entries from calls within the last
    `window_seconds`. `reserve(est_tokens)` blocks until adding
    `est_tokens` would not exceed `limit`, then records the reservation.

    The implementation is deliberately simple: a deque + lock. Granular
    enough for our scale (dozens of calls per request, single-host
    deployment); a Redis-backed bucket would be overkill.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self._limit = int(limit)
        self._window = float(window_seconds)
        self._calls: Deque[Tuple[float, int]] = collections.deque()
        self._lock = threading.Lock()

    def reserve(self, est_tokens: int) -> float:
        """Block until `est_tokens` can be reserved without exceeding
        the limit. Returns the total seconds slept."""
        slept = 0.0
        while True:
            with self._lock:
                now = time.time()
                # Drop entries that have rolled off the window.
                while self._calls and now - self._calls[0][0] > self._window:
                    self._calls.popleft()
                used = sum(t for _, t in self._calls)
                if used + est_tokens <= self._limit or not self._calls:
                    self._calls.append((now, est_tokens))
                    return slept
                # Need to wait until the oldest entry rolls off.
                oldest_ts = self._calls[0][0]
                wait_for = (oldest_ts + self._window) - now + 0.5  # buffer
            wait_for = max(0.1, wait_for)
            logger.info(
                "Token bucket throttle: used=%d limit=%d est=%d sleeping=%.1fs",
                used, self._limit, est_tokens, wait_for,
            )
            time.sleep(wait_for)
            slept += wait_for


_TOKEN_BUCKET: Optional[_TokenBucket] = None
_TOKEN_BUCKET_LOCK = threading.Lock()
_TOKEN_BUCKET_PROBED = False


def _get_token_bucket() -> Optional[_TokenBucket]:
    """Lazy-init the module-level bucket from env. Returns None if
    throttling is disabled (CQUALITY_LITELLM_TPM unset or 0)."""
    global _TOKEN_BUCKET, _TOKEN_BUCKET_PROBED
    if _TOKEN_BUCKET_PROBED:
        return _TOKEN_BUCKET
    with _TOKEN_BUCKET_LOCK:
        if _TOKEN_BUCKET_PROBED:
            return _TOKEN_BUCKET
        raw = os.environ.get("CQUALITY_LITELLM_TPM", "0").strip()
        try:
            tpm = int(raw)
        except ValueError:
            tpm = 0
        if tpm > 0:
            _TOKEN_BUCKET = _TokenBucket(tpm)
            logger.info(
                "LiteLLM token-bucket throttle ENABLED at %d TPM "
                "(set CQUALITY_LITELLM_TPM=0 to disable)",
                tpm,
            )
        else:
            logger.info(
                "LiteLLM token-bucket throttle DISABLED "
                "(set CQUALITY_LITELLM_TPM=N to throttle to N tokens/min)",
            )
        _TOKEN_BUCKET_PROBED = True
        return _TOKEN_BUCKET


def _estimate_tokens_per_call() -> int:
    raw = os.environ.get("CQUALITY_LITELLM_TOKEN_EST", "1500").strip()
    try:
        return max(100, int(raw))
    except ValueError:
        return 1500


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
        max_retries: int = 5,
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
        # Model-fallback chain. When the primary model returns 404
        # (OpenRouter rotated it out of the free tier), permanently
        # switch to the next available alternative for this judge
        # instance. Configure via env CQUALITY_LITELLM_FALLBACK_MODELS
        # as a comma-separated list, or rely on the default chain
        # below which lists known-good free models in priority order.
        env_chain = os.environ.get("CQUALITY_LITELLM_FALLBACK_MODELS", "").strip()
        if env_chain:
            self._fallback_chain = [m.strip() for m in env_chain.split(",") if m.strip()]
        else:
            self._fallback_chain = [
                # OpenRouter free models that have stayed available the
                # longest. Order = priority; the chain is consulted only
                # when the primary returns 404 (model removed).
                "openrouter/meta-llama/llama-3.3-70b-instruct:free",
                "openrouter/google/gemma-3-27b-it:free",
                "openrouter/meta-llama/llama-4-scout:free",
                "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            ]
        # Skip our own model in the chain (avoid cycling).
        self._fallback_chain = [m for m in self._fallback_chain if m != self._model]
        self._exhausted_models: set = set()

    def _try_fallback_model(self, current_model: str) -> bool:
        """When the current model returns 404, switch to the next
        fallback. Returns True if a switch happened, False if the
        chain is exhausted."""
        self._exhausted_models.add(current_model)
        for candidate in self._fallback_chain:
            if candidate not in self._exhausted_models:
                logger.warning(
                    "LiteLLM model %r unavailable (404); switching to %r",
                    current_model, candidate,
                )
                self._model = candidate
                return True
        return False

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

        # Throttle BEFORE making the call so we don't burn through TPM in
        # a burst. When CQUALITY_LITELLM_TPM is unset, this is a no-op
        # (legacy path). When set (e.g. =5500 for Groq free 6K-TPM tier),
        # the bucket sleeps as needed to keep within the rolling window.
        bucket = _get_token_bucket()
        if bucket is not None:
            bucket.reserve(_estimate_tokens_per_call())

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
                # Model-removed: OpenRouter rotates :free endpoints. When
                # the configured model returns 404 we permanently switch
                # to the next entry in the fallback chain (no per-call
                # cost — the switch sticks for the rest of the process).
                is_not_found = (
                    type(exc).__name__ == "NotFoundError"
                    or "no endpoints found" in msg.lower()
                    or '"code":404' in msg
                )
                if is_not_found:
                    if self._try_fallback_model(self._model):
                        # Retry the same call against the new model
                        # (doesn't count against retry budget — this is
                        # a different model, fresh attempt).
                        continue
                    # Chain exhausted → fall through to fallback judge.
                    logger.warning(
                        "LiteLLM model fallback chain exhausted; using disabled judge",
                    )
                    self._record_unavailable(msg)
                    return _FALLBACK.judge(req, unit)
                # Rate-limit-aware retry. LiteLLM normalises rate-limit
                # exceptions to litellm.RateLimitError; we also catch
                # by string sniff in case the underlying exception
                # leaks through (some providers wrap differently).
                is_rate_limit = (
                    type(exc).__name__ == "RateLimitError"
                    or "rate" in msg.lower() and "limit" in msg.lower()
                    or "429" in msg
                )
                # OpenRouter free-tier upstream rate limits ("temporarily
                # rate-limited upstream") are short-lived shared-pool peaks
                # that don't include a 'try again in Xs' hint. Detect them
                # so we sleep longer (jittered 30-60s) instead of falling
                # straight through after exhausted retries.
                is_upstream_overload = (
                    "rate-limited upstream" in msg.lower()
                    or "rate limited upstream" in msg.lower()
                    or "temporarily" in msg.lower() and "limit" in msg.lower()
                )
                if is_rate_limit and attempt < self._max_retries:
                    # Honor the API's 'try again in Xs' hint when present
                    # — that's the authoritative number, not our guess.
                    # Add a 0.5s buffer so we don't race the reset window.
                    # Cap at 65s so a malicious / weird hint can't stall
                    # the pipeline for a whole package.
                    hinted = _parse_retry_after_hint(msg)
                    if hinted is not None:
                        wait = min(hinted + 0.5, 65.0)
                        wait = max(wait, self._retry_backoff)
                        why = f"hint={hinted:.2f}s"
                    elif is_upstream_overload:
                        # Upstream overload: peaks usually clear in 20-40s.
                        # Use jittered 30/45/60/60/60 sequence — multiple
                        # parallel callers won't sync up and re-saturate.
                        import random
                        base = 30.0 + 15.0 * min(attempt, 2)
                        jitter = random.uniform(-3.0, 3.0)
                        wait = min(max(base + jitter, 20.0), 65.0)
                        why = f"upstream overload, jittered {base:.0f}s±3s"
                    else:
                        # Linear-ish backoff (4/8/16/32/64) capped at 65s
                        # — gives the TPM window time to drain across
                        # multiple retries without exponentially diverging.
                        wait = min(
                            self._retry_backoff * (2 ** min(attempt, 4)),
                            65.0,
                        )
                        why = "no hint, exp backoff"
                    logger.warning(
                        "LiteLLM rate-limited (%s); sleeping %.1fs (%s) before retry %d/%d",
                        msg[:120], wait, why, attempt + 1, self._max_retries,
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

    # ── Batch path: 1 LLM call → N verdicts for one requirement ─────────
    #
    # PairJudgeService probes for `judge_batch` and uses it when present
    # (currently used by CrossEncoderCoverageJudge for GPU batching). For
    # rate-limited free APIs (Groq free tier 6K TPM, OpenRouter free tier
    # 50 reqs/day, shared-pool upstream overloads) calling once for N
    # candidates instead of N times cuts our call count 3-5× and fits
    # comfortably in those quotas.
    #
    # Activation: just having this method on the class is enough —
    # PairJudgeService picks it up automatically. To force per-pair path
    # for debugging, set CQUALITY_BATCH_JUDGE=0 in env.

    def judge_batch(
        self, req: RequirementUnit, units: list,
    ) -> list:
        """Judge N candidates for one requirement in a single LLM call.

        Returns a list of PairJudgment in the SAME ORDER as `units`. On
        any failure (rate-limit retries exhausted, parse error, length
        mismatch, missing entries) we fall back to per-pair `judge()`
        for the affected slots so the pipeline never loses a row.
        """
        if not units:
            return []

        # Disable switch — operators can force the legacy per-pair path
        # if a particular model misbehaves on batch prompts.
        if os.environ.get("CQUALITY_BATCH_JUDGE", "1").strip() in ("0", "false", "no"):
            return [self.judge(req, u) for u in units]

        try:
            import litellm
        except ImportError:
            self._record_unavailable("litellm not installed")
            return [_FALLBACK.judge(req, u) for u in units]

        from app.infrastructure.llm.prompts import build_judge_batch_prompt
        from app.domain.c_quality_enums import LLMLabel
        from app.domain.c_quality_models import PairJudgment

        system_prompt, user_prompt = build_judge_batch_prompt(req, units)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Throttle once per batch, not per candidate. Estimate tokens
        # proportionally to the number of units (each candidate
        # contributes prompt + response tokens).
        bucket = _get_token_bucket()
        if bucket is not None:
            est = _estimate_tokens_per_call() * max(1, len(units))
            bucket.reserve(min(est, 60000))  # cap so we don't deadlock

        attempt = 0
        last_exc: Optional[Exception] = None
        raw_text: str = ""
        while attempt <= self._max_retries:
            try:
                resp = litellm.completion(
                    model=self._model,
                    messages=messages,
                    max_tokens=min(4096, self._max_tokens * len(units)),
                    timeout=self._timeout,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw_text = self._extract_text(resp)
                break
            except Exception as exc:
                last_exc = exc
                msg = f"{type(exc).__name__}: {exc}"
                # Model-removed: same fallback chain logic as judge().
                is_not_found = (
                    type(exc).__name__ == "NotFoundError"
                    or "no endpoints found" in msg.lower()
                    or '"code":404' in msg
                )
                if is_not_found:
                    if self._try_fallback_model(self._model):
                        continue
                    logger.warning(
                        "LiteLLM batch model fallback chain exhausted; per-pair fallback",
                    )
                    self._record_unavailable(msg)
                    return [self.judge(req, u) for u in units]
                is_rate_limit = (
                    type(exc).__name__ == "RateLimitError"
                    or "rate" in msg.lower() and "limit" in msg.lower()
                    or "429" in msg
                )
                is_upstream_overload = (
                    "rate-limited upstream" in msg.lower()
                    or "rate limited upstream" in msg.lower()
                    or "temporarily" in msg.lower() and "limit" in msg.lower()
                )
                if is_rate_limit and attempt < self._max_retries:
                    hinted = _parse_retry_after_hint(msg)
                    if hinted is not None:
                        wait = min(hinted + 0.5, 65.0)
                        wait = max(wait, self._retry_backoff)
                    elif is_upstream_overload:
                        import random
                        base = 30.0 + 15.0 * min(attempt, 2)
                        wait = min(max(base + random.uniform(-3.0, 3.0), 20.0), 65.0)
                    else:
                        wait = min(self._retry_backoff * (2 ** min(attempt, 4)), 65.0)
                    logger.warning(
                        "LiteLLM batch rate-limited (%s); sleeping %.1fs before retry %d/%d",
                        msg[:120], wait, attempt + 1, self._max_retries,
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                logger.warning(
                    "LiteLLM batch judge error (model=%s); falling back to per-pair: %s",
                    self._model, msg,
                )
                self._record_unavailable(msg)
                return [self.judge(req, u) for u in units]
        else:
            self._record_unavailable(f"batch retries exhausted: {last_exc}")
            return [_FALLBACK.judge(req, u) for u in units]

        # Parse the JSON array.
        verdicts = self._parse_batch_response(raw_text, expected_count=len(units))
        if verdicts is None:
            logger.warning(
                "LiteLLM batch parse failed (model=%s); falling back to per-pair. "
                "raw=%s",
                self._model, raw_text[:300],
            )
            return [self.judge(req, u) for u in units]

        # Build PairJudgments in input order. Per-slot fallback on holes.
        from app.infrastructure.llm.ollama_coverage_judge import _parse_response as _per_pair_parse

        out = []
        for i, unit in enumerate(units):
            entry = verdicts.get(i)
            if entry is None:
                # Slot missing in response → per-pair fallback (1 extra call).
                logger.info(
                    "LiteLLM batch: slot %d missing in response; per-pair fallback",
                    i,
                )
                out.append(self.judge(req, unit))
                continue
            try:
                judgment = _per_pair_parse(
                    entry,
                    req.req_id,
                    unit.unit_id,
                    unit.target_document_id,
                    evidence_text=unit.text,
                )
                out.append(judgment)
            except Exception as exc:
                logger.warning(
                    "LiteLLM batch: slot %d parse failed (%s); per-pair fallback",
                    i, exc,
                )
                out.append(self.judge(req, unit))
        return out

    @staticmethod
    def _parse_batch_response(raw_text: str, expected_count: int):
        """Parse the JSON array response from a batch call. Returns a
        dict mapping index → verdict-dict, or None if parse failed.

        Tolerant of:
          - top-level array OR object with "results"/"verdicts" key;
          - missing "index" field (uses array position);
          - slight count drift (returns whatever indices it can parse).
        """
        import json

        # Try direct parse first.
        parsed: Any = None
        try:
            parsed = json.loads(raw_text)
        except Exception:
            # Fall back to extracting JSON-looking substring.
            from app.infrastructure.llm.ollama_coverage_judge import _extract_json
            extracted = _extract_json(raw_text)
            if isinstance(extracted, dict):
                parsed = extracted
            elif extracted:
                parsed = extracted

        # Coerce wrapper objects into a list.
        items = None
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            for key in ("results", "verdicts", "items", "data"):
                if isinstance(parsed.get(key), list):
                    items = parsed[key]
                    break
        if not isinstance(items, list) or not items:
            return None

        result: dict = {}
        for pos, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            idx = it.get("index")
            if not isinstance(idx, int) or idx < 0:
                idx = pos
            if idx < expected_count and idx not in result:
                result[idx] = it
        return result if result else None

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
