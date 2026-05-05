"""
Ollama-backed coverage judge.
Reuses the subprocess pattern already present in app/judge/llm_judge.py,
adapted for the CoverageJudge interface and structured JSON schema.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List

import requests

from app.core.logging import get_logger
from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.coverage_judge import CoverageJudge
from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
from app.infrastructure.llm.judgment_cache import JudgmentCache
from app.infrastructure.llm.prompts import (
    PROMPT_VERSION,
    build_judge_prompt,
    build_judge_prompt_compact,
    should_use_compact_prompt,
)

logger = get_logger(__name__)

_VALID_LABELS = {l.value for l in LLMLabel}
_FALLBACK = DisabledCoverageJudge()

# Retry parameters for transient JSON-parse failures (empty / garbled response
# from small models like qwen2.5:3b). Only parse failures are retried —
# timeouts and connection errors are NOT retried because they indicate a
# systemic issue (Ollama overloaded / VRAM exhausted) where a fast retry
# would just pile onto the problem.
#
# CQUALITY_JUDGE_RETRIES: total attempts (default 2 = 1 original + 1 retry).
# Hard max 5 to avoid hanging the pipeline. Set to 1 to disable retries.
_RETRY_BACKOFF_SECS: float = 1.0
_RETRY_MAX_CAP: int = 5


def _resolve_max_attempts() -> int:
    """Read CQUALITY_JUDGE_RETRIES, clamp to [1, cap]. Default 2."""
    raw = os.environ.get("CQUALITY_JUDGE_RETRIES", "2").strip()
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "CQUALITY_JUDGE_RETRIES=%r is not an int; using 2", raw,
        )
        return 2
    if n < 1:
        return 1
    if n > _RETRY_MAX_CAP:
        logger.warning(
            "CQUALITY_JUDGE_RETRIES=%d > cap %d; clamping", n, _RETRY_MAX_CAP,
        )
        return _RETRY_MAX_CAP
    return n


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from the model response."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Regex fallback: grab first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _normalize_for_grounding(text: str) -> str:
    """Lowercase + collapse whitespace. Used to compare cited_phrases against
    the underlying CoverageUnit text — the LLM frequently introduces minor
    whitespace / case differences that should NOT count as ungrounded."""
    return " ".join((text or "").lower().split())


def _parse_response(
    raw: dict,
    req_id: str,
    unit_id: str,
    doc_id: str,
    evidence_text: str = "",
) -> PairJudgment:
    raw_label = str(raw.get("label", "IRRELEVANT")).upper()
    label = LLMLabel(raw_label) if raw_label in _VALID_LABELS else LLMLabel.IRRELEVANT

    def _str_list(key: str) -> List[str]:
        val = raw.get(key, [])
        if isinstance(val, list):
            return [str(v) for v in val]
        return []

    matched = _str_list("matched_aspects")
    missing = _str_list("missing_aspects")
    conflict = _str_list("conflict_aspects")
    cited = _str_list("cited_phrases")

    # Post-validation: small local LLMs frequently output label=CONFLICT while
    # leaving conflict_aspects empty and describing a partial-coverage case in
    # the free-text explanation ("содержит X, но не полностью…"). A true
    # CONFLICT requires an explicit contradicting aspect. Without it we
    # downgrade: PARTIAL if there is any overlap signal, IRRELEVANT otherwise.
    if label == LLMLabel.CONFLICT and not conflict:
        label = LLMLabel.PARTIAL if matched else LLMLabel.IRRELEVANT

    # ── BUG-3: grounding gate ──────────────────────────────────────────────
    # For any non-IRRELEVANT verdict the LLM MUST quote at least one phrase
    # that we can locate inside the evidence text (substring match,
    # case/whitespace-insensitive). The audit-time symptom was a CONFLICT
    # rationale that mentioned "период защиты" while no evidence fragment
    # contained that phrase — the verdict was a hallucination.
    #
    # Contract:
    #   * label == IRRELEVANT          → cited_phrases ignored (always empty).
    #   * label != IRRELEVANT, no
    #     evidence_text passed in     → grounding check skipped (legacy path).
    #   * label != IRRELEVANT,
    #     evidence_text non-empty,
    #     cited_phrases empty OR
    #     none match                   → demoted to IRRELEVANT, low_confidence
    #                                    is set so the aggregator can flag it.
    low_confidence = False
    if label != LLMLabel.IRRELEVANT and evidence_text:
        evidence_norm = _normalize_for_grounding(evidence_text)
        grounded = [
            p for p in cited
            if p and _normalize_for_grounding(p) and _normalize_for_grounding(p) in evidence_norm
        ]
        # Audit (Polyakov: every "[ungrounded]" demotion): the strict substring
        # gate kills legitimate verdicts whose evidence is on-topic but whose
        # citations the LLM paraphrases (synonym, word reorder, dropped
        # punctuation). Token-overlap fallback: count distinct content tokens
        # (≥ 4 chars) shared between cited_phrases and evidence. ≥ 3 is the
        # minimum bar where the citation is genuinely supported even if not
        # a verbatim substring; below that the LLM most likely hallucinated.
        if not grounded and cited:
            cited_tokens = set()
            for p in cited:
                for tok in _normalize_for_grounding(p).split():
                    if len(tok) >= 4:
                        cited_tokens.add(tok)
            evidence_tokens = set(
                tok for tok in evidence_norm.split() if len(tok) >= 4
            )
            shared = cited_tokens & evidence_tokens
            if len(shared) >= 3:
                grounded = [f"<fuzzy:{len(shared)}_tokens>"]
        if not grounded:
            label = LLMLabel.IRRELEVANT
            low_confidence = True
            cited = []
            matched = []
            missing = []
            conflict = []

    return PairJudgment(
        req_id=req_id,
        unit_id=unit_id,
        target_document_id=doc_id,
        llm_label=label,
        # Calibrate confidence down when grounding failed — caps at 0.3 so
        # the aggregator's strong-COVERED suppression rule can't latch onto
        # an ungrounded judgment.
        llm_confidence=min(float(raw.get("confidence", 0.5)), 0.3) if low_confidence else float(raw.get("confidence", 0.5)),
        rule_adjusted_label=label,
        matched_aspects=matched,
        missing_aspects=missing,
        conflict_aspects=conflict,
        cited_phrases=cited,
        low_confidence=low_confidence,
        # PR-K P0: when the response-parser demoted the verdict to IRRELEVANT
        # because cited_phrases weren't substring-matched, this is a true
        # grounding failure (LLM hallucinated citations). The aggregator
        # treats this as "ungrounded" and rejects COVERED. The pipeline-side
        # below-evidence-floor flag (set in run_coverage_analysis) should
        # NOT set this — that's a retrieval-quality issue, not a grounding bug.
        grounding_failed=low_confidence,
        explanation=(
            "[ungrounded] LLM verdict not supported by any phrase substring "
            "of the evidence; demoted to IRRELEVANT. Original explanation: "
            + str(raw.get("explanation", ""))
        ) if low_confidence else str(raw.get("explanation", "")),
    )


class OllamaCoverageJudge(CoverageJudge):
    """Calls the local Ollama HTTP API. Falls back to DisabledCoverageJudge on error.

    The previous implementation shelled out to `ollama run` per pair, which
    re-loaded the model for every call and made realistic pair counts
    (dozens to hundreds) unworkable within any sane request timeout. The
    daemon API (`POST /api/generate`) keeps the model hot across calls.
    """

    def __init__(self, model_name: str = "llama3:8b", timeout: int = 120) -> None:
        self._model = model_name
        self._timeout = timeout
        self._url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        # BUG-09 fix: track silent fallbacks so the pipeline can surface
        # an LLM_UNAVAILABLE warning to the user. Without this, every pair
        # is silently judged IRRELEVANT (DisabledCoverageJudge) and the
        # final report is full of MISSING with no explanation. The pipeline
        # consumes these via `consume_unavailability()` after judging.
        self.unavailable_count: int = 0
        self.last_error: str = ""
        # PR-K post-fix: optional persistent judgment cache. Enabled by
        # CQUALITY_JUDGE_CACHE_DIR env var. None when disabled — the
        # judge then behaves exactly as before. The cache key includes
        # the model + prompt version + backend, so changing any of
        # those auto-invalidates entries.
        self._cache = JudgmentCache.from_env(
            model=self._model,
            prompt_version=PROMPT_VERSION,
            backend="ollama",
        )
        if self._cache is not None:
            logger.info(
                "OllamaCoverageJudge: judgment cache enabled at %s",
                self._cache.stats().get("db_path"),
            )

    def _record_unavailable(self, reason: str) -> None:
        self.unavailable_count += 1
        # Keep the most recent error for diagnostics; old ones are usually
        # the same root cause (timeout / connection refused).
        self.last_error = reason

    def consume_unavailability(self) -> tuple[int, str]:
        """Read-and-reset the unavailability counter and last error message.

        Pipeline calls this once after the judging loop completes to decide
        whether to append an LLM_UNAVAILABLE warning to the result.
        """
        count = self.unavailable_count
        err = self.last_error
        self.unavailable_count = 0
        self.last_error = ""
        return count, err

    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        # PR-K post-fix: cache lookup. Returned judgment has its
        # req_id/unit_id/target_document_id rebound to the live pair
        # by the cache itself, so the caller doesn't need to handle
        # cross-package id collisions.
        if self._cache is not None:
            cached = self._cache.get(req, unit)
            if cached is not None:
                logger.debug(
                    "Ollama judge: req=%s unit=%s → CACHE HIT (%s)",
                    req.req_id[:8], unit.unit_id[:8], cached.llm_label.value,
                )
                return cached

        # PR-K P1: small models (qwen2.5:3b etc) get a compact prompt that
        # avoids prepended metadata fields (the smoke-time symptom: 3B
        # echoed prompt-v5 field labels back into JSON, breaking the parse).
        if should_use_compact_prompt(self._model):
            system_prompt, user_prompt = build_judge_prompt_compact(req, unit)
        else:
            system_prompt, user_prompt = build_judge_prompt(req, unit)
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

        # ── Retry loop ────────────────────────────────────────────────────
        # Only parse failures (empty / garbled JSON) are retried — they are
        # transient (3B models occasionally produce malformed output on the
        # first attempt but succeed on the second). Hard errors (timeout,
        # connection refused, HTTP error) break out immediately: retrying
        # into an overloaded Ollama instance makes things worse, not better.
        max_attempts = _resolve_max_attempts()
        last_parse_error: str = ""

        for attempt in range(max_attempts):
            if attempt > 0:
                delay = _RETRY_BACKOFF_SECS * (2 ** (attempt - 1))  # 1 s, 2 s, …
                logger.info(
                    "Ollama judge: parse retry %d/%d for req=%s unit=%s "
                    "(delay=%.1fs, reason: %s)",
                    attempt, max_attempts - 1,
                    req.req_id[:8], unit.unit_id[:8], delay, last_parse_error,
                )
                time.sleep(delay)

            try:
                resp = requests.post(
                    self._url,
                    json={
                        "model": self._model,
                        "prompt": full_prompt,
                        "stream": False,
                        # keep_alive prevents Ollama from unloading the model
                        # between pair calls (model reload = +3-5s per call).
                        "keep_alive": "30m",
                        "options": {
                            "temperature": 0.1,
                            # Cap context to 2k tokens — judge prompts are
                            # short, default 4k wastes prompt_eval time.
                            "num_ctx": 2048,
                            # PR-F BUG-3-followup: 256 was too tight for v3
                            # prompt (cited_phrases JSON tail got truncated).
                            # 512 absorbs ≈430-token worst case without
                            # noticeably slowing throughput.
                            "num_predict": 512,
                        },
                    },
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                raw_text = (resp.json().get("response") or "").strip()
                if not raw_text:
                    last_parse_error = "empty response"
                    logger.debug(
                        "Ollama judge: empty response (attempt %d) req=%s unit=%s",
                        attempt + 1, req.req_id[:8], unit.unit_id[:8],
                    )
                    continue  # retry

                parsed = _extract_json(raw_text)
                if not parsed:
                    last_parse_error = f"JSON parse failed: {raw_text[:120]!r}"
                    logger.debug(
                        "Ollama judge: JSON parse failed (attempt %d) req=%s unit=%s",
                        attempt + 1, req.req_id[:8], unit.unit_id[:8],
                    )
                    continue  # retry

                # ── Success path ─────────────────────────────────────────
                judgment = _parse_response(
                    parsed, req.req_id, unit.unit_id, unit.target_document_id,
                    evidence_text=unit.text,
                )
                logger.debug(
                    "Ollama judge: req=%s unit=%s → %s (attempt %d)",
                    req.req_id[:8], unit.unit_id[:8], judgment.llm_label, attempt + 1,
                )
                # Persist on the success path only. Failures are deliberately
                # NOT cached — we want a real LLM response on the next attempt.
                if self._cache is not None:
                    self._cache.put(req, unit, judgment)
                return judgment

            except requests.Timeout:
                msg = f"timeout after {self._timeout}s"
                logger.warning(
                    "Ollama timed out: req=%s unit=%s (%s)",
                    req.req_id[:8], unit.unit_id[:8], msg,
                )
                self._record_unavailable(msg)
                return _FALLBACK.judge(req, unit)  # no retry

            except requests.ConnectionError as exc:
                msg = f"ConnectionError: {exc}"
                logger.warning(
                    "Ollama connection error: req=%s unit=%s: %s",
                    req.req_id[:8], unit.unit_id[:8], msg,
                )
                self._record_unavailable(msg)
                return _FALLBACK.judge(req, unit)  # no retry

            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                msg = f"HTTP {status}: {exc}"
                logger.warning(
                    "Ollama HTTP error: req=%s unit=%s: %s",
                    req.req_id[:8], unit.unit_id[:8], msg,
                )
                self._record_unavailable(msg)
                return _FALLBACK.judge(req, unit)  # no retry

            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Ollama judge unexpected error: req=%s unit=%s: %s",
                    req.req_id[:8], unit.unit_id[:8], msg,
                )
                self._record_unavailable(msg)
                return _FALLBACK.judge(req, unit)  # no retry

        # All attempts exhausted by parse failures — fall back gracefully.
        msg = (
            f"JSON parse failed after {max_attempts} attempt(s): {last_parse_error}"
        )
        logger.warning(
            "Ollama judge parse exhausted: req=%s unit=%s: %s",
            req.req_id[:8], unit.unit_id[:8], msg,
        )
        self._record_unavailable(msg)
        return _FALLBACK.judge(req, unit)
