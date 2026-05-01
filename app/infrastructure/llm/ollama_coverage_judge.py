"""
Ollama-backed coverage judge.
Reuses the subprocess pattern already present in app/judge/llm_judge.py,
adapted for the CoverageJudge interface and structured JSON schema.
"""
from __future__ import annotations

import json
import os
import re
from typing import List

import requests

from app.core.logging import get_logger
from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.coverage_judge import CoverageJudge
from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
from app.infrastructure.llm.prompts import build_judge_prompt

logger = get_logger(__name__)

_VALID_LABELS = {l.value for l in LLMLabel}
_FALLBACK = DisabledCoverageJudge()


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
        system_prompt, user_prompt = build_judge_prompt(req, unit)
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

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
                        # Cap context to 2k tokens — judge prompts are short,
                        # default 4k wastes prompt_eval time.
                        "num_ctx": 2048,
                        # PR-F BUG-3-followup: 256 was too tight for the v3
                        # prompt that asks for cited_phrases. On real packages
                        # with multi-phrase quotes the JSON tail got truncated
                        # mid-array (saw `\"cited_phrases\": [\\n  \"...`)
                        # and the parser fell back to DisabledCoverageJudge.
                        # 512 absorbs the worst observed cases (≈430 tokens
                        # for 5 long quotes in a CONFLICT verdict) without
                        # noticeably slowing throughput.
                        "num_predict": 512,
                    },
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw_text = (resp.json().get("response") or "").strip()
            if not raw_text:
                raise ValueError("Empty Ollama response")

            parsed = _extract_json(raw_text)
            if not parsed:
                raise ValueError(f"Could not parse JSON from: {raw_text[:200]}")

            judgment = _parse_response(
                parsed, req.req_id, unit.unit_id, unit.target_document_id,
                evidence_text=unit.text,
            )
            logger.debug("Ollama judge: req=%s unit=%s → %s", req.req_id[:8], unit.unit_id[:8], judgment.llm_label)
            return judgment

        except requests.Timeout:
            msg = f"timeout after {self._timeout}s"
            logger.warning("Ollama timed out for req=%s unit=%s (%s)",
                           req.req_id[:8], unit.unit_id[:8], msg)
            self._record_unavailable(msg)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.warning("Ollama judge error: %s", msg)
            self._record_unavailable(msg)

        return _FALLBACK.judge(req, unit)
