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


def _parse_response(raw: dict, req_id: str, unit_id: str, doc_id: str) -> PairJudgment:
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

    # Post-validation: small local LLMs frequently output label=CONFLICT while
    # leaving conflict_aspects empty and describing a partial-coverage case in
    # the free-text explanation ("содержит X, но не полностью…"). A true
    # CONFLICT requires an explicit contradicting aspect. Without it we
    # downgrade: PARTIAL if there is any overlap signal, IRRELEVANT otherwise.
    if label == LLMLabel.CONFLICT and not conflict:
        label = LLMLabel.PARTIAL if matched else LLMLabel.IRRELEVANT

    return PairJudgment(
        req_id=req_id,
        unit_id=unit_id,
        target_document_id=doc_id,
        llm_label=label,
        llm_confidence=float(raw.get("confidence", 0.5)),
        rule_adjusted_label=label,
        matched_aspects=matched,
        missing_aspects=missing,
        conflict_aspects=conflict,
        explanation=str(raw.get("explanation", "")),
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
                        # Cap response length. The JSON verdict fits in <200
                        # tokens; longer outputs are model padding.
                        "num_predict": 256,
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

            judgment = _parse_response(parsed, req.req_id, unit.unit_id, unit.target_document_id)
            logger.debug("Ollama judge: req=%s unit=%s → %s", req.req_id[:8], unit.unit_id[:8], judgment.llm_label)
            return judgment

        except requests.Timeout:
            logger.warning("Ollama timed out for req=%s unit=%s", req.req_id[:8], unit.unit_id[:8])
        except Exception as exc:
            logger.warning("Ollama judge error: %s", exc)

        return _FALLBACK.judge(req, unit)
