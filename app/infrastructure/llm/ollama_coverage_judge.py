"""
Ollama-backed coverage judge.
Reuses the subprocess pattern already present in app/judge/llm_judge.py,
adapted for the CoverageJudge interface and structured JSON schema.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import List

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

    return PairJudgment(
        req_id=req_id,
        unit_id=unit_id,
        target_document_id=doc_id,
        llm_label=label,
        llm_confidence=float(raw.get("confidence", 0.5)),
        rule_adjusted_label=label,
        matched_aspects=_str_list("matched_aspects"),
        missing_aspects=_str_list("missing_aspects"),
        conflict_aspects=_str_list("conflict_aspects"),
        explanation=str(raw.get("explanation", "")),
    )


class OllamaCoverageJudge(CoverageJudge):
    """Calls local Ollama via subprocess. Falls back to DisabledCoverageJudge on error."""

    def __init__(self, model_name: str = "llama3:8b", timeout: int = 120) -> None:
        self._model = model_name
        self._timeout = timeout
        self._ollama_path = os.environ.get(
            "OLLAMA_PATH",
            r"C:\Users\Marilka\AppData\Local\Programs\Ollama\ollama.exe",
        )

    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        system_prompt, user_prompt = build_judge_prompt(req, unit)
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

        try:
            result = subprocess.run(
                [self._ollama_path, "run", self._model],
                input=full_prompt,
                capture_output=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
            raw_text = result.stdout.strip()
            if not raw_text:
                raise ValueError("Empty Ollama response")

            parsed = _extract_json(raw_text)
            if not parsed:
                raise ValueError(f"Could not parse JSON from: {raw_text[:200]}")

            judgment = _parse_response(parsed, req.req_id, unit.unit_id, unit.target_document_id)
            logger.debug("Ollama judge: req=%s unit=%s → %s", req.req_id[:8], unit.unit_id[:8], judgment.llm_label)
            return judgment

        except subprocess.TimeoutExpired:
            logger.warning("Ollama timed out for req=%s unit=%s", req.req_id[:8], unit.unit_id[:8])
        except Exception as exc:
            logger.warning("Ollama judge error: %s", exc)

        return _FALLBACK.judge(req, unit)
