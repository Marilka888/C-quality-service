"""
Polyakov-regression: env-var overrides for retrieval thresholds.

The C-quality service ships with conservative retrieval thresholds
(`min_retrieval_score=0.05`, `evidence_floor=0.30`) calibrated against
typical packages. Polyakov-class packages whose ПЗ paraphrases the ТЗ
heavily have max retrieval score 0.20-0.34 on the BoW component —
genuine COVERED paraphrases get reported as MISSING_NO_EVIDENCE
because the floor rejects the LLM verdict.

The fix exposes both thresholds through env vars without requiring
per-request flags or a config refactor:

    CQUALITY_MIN_RETRIEVAL_SCORE=0.15  # lower retrieval gate
    CQUALITY_EVIDENCE_FLOOR=0.18       # lower trust floor

This test pins the contract: `CoverageConfig.from_env()` reads both
variables, validates them, and falls back to defaults when unset or
unparseable.
"""
from __future__ import annotations

from unittest.mock import patch

from app.core.config import CoverageConfig


def test_from_env_no_overrides_uses_defaults_polyakov() -> None:
    with patch.dict("os.environ", {}, clear=False):
        # Drop our keys if a previous test set them.
        import os
        for k in ("CQUALITY_MIN_RETRIEVAL_SCORE", "CQUALITY_EVIDENCE_FLOOR",
                  "CQUALITY_LLM_MODEL_NAME"):
            os.environ.pop(k, None)
        cfg = CoverageConfig.from_env()
    assert cfg.retrieval.min_retrieval_score == 0.05
    assert cfg.retrieval.evidence_floor == 0.30


def test_from_env_lower_thresholds_for_polyakov() -> None:
    # The recommended Polyakov-class calibration: min_retrieval_score
    # 0.15, evidence_floor 0.18. Both must propagate.
    with patch.dict("os.environ", {
        "CQUALITY_MIN_RETRIEVAL_SCORE": "0.15",
        "CQUALITY_EVIDENCE_FLOOR": "0.18",
    }, clear=False):
        cfg = CoverageConfig.from_env()
    assert abs(cfg.retrieval.min_retrieval_score - 0.15) < 1e-9
    assert abs(cfg.retrieval.evidence_floor - 0.18) < 1e-9


def test_from_env_rejects_unparseable_values_polyakov() -> None:
    # Garbage env values must NOT crash the service — fall back to the
    # built-in defaults so a typo in deployment config doesn't take down
    # C-quality.
    with patch.dict("os.environ", {
        "CQUALITY_MIN_RETRIEVAL_SCORE": "not-a-number",
        "CQUALITY_EVIDENCE_FLOOR": "",
    }, clear=False):
        cfg = CoverageConfig.from_env()
    assert cfg.retrieval.min_retrieval_score == 0.05
    assert cfg.retrieval.evidence_floor == 0.30


def test_from_env_rejects_out_of_range_values_polyakov() -> None:
    # Out-of-[0, 1] values silently fall back. Any pydantic validation
    # error from constructing the field with those would crash the
    # service — env path explicitly filters first.
    with patch.dict("os.environ", {
        "CQUALITY_MIN_RETRIEVAL_SCORE": "1.5",
        "CQUALITY_EVIDENCE_FLOOR": "-0.2",
    }, clear=False):
        cfg = CoverageConfig.from_env()
    assert cfg.retrieval.min_retrieval_score == 0.05
    assert cfg.retrieval.evidence_floor == 0.30


def test_from_env_propagates_llm_model_name() -> None:
    with patch.dict("os.environ", {
        "CQUALITY_LLM_MODEL_NAME": "qwen2.5:14b",
    }, clear=False):
        cfg = CoverageConfig.from_env()
    assert cfg.llm.model_name == "qwen2.5:14b"
