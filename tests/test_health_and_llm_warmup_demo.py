"""
P0 #8 — pin large LLM for the demo and add /health + warmup.

Three contracts pinned:

  1. Default LLM model is at least 7B (qwen2.5:7b). qwen2.5:3b produced
     unstable confidence ≥ 0.7 on marginal matches across all five
     demo packages.
  2. The probe function (used by both startup warmup and /health) never
     raises — connection / timeout errors return a structured payload
     with ok=False so the docker-compose health-check loop can wait
     until the LLM is hot without crashing the service.
  3. CQUALITY_LLM_MODEL_NAME env var overrides the default at runtime.
"""
from __future__ import annotations

import os

from unittest.mock import patch

from app.core.config import CoverageConfig
from app.core.llm_warmup import probe_llm, _resolve_llm_settings


def test_default_llm_model_is_at_least_7b_demo() -> None:
    cfg = CoverageConfig().llm
    assert cfg.model_name == "qwen2.5:7b", (
        f"default model must be ≥ 7B for the demo (got {cfg.model_name!r}); "
        "qwen2.5:3b produced unstable confidence on marginal matches"
    )
    assert cfg.backend == "ollama"


def test_probe_llm_never_raises_on_connection_error_demo() -> None:
    # Direct exercise of probe_llm with a guaranteed-unreachable URL:
    # the function must catch the exception and return a structured
    # error payload rather than propagate. This contract is what makes
    # /health safe to use in `docker-compose … condition: service_healthy`
    # loops — a brief Ollama outage must not 500 the health endpoint.
    with patch.dict(
        os.environ,
        {"OLLAMA_URL": "http://127.0.0.1:1/api/generate"},
        clear=False,
    ):
        out = probe_llm(timeout_s=1.0)
    assert out["ok"] is False
    assert "error" in out
    # Whichever exception the requests stack raises, it must be
    # captured into the payload — no propagation.
    assert out["backend"] == "ollama"


def test_probe_llm_skips_when_backend_is_not_ollama_demo() -> None:
    # When the deployment switches to litellm / disabled, the probe
    # should report ok=True without trying to call /api/generate.
    cfg = CoverageConfig()
    original_backend = cfg.llm.backend
    try:
        cfg.llm.backend = "litellm"
        # _resolve_llm_settings re-reads CoverageConfig() each call, so
        # we patch the constructor for this test.
        with patch("app.core.config.CoverageConfig", return_value=cfg):
            out = probe_llm(timeout_s=1.0)
        assert out["ok"] is True
        assert out["backend"] == "litellm"
        assert "skipping warmup" in out["note"]
    finally:
        cfg.llm.backend = original_backend


def test_env_var_overrides_default_model_name_demo() -> None:
    with patch.dict(
        os.environ,
        {"CQUALITY_LLM_MODEL_NAME": "qwen2.5:14b"},
        clear=False,
    ):
        backend, model, url, timeout = _resolve_llm_settings()
    assert model == "qwen2.5:14b"
    assert backend == "ollama"
    assert "11434" in url
