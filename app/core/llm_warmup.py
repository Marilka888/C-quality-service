"""LLM warmup + health-probe helpers for C-quality.

Lives in app/core (not app/main) so the warmup logic can be unit-tested
without pulling in the full FastAPI route graph (which transitively
imports numpy / sentence-transformers).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _resolve_llm_settings() -> tuple[str, str, str, float]:
    """Return (backend, model_name, ollama_url, warmup_timeout_s).

    Honours OLLAMA_URL / CQUALITY_LLM_MODEL_NAME env-var overrides so
    operators can repoint the warmup at a remote Ollama (typical in
    docker-compose) without rebuilding the image.
    """
    from app.core.config import CoverageConfig

    cfg = CoverageConfig().llm
    backend = (cfg.backend or "ollama").lower()
    model = os.environ.get("CQUALITY_LLM_MODEL_NAME") or cfg.model_name
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    timeout = float(os.environ.get("CQUALITY_LLM_WARMUP_TIMEOUT", "90"))
    return backend, model, url, timeout


def probe_llm(timeout_s: float) -> Dict[str, Any]:
    """Synchronously call Ollama with a tiny prompt; return a status
    dict the /health response can serialise. Never raises.
    """
    backend, model, url, _ = _resolve_llm_settings()
    if backend != "ollama":
        return {
            "ok": True,
            "backend": backend,
            "model": model,
            "note": "non-ollama backend — skipping warmup probe",
        }
    started = time.monotonic()
    try:
        import requests
        resp = requests.post(
            url,
            json={
                "model": model,
                "prompt": "ping",
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1, "temperature": 0.0, "num_ctx": 64},
            },
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - started
        if resp.status_code != 200:
            return {
                "ok": False,
                "backend": backend,
                "model": model,
                "url": url,
                "elapsed_s": round(elapsed, 2),
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
        return {
            "ok": True,
            "backend": backend,
            "model": model,
            "url": url,
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as exc:  # noqa: BLE001 — health probe must never crash
        return {
            "ok": False,
            "backend": backend,
            "model": model,
            "url": url,
            "elapsed_s": round(time.monotonic() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def warmup_async() -> None:
    """Background warmup: pin the model into Ollama's RAM so the first
    real /coverage/analyze call doesn't pay the cold-load tax.

    Runs in a thread because `requests` is sync; we don't want to block
    the asyncio loop while Ollama loads the model.
    """
    backend, model, _, timeout = _resolve_llm_settings()
    if backend != "ollama":
        logger.info("LLM warmup skipped (backend=%s)", backend)
        return
    logger.info("LLM warmup starting (model=%s, timeout=%ss)", model, timeout)
    result = await asyncio.to_thread(probe_llm, timeout)
    if result.get("ok"):
        logger.info(
            "LLM warmup OK: model=%s elapsed=%.2fs",
            model, result.get("elapsed_s", -1),
        )
    else:
        logger.warning(
            "LLM warmup FAILED: model=%s error=%s — service will still "
            "serve non-LLM routes; first coverage call may pay cold-load tax",
            model, result.get("error"),
        )
