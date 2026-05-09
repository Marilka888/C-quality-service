"""FastAPI entrypoint for the C-quality service.

P0 #8 — startup warmup and /health endpoint. See app/core/llm_warmup
for the probe logic; this module wires it into FastAPI's lifespan and
exposes /health.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI

from app.api.routes import router
from app.core.llm_warmup import probe_llm, warmup_async


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Fire-and-forget — startup must not block on a slow Ollama load.
    asyncio.create_task(warmup_async())
    yield


app = FastAPI(title="C Quality Service", version="2.0.0", lifespan=_lifespan)
app.include_router(router)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + LLM readiness probe.

    docker-compose can use this with `condition: service_healthy` to
    sequence start-up so the orchestrator only fires once the C-quality
    LLM is hot. The probe uses a tighter timeout than warmup so the
    health check itself stays responsive — if the model isn't ready
    yet the response carries `llm.ok=false` and the orchestrator can
    retry. The HTTP status is always 200 (the service ITSELF is up);
    callers branch on the `llm.ok` field.
    """
    health_timeout = float(os.environ.get("CQUALITY_HEALTH_TIMEOUT", "5"))
    return {
        "service": "c-quality",
        "version": app.version,
        "llm": probe_llm(health_timeout),
    }
