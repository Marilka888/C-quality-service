"""
PR-K post-fix: persistent SQLite cache for LLM judgments.

Why: on real packages a typical run is 20-30 minutes of LLM time on
qwen2.5:3b. During iteration (debugging false-positives, comparing
prompt variants, re-running after a verifier patch) the same (req,
unit) pair is judged many times with identical input. Caching the
PairJudgment for a hash(req_text, unit_text, model, prompt_version,
backend) lets the second run finish in seconds instead of minutes.

Key design:
  * env-driven enable: `CQUALITY_JUDGE_CACHE_DIR=<path>` activates the
    cache; unset → disabled (same as before).
  * SQLite (stdlib, no extra deps) at <dir>/judgments.db.
  * cache key = sha256(req_text || \\x1f || unit_text || \\x1f || model
    || \\x1f || prompt_version || \\x1f || backend).  \\x1f is the ASCII
    Unit Separator — collision-safe vs spaces in the strings.
  * cache value = pickle(PairJudgment). PairJudgment is a Pydantic
    model; we store its `model_dump_json()` so the cache survives a
    schema-additive change (new optional field → old hits still load
    via Pydantic's default-fill).
  * graceful failure: any cache read/write error → log + bypass cache.
    Never blocks the LLM call.

Invalidation:
  * change PROMPT_VERSION in prompts.py → instant invalidation
    (different cache key).
  * change model_name → instant invalidation (different cache key).
  * If you change the schema of PairJudgment in a non-additive way,
    delete the .cache directory.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit

logger = get_logger(__name__)

# ASCII Unit Separator — collision-safe boundary between hash inputs.
_SEP = "\x1f"

# Schema rev — bump when the table layout changes.
_SCHEMA_REV = 1


def _make_key(
    req_text: str,
    unit_text: str,
    model: str,
    prompt_version: str,
    backend: str,
) -> str:
    """Stable cache key. Whitespace in inputs is preserved (cited
    phrases need exact substring match later)."""
    payload = _SEP.join((req_text, unit_text, model, prompt_version, backend))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgmentCache:
    """SQLite-backed cache of PairJudgment objects.

    Thread-safe: a per-instance Lock guards write access. Reads are
    direct (SQLite handles concurrent reads itself).

    Lifecycle:
      cache = JudgmentCache.from_env(model="qwen2.5:7b", backend="ollama",
                                     prompt_version="v5")
      if cache:
          cached = cache.get(req, unit)
          if cached: return cached
      # ... LLM call ...
      if cache: cache.put(req, unit, judgment)

    `from_env(...)` returns None when the cache dir env var is unset,
    so callers can use `if cache:` as the enable check.
    """

    _ENV_DIR = "CQUALITY_JUDGE_CACHE_DIR"

    def __init__(
        self,
        cache_dir: Path,
        *,
        model: str,
        prompt_version: str,
        backend: str,
    ) -> None:
        self._dir = cache_dir
        self._model = model
        self._prompt_version = prompt_version
        self._backend = backend
        self._lock = threading.Lock()
        self._db_path = self._dir / "judgments.db"
        # `hits` / `misses` for diagnostics (consumed by the pipeline
        # if it wants to log a summary at the end of a run).
        self.hits: int = 0
        self.misses: int = 0
        self._ensure_schema()

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        prompt_version: str,
        backend: str,
    ) -> Optional["JudgmentCache"]:
        """Return a cache instance if the env var is set, else None."""
        cache_dir = os.environ.get(cls._ENV_DIR, "").strip()
        if not cache_dir:
            return None
        try:
            path = Path(cache_dir).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return cls(
                path,
                model=model,
                prompt_version=prompt_version,
                backend=backend,
            )
        except Exception as exc:
            logger.warning(
                "JudgmentCache disabled — could not initialise dir %r: %s",
                cache_dir, exc,
            )
            return None

    # ──────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        # Per-call connection — sqlite3 connections are not thread-safe
        # by default. WAL is on for concurrent reads + serialised writes.
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS judgments (
                        cache_key TEXT PRIMARY KEY,
                        schema_rev INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        backend TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_judgments_model "
                    "ON judgments (model, prompt_version)"
                )
        except Exception as exc:
            logger.warning("JudgmentCache schema init failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────

    def get(
        self, req: RequirementUnit, unit: CoverageUnit,
    ) -> Optional[PairJudgment]:
        """Return cached judgment if present, else None.
        Caller must rebind req_id/unit_id/target_document_id from the
        live req/unit — they are NOT part of the cache key (text +
        model + prompt + backend identify the LLM input)."""
        key = _make_key(
            req_text=req.text,
            unit_text=unit.text,
            model=self._model,
            prompt_version=self._prompt_version,
            backend=self._backend,
        )
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload_json FROM judgments "
                    "WHERE cache_key = ? AND schema_rev = ?",
                    (key, _SCHEMA_REV),
                ).fetchone()
        except Exception as exc:
            logger.warning("JudgmentCache get failed (key=%s…): %s", key[:12], exc)
            return None

        if row is None:
            self.misses += 1
            return None

        try:
            j = PairJudgment.model_validate_json(row[0])
        except Exception as exc:
            # Corrupted entry or schema mismatch — drop it and miss.
            logger.warning(
                "JudgmentCache: corrupted entry key=%s…, removing (%s)",
                key[:12], exc,
            )
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM judgments WHERE cache_key = ?", (key,))
            except Exception:
                pass
            self.misses += 1
            return None

        # Caller's live (req, unit) ids must override the cached values
        # so the judgment refers to the right pair in the current run.
        j.req_id = req.req_id
        j.unit_id = unit.unit_id
        j.target_document_id = unit.target_document_id
        self.hits += 1
        return j

    def put(
        self, req: RequirementUnit, unit: CoverageUnit, judgment: PairJudgment,
    ) -> None:
        """Insert / replace cache entry. Errors are swallowed."""
        key = _make_key(
            req_text=req.text,
            unit_text=unit.text,
            model=self._model,
            prompt_version=self._prompt_version,
            backend=self._backend,
        )
        try:
            payload = judgment.model_dump_json()
        except Exception as exc:
            logger.warning("JudgmentCache put serialise failed: %s", exc)
            return

        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO judgments "
                        "(cache_key, schema_rev, model, prompt_version, backend, payload_json) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            key, _SCHEMA_REV,
                            self._model, self._prompt_version, self._backend,
                            payload,
                        ),
                    )
            except Exception as exc:
                logger.warning(
                    "JudgmentCache put failed (key=%s…): %s", key[:12], exc,
                )

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "model": self._model,
            "prompt_version": self._prompt_version,
            "backend": self._backend,
            "db_path": str(self._db_path),
        }
