"""
PR-K post-fix regression tests for JudgmentCache.

Cache contract:
  * disabled by default (CQUALITY_JUDGE_CACHE_DIR unset → from_env returns None)
  * enabled by setting the env var to a directory path
  * key = sha256(req.text + unit.text + model + prompt_version + backend)
  * different model / prompt_version / backend → different key (no collision)
  * stored payload survives a re-open of the cache (persistence)
  * cache rebinds req_id/unit_id/target_document_id on read so the
    judgment refers to the right pair in the current run
  * any error during read/write is silently logged and falls through
    to a fresh LLM call (never blocks)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.judgment_cache import (
    JudgmentCache,
    _make_key,
)


def _req(text: str = "Система должна логировать действия пользователя.",
         req_id: str = "r1") -> RequirementUnit:
    return RequirementUnit(
        req_id=req_id, source_document_id="doc-tz",
        text=text, normalized_text=text.lower(),
    )


def _unit(text: str = "Система записывает действия пользователя в журнал.",
          unit_id: str = "u1") -> CoverageUnit:
    return CoverageUnit(
        unit_id=unit_id, target_document_id="doc-pmi", target_doc_role="pmi",
        text=text, normalized_text=text.lower(),
    )


def _judgment(label: LLMLabel = LLMLabel.COVERED, conf: float = 0.85) -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=label, rule_adjusted_label=label, llm_confidence=conf,
        cited_phrases=["записывает действия"],
        explanation="Same idea on both sides.",
    )


# ── from_env enable / disable ────────────────────────────────────────────


class TestEnvDrivenEnable:
    def test_disabled_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("CQUALITY_JUDGE_CACHE_DIR", raising=False)
        cache = JudgmentCache.from_env(
            model="qwen2.5:3b", prompt_version="v5", backend="ollama",
        )
        assert cache is None

    def test_disabled_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("CQUALITY_JUDGE_CACHE_DIR", "")
        cache = JudgmentCache.from_env(
            model="qwen2.5:3b", prompt_version="v5", backend="ollama",
        )
        assert cache is None

    def test_enabled_when_env_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CQUALITY_JUDGE_CACHE_DIR", str(tmp_path))
        cache = JudgmentCache.from_env(
            model="qwen2.5:3b", prompt_version="v5", backend="ollama",
        )
        assert cache is not None
        assert (tmp_path / "judgments.db").exists()


# ── Hit / miss / persistence ─────────────────────────────────────────────


class TestHitMissPersistence:
    def setup_method(self):
        self.tmp = Path()  # filled in tests via tmp_path
        self.cache: JudgmentCache | None = None

    def _cache(self, tmp_path: Path, **kw) -> JudgmentCache:
        defaults = dict(model="qwen2.5:3b", prompt_version="v5", backend="ollama")
        defaults.update(kw)
        return JudgmentCache(tmp_path, **defaults)

    def test_get_returns_none_on_miss(self, tmp_path):
        cache = self._cache(tmp_path)
        out = cache.get(_req(), _unit())
        assert out is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_put_then_get_returns_cached(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put(_req(), _unit(), _judgment())
        out = cache.get(_req(), _unit())
        assert out is not None
        assert out.llm_label == LLMLabel.COVERED
        assert out.llm_confidence == pytest.approx(0.85)
        assert cache.hits == 1

    def test_persists_across_instances(self, tmp_path):
        cache1 = self._cache(tmp_path)
        cache1.put(_req(), _unit(), _judgment())
        # Fresh instance, same dir.
        cache2 = self._cache(tmp_path)
        out = cache2.get(_req(), _unit())
        assert out is not None
        assert out.llm_label == LLMLabel.COVERED

    def test_get_rebinds_ids_to_live_pair(self, tmp_path):
        """When a cached judgment is returned, its req_id/unit_id/
        target_document_id must reflect the LIVE pair, not whatever was
        cached. Otherwise cross-package re-runs would carry stale IDs."""
        cache = self._cache(tmp_path)
        cache.put(
            _req(text="X", req_id="OLD_R"),
            _unit(text="Y", unit_id="OLD_U"),
            _judgment(),
        )
        # Same TEXT (cache key), DIFFERENT live ids.
        live_req = _req(text="X", req_id="NEW_R")
        live_unit = _unit(text="Y", unit_id="NEW_U")
        out = cache.get(live_req, live_unit)
        assert out is not None
        assert out.req_id == "NEW_R"
        assert out.unit_id == "NEW_U"


# ── Key collision-safety ─────────────────────────────────────────────────


class TestKeyCollisionSafety:
    def test_different_model_different_key(self):
        k1 = _make_key("a", "b", "qwen2.5:3b", "v5", "ollama")
        k2 = _make_key("a", "b", "qwen2.5:7b", "v5", "ollama")
        assert k1 != k2

    def test_different_prompt_version_different_key(self):
        k1 = _make_key("a", "b", "m", "v5", "ollama")
        k2 = _make_key("a", "b", "m", "v6", "ollama")
        assert k1 != k2

    def test_different_backend_different_key(self):
        k1 = _make_key("a", "b", "m", "v", "ollama")
        k2 = _make_key("a", "b", "m", "v", "litellm")
        assert k1 != k2

    def test_swapped_req_unit_different_key(self):
        """The pair (req='X', unit='Y') is NOT the same as (req='Y',
        unit='X') — one says req covers unit, the other vice-versa.
        Different cache keys must be produced."""
        k1 = _make_key("X", "Y", "m", "v", "b")
        k2 = _make_key("Y", "X", "m", "v", "b")
        assert k1 != k2

    def test_concat_collision_guard(self):
        """Without the unit-separator, ('foo', 'bar') would have the
        same SHA as ('foobar', '') after concat. The \\x1f separator
        prevents this."""
        k1 = _make_key("foo", "bar", "m", "v", "b")
        k2 = _make_key("foobar", "", "m", "v", "b")
        assert k1 != k2


# ── Cross-model invalidation via env-driven cache ────────────────────────


class TestCrossModelInvalidation:
    """Different models / prompt versions must produce different cache
    entries — invalidation is automatic on env / config change."""

    def _cache(self, tmp_path: Path, model: str, prompt_version: str,
               backend: str = "ollama") -> JudgmentCache:
        return JudgmentCache(
            tmp_path, model=model, prompt_version=prompt_version, backend=backend,
        )

    def test_different_model_separate_entry(self, tmp_path):
        c3b = self._cache(tmp_path, "qwen2.5:3b", "v5")
        c7b = self._cache(tmp_path, "qwen2.5:7b", "v5")
        c3b.put(_req(), _unit(), _judgment(LLMLabel.PARTIAL, 0.5))
        # 7B must miss (different cache key).
        assert c7b.get(_req(), _unit()) is None

    def test_different_prompt_version_separate_entry(self, tmp_path):
        c5 = self._cache(tmp_path, "qwen2.5:3b", "v5")
        c6 = self._cache(tmp_path, "qwen2.5:3b", "v6")
        c5.put(_req(), _unit(), _judgment())
        assert c6.get(_req(), _unit()) is None


# ── Graceful failure ──────────────────────────────────────────────────────


class TestGracefulFailure:
    def test_corrupted_payload_returns_miss(self, tmp_path, caplog):
        """When cache row exists but payload won't deserialise (schema
        bumped, manual edit, …), get() returns None and removes the
        bad row — never raises."""
        cache = JudgmentCache(
            tmp_path, model="m", prompt_version="v5", backend="ollama",
        )
        # Corrupt the row directly via sqlite.
        import sqlite3
        with sqlite3.connect(str(tmp_path / "judgments.db")) as conn:
            conn.execute(
                "INSERT INTO judgments "
                "(cache_key, schema_rev, model, prompt_version, backend, payload_json) "
                "VALUES (?, 1, ?, ?, ?, ?)",
                (
                    _make_key(_req().text, _unit().text, "m", "v5", "ollama"),
                    "m", "v5", "ollama",
                    "{not valid json",
                ),
            )
        # First read: corrupted entry → log + miss.
        out = cache.get(_req(), _unit())
        assert out is None
        # Second read: corrupted entry was removed → still miss, no errors.
        out = cache.get(_req(), _unit())
        assert out is None

    def test_put_swallow_serialise_error(self, tmp_path, monkeypatch):
        """If model_dump_json fails for some weird reason, put() must
        log + return without raising."""
        cache = JudgmentCache(
            tmp_path, model="m", prompt_version="v5", backend="ollama",
        )

        class BrokenJudgment:
            def model_dump_json(self):
                raise RuntimeError("simulated dump failure")
            # required attributes for type-hint compatibility:
            req_id = "r"; unit_id = "u"; target_document_id = "d"

        # Should not raise.
        cache.put(_req(), _unit(), BrokenJudgment())  # type: ignore[arg-type]
