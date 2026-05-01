"""
Regression tests for BUG-09: Ollama / LLM unavailable must surface as a
LLM_UNAVAILABLE warning in the result, not be silently swallowed into
DisabledCoverageJudge fallback.
"""
from __future__ import annotations

import pytest
import requests

from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)
from app.infrastructure.llm.ollama_coverage_judge import OllamaCoverageJudge


def _make_req(text: str = "Система должна хранить журнал.") -> RequirementUnit:
    from app.application.use_cases.build_requirements import (
        _extract_constraints,
        _extract_entities,
        _extract_modality,
        _extract_requirement_type,
        _normalize_text,
    )

    return RequirementUnit(
        req_id="req-1",
        source_document_id="doc-tz",
        text=text,
        normalized_text=_normalize_text(text),
        requirement_type=_extract_requirement_type(text),
        modality=_extract_modality(text),
        entities=_extract_entities(text),
        constraints=_extract_constraints(text),
    )


def _make_unit(text: str = "Проверить хранение журнала за 90 суток.") -> CoverageUnit:
    from app.application.use_cases.build_requirements import (
        _extract_constraints,
        _extract_entities,
        _normalize_text,
    )

    return CoverageUnit(
        unit_id="unit-1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        text=text,
        normalized_text=_normalize_text(text),
        entities=_extract_entities(text),
        constraints=_extract_constraints(text),
    )


# ── unit-level: OllamaCoverageJudge counter behaviour ──────────────────────


class TestOllamaJudgeCounter:

    def test_timeout_increments_unavailable_counter(self, monkeypatch):
        judge = OllamaCoverageJudge(model_name="llama3:8b", timeout=1)

        def _raise_timeout(*args, **kwargs):
            raise requests.Timeout("simulated timeout")

        monkeypatch.setattr("requests.post", _raise_timeout)

        result = judge.judge(_make_req(), _make_unit())
        assert isinstance(result, PairJudgment)
        # Fallback judgment: DisabledCoverageJudge labels everything IRRELEVANT.
        assert judge.unavailable_count == 1
        assert "timeout" in judge.last_error.lower()

    def test_connection_error_increments_unavailable_counter(self, monkeypatch):
        judge = OllamaCoverageJudge()

        def _raise_conn(*args, **kwargs):
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr("requests.post", _raise_conn)

        judge.judge(_make_req(), _make_unit())
        assert judge.unavailable_count == 1
        assert "ConnectionError" in judge.last_error

    def test_consume_resets_counter(self, monkeypatch):
        judge = OllamaCoverageJudge()
        monkeypatch.setattr(
            "requests.post",
            lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("x")),
        )
        judge.judge(_make_req(), _make_unit())
        judge.judge(_make_req(), _make_unit())
        count, err = judge.consume_unavailability()
        assert count == 2
        assert err
        # Second call returns zero — counter was reset.
        count2, _ = judge.consume_unavailability()
        assert count2 == 0


# ── integration: pipeline emits LLM_UNAVAILABLE warning ────────────────────


def _make_package(enable_llm: bool) -> dict:
    return {
        "package_id": "pkg-llm",
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi"],
        "documents": [
            {
                "document_id": "doc-tz",
                "doc_role": "tz",
                "prepared_artifact": {
                    "document_id": "doc-tz",
                    "doc_role": "tz",
                    "fragments": [
                        {
                            "fragment_id": "f1",
                            "text": "Система должна хранить журнал не менее 90 дней.",
                            "kind": "paragraph",
                        }
                    ],
                },
            },
            {
                "document_id": "doc-pmi",
                "doc_role": "pmi",
                "prepared_artifact": {
                    "document_id": "doc-pmi",
                    "doc_role": "pmi",
                    "fragments": [
                        {
                            "fragment_id": "f2",
                            "text": "Проверить хранение журнала за 90 суток.",
                            "kind": "test_step",
                        }
                    ],
                },
            },
        ],
        "options": {
            "top_k": 5,
            "enable_llm_judge": enable_llm,
            "judge_backend": "ollama" if enable_llm else "",
            "enable_rule_verification": False,
            "min_retrieval_score": 0.0,
        },
    }


class TestPipelineLLMUnavailableWarning:

    def test_warning_emitted_when_ollama_times_out(self, monkeypatch):
        """Run() with LLM enabled and Ollama unavailable must add an
        LLM_UNAVAILABLE warning to result.warnings."""

        def _raise(*args, **kwargs):
            raise requests.Timeout("simulated outage")

        monkeypatch.setattr("requests.post", _raise)

        pipeline = CoverageAnalysisPipeline()
        result = pipeline.run(_make_package(enable_llm=True))

        ll_warnings = [w for w in result.warnings if "LLM_UNAVAILABLE" in w]
        assert ll_warnings, (
            f"expected LLM_UNAVAILABLE in warnings, got: {result.warnings!r}"
        )
        # Warning should mention pair count and the underlying error.
        msg = ll_warnings[0]
        assert "judge backend fell back" in msg or "pair" in msg
        assert "simulated outage" in msg or "timeout" in msg.lower()

    def test_no_warning_when_llm_disabled_by_config(self):
        """If llm.enabled is False (default), the judge is DisabledCoverageJudge
        by design and IRRELEVANT is the expected output. No warning here.
        """
        pipeline = CoverageAnalysisPipeline()
        result = pipeline.run(_make_package(enable_llm=False))

        ll_warnings = [w for w in result.warnings if "LLM_UNAVAILABLE" in w]
        assert not ll_warnings, (
            f"unexpected LLM_UNAVAILABLE warning when LLM disabled: {result.warnings!r}"
        )
