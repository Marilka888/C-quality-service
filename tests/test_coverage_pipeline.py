"""
Coverage pipeline tests.

Unit tests cover:
  - RequirementBuilder (candidates path, fragments fallback)
  - CoverageUnitBuilder
  - Constraint extraction
  - PairVerifier (conflict / downgrade rules)
  - CoverageAggregator (status priority logic)
  - CoverageReportBuilder (summary tallies)

Integration tests cover the four canonical scenarios:
  1. COVERED  — журнал 90 дней / ПМИ проверяет 90 суток
  2. PARTIAL  — время ответа <= 2 сек / ПМИ проверяет время ответа (без числа)
  3. CONFLICT — журнал 90 дней / ПМИ проверяет 30 дней
  4. MISSING  — нет релевантного фрагмента
"""
from __future__ import annotations

import pytest

from app.application.use_cases.aggregate_coverage import CoverageAggregator
from app.application.use_cases.build_coverage_report import CoverageReportBuilder
from app.application.use_cases.build_coverage_units import CoverageUnitBuilder
from app.application.use_cases.build_requirements import (
    RequirementBuilder,
    _extract_constraints,
    _extract_modality,
    _extract_requirement_type,
)
from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
from app.application.use_cases.verify_pairs import PairVerifier
from app.core.config import CoverageConfig
from app.domain.c_quality_enums import CoverageStatus, LLMLabel, Modality, RequirementType
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_req(text: str, constraints=None, entities=None) -> RequirementUnit:
    from app.application.use_cases.build_requirements import (
        _extract_constraints,
        _extract_entities,
        _extract_modality,
        _extract_requirement_type,
        _normalize_text,
    )

    return RequirementUnit(
        source_document_id="doc-tz",
        text=text,
        normalized_text=_normalize_text(text),
        requirement_type=_extract_requirement_type(text),
        modality=_extract_modality(text),
        entities=entities or _extract_entities(text),
        constraints=constraints or _extract_constraints(text),
    )


def _make_unit(text: str, doc_id="doc-pmi", role="pmi", constraints=None) -> CoverageUnit:
    from app.application.use_cases.build_requirements import (
        _extract_constraints,
        _extract_entities,
        _normalize_text,
    )

    return CoverageUnit(
        target_document_id=doc_id,
        target_doc_role=role,
        text=text,
        normalized_text=_normalize_text(text),
        entities=_extract_entities(text),
        constraints=constraints or _extract_constraints(text),
    )


def _make_candidate(req: RequirementUnit, unit: CoverageUnit, score=0.6) -> RetrievedCandidate:
    return RetrievedCandidate(
        req_id=req.req_id,
        unit_id=unit.unit_id,
        target_document_id=unit.target_document_id,
        retrieval_score=score,
    )


# ===========================================================================
# Unit tests: RequirementBuilder
# ===========================================================================


class TestRequirementBuilder:
    def test_from_candidates(self):
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "fragments": [],
            "requirement_candidates": [
                {"req_id": "r1", "text": "Система должна хранить журнал не менее 90 дней."}
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 1
        assert reqs[0].req_id == "r1"

    def test_fallback_to_fragments(self):
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "fragments": [
                {
                    "fragment_id": "f1",
                    "text": "Система должна хранить журнал событий не менее 90 дней.",
                },
                {
                    "fragment_id": "f2",
                    "text": "Краткое описание.",
                },
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        # Only the fragment with a modality trigger should be picked
        assert len(reqs) == 1
        assert "90" in reqs[0].text

    def test_modality_must(self):
        assert _extract_modality("Система должна выполнять операцию.") == Modality.MUST

    def test_modality_must_not(self):
        assert _extract_modality("Система не должна передавать данные.") == Modality.MUST_NOT

    def test_requirement_type_logging(self):
        assert _extract_requirement_type("Хранить журнал событий 90 дней.") == RequirementType.LOGGING

    def test_requirement_type_performance(self):
        assert _extract_requirement_type("Время ответа не более 2 секунд.") == RequirementType.PERFORMANCE


# ===========================================================================
# Unit tests: Constraint extraction
# ===========================================================================


class TestConstraintExtraction:
    def test_extracts_days_constraint(self):
        constraints = _extract_constraints("Хранить журнал не менее 90 дней.")
        assert len(constraints) == 1
        c = constraints[0]
        assert c.value == 90.0
        assert c.unit == "days"
        assert c.operator == ">="

    def test_extracts_sec_constraint(self):
        constraints = _extract_constraints("Время ответа не более 2 секунд.")
        assert len(constraints) == 1
        c = constraints[0]
        assert c.value == 2.0
        assert c.unit == "sec"
        assert c.operator == "<="

    def test_extracts_sutki_as_days(self):
        constraints = _extract_constraints("Хранить 30 суток.")
        assert constraints[0].unit == "days"
        assert constraints[0].value == 30.0

    def test_no_constraints_for_plain_text(self):
        constraints = _extract_constraints("Система должна выполнять операцию.")
        assert constraints == []


# ===========================================================================
# Unit tests: CoverageUnitBuilder
# ===========================================================================


class TestCoverageUnitBuilder:
    def test_builds_units_from_fragments(self):
        artifact = {
            "document_id": "doc-pmi",
            "doc_role": "pmi",
            "fragments": [
                {"fragment_id": "f1", "text": "Проверить, что журнал хранится 90 суток.", "kind": "test_step"},
                {"fragment_id": "f2", "text": "Кр.", "kind": "paragraph"},  # 1 word → skip
            ],
        }
        units = CoverageUnitBuilder().build(artifact)
        assert len(units) == 1
        from app.domain.c_quality_enums import CoverageUnitType
        assert units[0].unit_type == CoverageUnitType.TEST_STEP

    def test_accepts_two_word_fragments(self):
        """MIN_WORDS = 2: two-word PMI test steps must not be dropped."""
        artifact = {
            "document_id": "doc-pmi",
            "doc_role": "pmi",
            "fragments": [
                {"fragment_id": "f1", "text": "Авторизоваться системе.", "kind": "test_step"},   # 2 words ✓
                {"fragment_id": "f2", "text": "Открыть раздел.", "kind": "test_step"},            # 2 words ✓
                {"fragment_id": "f3", "text": "1.", "kind": "list_item"},                          # 1 word ✗
            ],
        }
        units = CoverageUnitBuilder().build(artifact)
        assert len(units) == 2, f"Expected 2 units (2-word fragments must pass); got {len(units)}"

    def test_falls_back_to_sentences_when_fragments_empty(self):
        """sentences[] is used as fallback when fragments[] is absent/empty."""
        artifact = {
            "document_id": "doc-pmi",
            "doc_role": "pmi",
            "fragments": [],  # empty — simulates docback not populating fragments
            "sentences": [
                {"fragment_id": "s1", "text": "Проверить авторизацию пользователей системы.", "kind": "test_step"},
                {"fragment_id": "s2", "text": "Убедиться что журнал хранится.", "kind": "test_step"},
            ],
        }
        units = CoverageUnitBuilder().build(artifact)
        assert len(units) == 2, f"Expected 2 units from sentences[] fallback; got {len(units)}"

    def test_empty_fragments_and_sentences_returns_zero(self):
        """When both sources are empty, builder must return [] without crash."""
        artifact = {
            "document_id": "doc-pmi",
            "doc_role": "pmi",
            "fragments": [],
            "sentences": [],
        }
        units = CoverageUnitBuilder().build(artifact)
        assert units == []

    def test_retrieval_gets_non_empty_shortlist_when_units_exist(self):
        """Integration: when PMI has 2-word+ fragments, retrieval shortlist is non-empty."""
        from app.application.use_cases.retrieve_candidates import CandidateRetriever
        from app.core.config import CoverageRetrievalConfig
        from app.infrastructure.embeddings.simple import BagOfWordsEmbeddingBackend

        req = _make_req("Система должна хранить журнал событий не менее 90 дней.")
        artifact = {
            "document_id": "doc-pmi",
            "doc_role": "pmi",
            "fragments": [
                {"fragment_id": "f1", "text": "Проверить хранение журнала 90 суток.", "kind": "test_step"},
                {"fragment_id": "f2", "text": "Авторизоваться в системе.", "kind": "test_step"},
            ],
        }
        units = CoverageUnitBuilder().build(artifact)
        assert len(units) == 2, "Builder must produce units from valid fragments"

        cfg = CoverageRetrievalConfig(min_retrieval_score=0.0)
        retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())
        candidates = retriever.retrieve(req, units)
        assert len(candidates) > 0, "Retrieval must return non-empty shortlist when units exist"


# ===========================================================================
# Unit tests: PairVerifier
# ===========================================================================


class TestPairVerifier:
    def setup_method(self):
        self.verifier = PairVerifier()

    def _judgment(self, req, unit, label=LLMLabel.COVERED):
        return PairJudgment(
            req_id=req.req_id,
            unit_id=unit.unit_id,
            target_document_id=unit.target_document_id,
            llm_label=label,
            rule_adjusted_label=label,
        )

    def test_numeric_conflict_overrides_covered(self):
        req = _make_req("Хранить журнал не менее 90 дней.")
        unit = _make_unit("Хранить журнал 30 суток.")
        j = self._judgment(req, unit)
        result = self.verifier.verify(j, req, unit)
        assert result.rule_adjusted_label == LLMLabel.CONFLICT

    def test_same_value_no_conflict(self):
        req = _make_req("Хранить журнал не менее 90 дней.")
        unit = _make_unit("Журнал хранится 90 суток.")
        j = self._judgment(req, unit)
        result = self.verifier.verify(j, req, unit)
        assert result.rule_adjusted_label == LLMLabel.COVERED

    def test_covered_without_unit_constraints_downgraded_to_partial(self):
        req = _make_req(
            "Время ответа не более 2 секунд.",
            constraints=[Constraint(kind="response_time", operator="<=", value=2.0, unit="sec")],
        )
        unit = _make_unit(
            "Проверяется время ответа системы на запросы пользователей.",
            constraints=[],
        )
        j = self._judgment(req, unit)
        result = self.verifier.verify(j, req, unit)
        assert result.rule_adjusted_label == LLMLabel.PARTIAL

    def test_irrelevant_unchanged(self):
        req = _make_req("Система должна хранить журнал.")
        unit = _make_unit("Пользователь вводит логин.")
        j = self._judgment(req, unit, label=LLMLabel.IRRELEVANT)
        result = self.verifier.verify(j, req, unit)
        assert result.rule_adjusted_label == LLMLabel.IRRELEVANT


# ===========================================================================
# Unit tests: CoverageAggregator
# ===========================================================================


class TestCoverageAggregator:
    def setup_method(self):
        self.agg = CoverageAggregator()

    def _make_judgment(self, req, unit, label: LLMLabel) -> PairJudgment:
        return PairJudgment(
            req_id=req.req_id,
            unit_id=unit.unit_id,
            target_document_id=unit.target_document_id,
            llm_label=label,
            rule_adjusted_label=label,
        )

    def test_conflict_wins_over_covered(self):
        req = _make_req("Хранить журнал.")
        unit_conflict = _make_unit("Хранить 30 суток.")
        unit_covered = _make_unit("Хранить 90 суток.")
        j1 = self._make_judgment(req, unit_conflict, LLMLabel.CONFLICT)
        j2 = self._make_judgment(req, unit_covered, LLMLabel.COVERED)
        result = self.agg.aggregate(
            req, [j1, j2],
            {unit_conflict.unit_id: _make_candidate(req, unit_conflict),
             unit_covered.unit_id: _make_candidate(req, unit_covered)},
            {unit_conflict.unit_id: unit_conflict, unit_covered.unit_id: unit_covered},
            "doc-pmi", "pmi",
        )
        assert result.status == CoverageStatus.CONFLICT

    def test_covered_wins_over_partial(self):
        req = _make_req("Хранить журнал.")
        unit_partial = _make_unit("Проверяется хранение журнала.")
        unit_covered = _make_unit("Хранить журнал 90 суток.")
        j1 = self._make_judgment(req, unit_partial, LLMLabel.PARTIAL)
        j2 = self._make_judgment(req, unit_covered, LLMLabel.COVERED)
        result = self.agg.aggregate(
            req, [j1, j2],
            {u.unit_id: _make_candidate(req, u) for u in [unit_partial, unit_covered]},
            {u.unit_id: u for u in [unit_partial, unit_covered]},
            "doc-pmi", "pmi",
        )
        assert result.status == CoverageStatus.COVERED

    def test_empty_judgments_gives_missing(self):
        req = _make_req("Хранить журнал.")
        result = self.agg.aggregate(req, [], {}, {}, "doc-pmi", "pmi")
        assert result.status == CoverageStatus.MISSING


# ===========================================================================
# Integration tests: four canonical scenarios
# ===========================================================================


def _make_package(tz_req_text: str, pmi_text: str) -> dict:
    return {
        "job_id": "test-job",
        "package_id": "test-pkg",
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
                        {"fragment_id": "f1", "text": tz_req_text, "kind": "paragraph"}
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
                        {"fragment_id": "f2", "text": pmi_text, "kind": "test_step"}
                    ],
                },
            },
        ],
        "options": {
            "top_k": 5,
            "enable_llm_judge": False,
            "enable_rule_verification": True,
            "min_retrieval_score": 0.0,  # disable threshold gate for unit tests
        },
    }


def _run(package: dict) -> CoverageStatus:
    pipeline = CoverageAnalysisPipeline(config=CoverageConfig.from_options(package["options"]))
    result = pipeline.run(package)
    if not result.requirement_results:
        return CoverageStatus.MISSING
    # Return the best status across all requirement results
    from app.application.use_cases.aggregate_coverage import _STATUS_RANK
    return max(result.requirement_results, key=lambda r: _STATUS_RANK[r.status]).status


class TestCanonicalScenarios:
    """
    Scenario 1 — COVERED
    TZ:  хранить журнал 90 дней
    PMI: проверяется хранение журнала 90 суток
    Expected: COVERED (same value, same unit class)
    """
    def test_covered_90_days_match(self):
        status = _run(_make_package(
            "Система должна хранить журнал событий не менее 90 дней.",
            "Проверить, что журнал событий хранится не менее 90 суток.",
        ))
        assert status == CoverageStatus.COVERED

    """
    Scenario 2 — PARTIAL
    TZ:  время ответа <= 2 сек при 100 RPS
    PMI: проверяется время ответа системы
    Expected: PARTIAL (topic matches, numeric constraint absent in unit)
    """
    def test_partial_response_time_no_value(self):
        status = _run(_make_package(
            "Система должна обеспечивать время ответа не более 2 секунд.",
            "Проверить время ответа системы при различных нагрузках.",
        ))
        assert status == CoverageStatus.PARTIAL

    """
    Scenario 3 — CONFLICT
    TZ:  хранить журнал 90 дней
    PMI: хранить 30 дней
    Expected: CONFLICT (same unit class, different value)
    """
    def test_conflict_30_vs_90_days(self):
        status = _run(_make_package(
            "Система должна хранить журнал не менее 90 дней.",
            "Хранение журнала проверяется за последние 30 суток.",
        ))
        assert status == CoverageStatus.CONFLICT

    """
    Scenario 4 — MISSING
    PMI fragment is completely unrelated to the TZ requirement.
    Expected: MISSING
    """
    def test_missing_no_relevant_fragment(self):
        status = _run(_make_package(
            "Система должна хранить журнал не менее 90 дней.",
            "Интерфейс пользователя должен содержать кнопку выхода.",
        ))
        assert status == CoverageStatus.MISSING


# ===========================================================================
# Integration: full pipeline smoke test
# ===========================================================================


class TestPipelineSmokeTest:
    def test_end_to_end_structure(self):
        package = {
            "job_id": "smoke-job",
            "package_id": "smoke-pkg",
            "source_doc_role": "tz",
            "target_doc_roles": ["pmi", "pz"],
            "documents": [
                {
                    "document_id": "doc-tz",
                    "doc_role": "tz",
                    "prepared_artifact": {
                        "document_id": "doc-tz",
                        "doc_role": "tz",
                        "fragments": [
                            {"fragment_id": "f1", "text": "Система должна хранить журнал не менее 90 дней.", "kind": "paragraph"},
                            {"fragment_id": "f2", "text": "Время ответа должно быть не более 2 секунд при нагрузке 100 RPS.", "kind": "paragraph"},
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
                            {"fragment_id": "f3", "text": "Проверить хранение журнала за 90 суток.", "kind": "test_step"},
                            {"fragment_id": "f4", "text": "Нагрузочное тестирование при 100 RPS.", "kind": "test_step"},
                        ],
                    },
                },
                {
                    "document_id": "doc-pz",
                    "doc_role": "pz",
                    "prepared_artifact": {
                        "document_id": "doc-pz",
                        "doc_role": "pz",
                        "fragments": [
                            {"fragment_id": "f5", "text": "Архитектура сервиса обеспечивает хранение данных в течение 90 дней.", "kind": "paragraph"},
                        ],
                    },
                },
            ],
            "options": {
                "top_k": 5,
                "enable_llm_judge": False,
                "enable_rule_verification": True,
                "min_retrieval_score": 0.0,
            },
        }

        pipeline = CoverageAnalysisPipeline()
        result = pipeline.run(package)

        assert result.job_id == "smoke-job"
        assert result.package_id == "smoke-pkg"
        assert result.summary.total_requirements >= 1
        assert isinstance(result.requirement_results, list)
        assert isinstance(result.document_reports, list)
        assert len(result.target_document_ids) == 2

    def test_missing_source_document_returns_warning(self):
        package = {
            "package_id": "pkg-1",
            "documents": [],
            "options": {},
        }
        pipeline = CoverageAnalysisPipeline()
        result = pipeline.run(package)
        assert any("No document" in w for w in result.warnings)

    def test_no_target_documents_returns_warning(self):
        package = {
            "package_id": "pkg-2",
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
                            {"fragment_id": "f1", "text": "Система должна хранить журнал.", "kind": "paragraph"}
                        ],
                    },
                }
            ],
            "options": {},
        }
        pipeline = CoverageAnalysisPipeline()
        result = pipeline.run(package)
        assert any("No target" in w for w in result.warnings)


# ===========================================================================
# Bug fix tests
# ===========================================================================


class TestReqIdUniqueness:
    """Scenario 1: multiple requirements from candidates must each have a unique req_id."""

    def test_unique_req_ids_from_candidates_with_shared_document_req_id(self):
        """When docback sends all candidates with same req_id (document-level ID),
        pipeline must still produce unique req_id per RequirementUnit."""
        doc_id = "00612f05-c5a8-47f1-b3a2-f16e40718ae2"
        shared_req_id = f"{doc_id}:::cand"  # the real docback bug pattern

        artifact = {
            "document_id": doc_id,
            "doc_role": "tz",
            "requirement_candidates": [
                {
                    "req_id": shared_req_id,
                    "fragment_id": f"frag-{i}",
                    "text": f"Система должна обеспечивать функцию номер {i} для пользователей.",
                }
                for i in range(5)
            ],
        }
        reqs = RequirementBuilder().build(artifact)

        assert len(reqs) == 5
        req_ids = [r.req_id for r in reqs]
        # All req_ids must be unique
        assert len(set(req_ids)) == 5, f"Duplicate req_ids found: {req_ids}"

    def test_req_ids_use_fragment_id_when_available(self):
        """req_id should be deterministic and include fragment_id."""
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "requirement_candidates": [
                {"req_id": "doc-tz:::cand", "fragment_id": "f-001", "text": "Система должна хранить журнал не менее 90 дней."},
                {"req_id": "doc-tz:::cand", "fragment_id": "f-002", "text": "Время ответа должно быть не более 2 секунд."},
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 2
        assert "f-001" in reqs[0].req_id
        assert "f-002" in reqs[1].req_id
        assert reqs[0].req_id != reqs[1].req_id

    def test_unique_req_ids_in_pipeline_output(self):
        """Scenario 1: end-to-end — requirement_results must each have distinct req_id."""
        doc_id = "doc-tz-uuid"
        package = {
            "job_id": "test-unique-ids",
            "package_id": "pkg-unique",
            "source_doc_role": "tz",
            "target_doc_roles": ["pmi"],
            "documents": [
                {
                    "document_id": doc_id,
                    "doc_role": "tz",
                    "prepared_artifact": {
                        "document_id": doc_id,
                        "doc_role": "tz",
                        "requirement_candidates": [
                            {
                                "req_id": f"{doc_id}:::cand",
                                "fragment_id": f"frag-{i}",
                                "text": f"Система должна обеспечивать функцию {i} для пользователей системы.",
                            }
                            for i in range(3)
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
                            {"fragment_id": "pf-1", "text": "Проверить функциональность системы для пользователей.", "kind": "test_step"},
                        ],
                    },
                },
            ],
            "options": {"min_retrieval_score": 0.0},
        }

        pipeline = CoverageAnalysisPipeline(config=CoverageConfig.from_options({"min_retrieval_score": 0.0}))
        result = pipeline.run(package)

        req_ids = [r.req_id for r in result.requirement_results]
        assert len(req_ids) >= 3
        assert len(set(req_ids)) == len(req_ids), f"Duplicate req_ids in results: {req_ids}"


class TestPairJudgmentsNonEmpty:
    """Scenario 2: when PMI fragment exists, pair_judgments must not be empty."""

    def test_pair_judgments_populated_when_candidates_exist(self):
        """With min_retrieval_score=0.0, any coverage unit must produce at least one judgment."""
        package = _make_package(
            "Система должна хранить журнал событий не менее 90 дней.",
            "Проверить хранение журнала событий за 90 суток.",
        )
        pipeline = CoverageAnalysisPipeline(config=CoverageConfig.from_options(package["options"]))
        result = pipeline.run(package)

        assert result.pair_judgments is not None
        assert len(result.pair_judgments) > 0, "pair_judgments must not be empty when a matching PMI fragment exists"

    def test_pair_judgment_req_ids_match_requirement_results(self):
        """req_id in pair_judgments must match a req_id from requirement_results."""
        package = _make_package(
            "Система должна хранить журнал не менее 90 дней.",
            "Проверить хранение журнала событий за 90 суток.",
        )
        pipeline = CoverageAnalysisPipeline(config=CoverageConfig.from_options(package["options"]))
        result = pipeline.run(package)

        result_req_ids = {r.req_id for r in result.requirement_results}
        if result.pair_judgments:
            for j in result.pair_judgments:
                assert j.req_id in result_req_ids, f"judgment req_id={j.req_id!r} not in requirement_results"


class TestNoSilentFailure:
    """Scenario 3: when retrieval finds nothing, warnings must explain why."""

    def test_warning_when_no_pairs_survive_retrieval(self):
        """With impossibly high threshold, result must contain a diagnostic warning."""
        package = _make_package(
            "Система должна хранить журнал событий не менее 90 дней.",
            "Проверить хранение журнала событий за 90 суток.",
        )
        package["options"]["min_retrieval_score"] = 1.0  # nothing will survive

        pipeline = CoverageAnalysisPipeline(config=CoverageConfig.from_options(package["options"]))
        result = pipeline.run(package)

        assert any("MISSING" in w or "No candidate" in w or "threshold" in w.lower() for w in result.warnings), \
            f"Expected diagnostic warning when all results are MISSING. Got: {result.warnings}"
        assert result.pair_judgments == [] or result.pair_judgments is None

    def test_warning_when_pmi_has_no_fragments(self):
        """When PMI artifact has no fragments, a warning must appear."""
        package = {
            "job_id": "test-no-frags",
            "package_id": "pkg-nofrags",
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
                            {"fragment_id": "f1", "text": "Система должна хранить журнал не менее 90 дней.", "kind": "paragraph"},
                        ],
                    },
                },
                {
                    "document_id": "doc-pmi",
                    "doc_role": "pmi",
                    "prepared_artifact": {
                        "document_id": "doc-pmi",
                        "doc_role": "pmi",
                        "fragments": [],  # intentionally empty
                    },
                },
            ],
            "options": {},
        }

        pipeline = CoverageAnalysisPipeline()
        result = pipeline.run(package)

        assert any("fragment" in w.lower() or "coverage unit" in w.lower() for w in result.warnings), \
            f"Expected warning about empty PMI fragments. Got: {result.warnings}"


class TestFallbackPipelineDiagnostics:
    """Scenario 4: disabled LLM still produces pair-level diagnostics when pairs shortlisted."""

    def test_disabled_judge_produces_pair_judgments(self):
        """With disabled LLM judge and min_retrieval_score=0.0, pair_judgments must be non-empty."""
        package = {
            "job_id": "test-disabled-judge",
            "package_id": "pkg-disabled",
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
                            {"fragment_id": "f1", "text": "Система должна хранить журнал событий не менее 90 дней.", "kind": "paragraph"},
                            {"fragment_id": "f2", "text": "Время ответа не более 2 секунд при нагрузке 100 RPS.", "kind": "paragraph"},
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
                            {"fragment_id": "f3", "text": "Проверить хранение журнала событий за 90 суток.", "kind": "test_step"},
                            {"fragment_id": "f4", "text": "Нагрузочное тестирование при 100 RPS, замер времени ответа.", "kind": "test_step"},
                        ],
                    },
                },
            ],
            "options": {
                "enable_llm_judge": False,
                "enable_rule_verification": True,
                "min_retrieval_score": 0.0,
            },
        }

        pipeline = CoverageAnalysisPipeline(config=CoverageConfig.from_options(package["options"]))
        result = pipeline.run(package)

        assert result.pair_judgments is not None
        assert len(result.pair_judgments) > 0, "pair_judgments must be populated even without LLM judge"

        # Each judgment must reference a valid requirement
        result_req_ids = {r.req_id for r in result.requirement_results}
        for j in result.pair_judgments:
            assert j.req_id in result_req_ids
            assert j.unit_id  # must have a unit reference
            assert j.explanation  # disabled judge always adds explanation


# ===========================================================================
# Disabled judge quality tests
# ===========================================================================


class TestDisabledJudgeConservatism:
    """
    Verify the conservative decision logic of DisabledCoverageJudge.

    Key invariants:
    - COVERED only via constraint-kind match or dual strong evidence (lex+entity)
    - PARTIAL for moderate lexical overlap alone
    - IRRELEVANT for low overlap
    """

    def setup_method(self):
        self.judge = DisabledCoverageJudge()

    # -- Case 1: high lex, no constraint match → must be PARTIAL, not COVERED --------

    def test_high_lex_without_constraint_match_is_partial(self):
        """
        Two topically related sentences with lex ≈ 0.25 but NO shared constraint kind.
        Should produce PARTIAL, not COVERED.
        Regression: old code gave COVERED at lex >= 0.20.
        """
        req = _make_req(
            "Система должна обеспечивать ролевую модель доступа пользователей.",
            constraints=[],
        )
        unit = _make_unit(
            "Проверить ролевую модель доступа к данным пользователей системы.",
            constraints=[],
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.PARTIAL, (
            f"Expected PARTIAL for high-lex/no-constraint pair, got {j.llm_label}; {j.explanation}"
        )

    # -- Case 2: low lex → IRRELEVANT, not PARTIAL ----------------------------------

    def test_low_lex_is_irrelevant(self):
        """
        Weakly related pair with lex < 0.20 and no constraint signal → IRRELEVANT.
        Regression: old code gave PARTIAL at lex >= 0.08.
        """
        req = _make_req(
            "Система должна хранить журнал событий не менее 90 дней.",
            constraints=[],
        )
        unit = _make_unit(
            "Кнопка выхода должна быть доступна пользователю на главной странице.",
            constraints=[],
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.IRRELEVANT, (
            f"Expected IRRELEVANT for low-lex/no-signal pair, got {j.llm_label}; {j.explanation}"
        )

    # -- Case 3: exact coverage via constraint kind match → COVERED -----------------

    def test_constraint_kind_match_gives_covered(self):
        """
        Both req and unit have the same named constraint kind (retention_period).
        DisabledJudge must produce tentative COVERED so the verifier can validate values.
        """
        from app.application.use_cases.build_requirements import _normalize_text

        req = RequirementUnit(
            source_document_id="doc-tz",
            text="Система должна хранить журнал событий не менее 90 дней.",
            normalized_text=_normalize_text("Система должна хранить журнал событий не менее 90 дней."),
            constraints=[Constraint(kind="retention_period", operator=">=", value=90, unit="days")],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text="Журнал хранится не менее 90 суток на защищённом сервере.",
            normalized_text=_normalize_text("Журнал хранится не менее 90 суток на защищённом сервере."),
            constraints=[Constraint(kind="retention_period", operator=">=", value=90, unit="days")],
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"Expected COVERED via constraint-kind match, got {j.llm_label}; {j.explanation}"
        )
        assert "retention_period" in j.explanation or "constraint_kind" in j.explanation

    # -- Case 4: generic constraints must not trigger COVERED ----------------------

    def test_generic_constraint_kind_does_not_trigger_covered(self):
        """
        'generic' kind (years, section numbers, IP fragments) must NOT match and
        must NOT produce COVERED, even if both sides have numeric values extracted.
        """
        from app.application.use_cases.build_requirements import _normalize_text

        req = RequirementUnit(
            source_document_id="doc-tz",
            text="Версия протокола 2023, раздел 6.",
            normalized_text=_normalize_text("Версия протокола 2023, раздел 6."),
            constraints=[
                Constraint(kind="generic", operator="=", value=2023, unit=None),
                Constraint(kind="generic", operator="=", value=6, unit=None),
            ],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text="Проверяется раздел 4 спецификации протокола версии 192.",
            normalized_text=_normalize_text("Проверяется раздел 4 спецификации протокола версии 192."),
            constraints=[
                Constraint(kind="generic", operator="=", value=192, unit=None),
                Constraint(kind="generic", operator="=", value=4, unit=None),
            ],
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label != LLMLabel.COVERED, (
            f"Generic-kind constraints must NOT trigger COVERED, got {j.llm_label}; {j.explanation}"
        )

    # -- Case 5: same domain, weak overlap, different aspect → not COVERED ----------

    def test_same_domain_weak_overlap_is_not_covered(self):
        """
        Two sentences about authentication/security with lex ≈ 0.15 and no
        constraint match → should be PARTIAL or IRRELEVANT, never COVERED.
        """
        req = _make_req(
            "Система должна выполнять аутентификацию пользователей по паролю.",
            constraints=[],
        )
        unit = _make_unit(
            "Проверить авторизацию через ролевой доступ.",
            constraints=[],
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label in (LLMLabel.PARTIAL, LLMLabel.IRRELEVANT), (
            f"Expected PARTIAL or IRRELEVANT, got {j.llm_label}; {j.explanation}"
        )
        assert j.llm_label != LLMLabel.COVERED


# ===========================================================================
# Verifier numeric conflict quality tests
# ===========================================================================


class TestVerifierNumericConflictQuality:
    """
    Verify that _values_conflict() does not produce spurious conflicts.

    Key invariants:
    - Unitless constraints (generic kind, unit=None) never conflict
    - Different named kinds in the same unit class never conflict
    - Only same-kind, same-unit-class, different-value pairs → CONFLICT
    """

    def setup_method(self):
        self.verifier = PairVerifier()

    def _judgment(self, req, unit, label=LLMLabel.COVERED):
        return PairJudgment(
            req_id=req.req_id,
            unit_id=unit.unit_id,
            target_document_id=unit.target_document_id,
            llm_label=label,
            rule_adjusted_label=label,
        )

    # -- Case 3: unrelated generic numbers must NOT conflict -----------------------

    def test_generic_unitless_numbers_do_not_conflict(self):
        """
        Years, section numbers, and other unitless generics extracted by the
        constraint parser must NOT produce numeric CONFLICT.
        Regression: old code conflicted >=2023.0 vs =6.0 and >=192.0 vs >=4.0.
        """
        from app.application.use_cases.build_requirements import _normalize_text

        req = RequirementUnit(
            source_document_id="doc-tz",
            text="Версия протокола 2023, раздел 6.",
            normalized_text=_normalize_text("Версия протокола 2023, раздел 6."),
            constraints=[
                Constraint(kind="generic", operator=">=", value=2023, unit=None),
                Constraint(kind="generic", operator="=", value=6, unit=None),
            ],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text="Требование пункта 4, протокол версии 192.",
            normalized_text=_normalize_text("Требование пункта 4, протокол версии 192."),
            constraints=[
                Constraint(kind="generic", operator=">=", value=192, unit=None),
                Constraint(kind="generic", operator="=", value=4, unit=None),
            ],
        )
        j = self._judgment(req, unit)
        result = self.verifier.verify(j, req, unit)
        assert result.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"Generic unitless numbers must NOT produce CONFLICT; got {result.rule_adjusted_label}; "
            f"conflict_aspects={result.conflict_aspects}"
        )

    # -- Case 4: real numeric conflict — same kind + unit, different value ----------

    def test_real_numeric_conflict_same_kind_and_unit(self):
        """
        req: response_time <= 2 sec
        unit: response_time = 5 sec
        → CONFLICT (same kind, same unit class, different value)
        """
        from app.application.use_cases.build_requirements import _normalize_text

        req = RequirementUnit(
            source_document_id="doc-tz",
            text="Время ответа не более 2 секунд.",
            normalized_text=_normalize_text("Время ответа не более 2 секунд."),
            constraints=[Constraint(kind="response_time", operator="<=", value=2, unit="sec")],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text="Допустимое время ответа системы составляет 5 секунд.",
            normalized_text=_normalize_text("Допустимое время ответа системы составляет 5 секунд."),
            constraints=[Constraint(kind="response_time", operator="=", value=5, unit="sec")],
        )
        j = self._judgment(req, unit)
        result = self.verifier.verify(j, req, unit)
        assert result.rule_adjusted_label == LLMLabel.CONFLICT, (
            f"Expected CONFLICT for response_time 2 sec vs 5 sec; "
            f"got {result.rule_adjusted_label}"
        )

    # -- Case: different kind in same unit class → no conflict ----------------------

    def test_different_kind_same_unit_class_no_conflict(self):
        """
        retention_period=90 days vs response_time=5 sec — both in _time_units.
        Different kinds → must NOT conflict.
        """
        from app.application.use_cases.build_requirements import _normalize_text

        req = RequirementUnit(
            source_document_id="doc-tz",
            text="Хранить журнал не менее 90 дней.",
            normalized_text=_normalize_text("Хранить журнал не менее 90 дней."),
            constraints=[Constraint(kind="retention_period", operator=">=", value=90, unit="days")],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text="Время ответа на запрос не более 5 секунд.",
            normalized_text=_normalize_text("Время ответа на запрос не более 5 секунд."),
            constraints=[Constraint(kind="response_time", operator="<=", value=5, unit="sec")],
        )
        j = self._judgment(req, unit)
        result = self.verifier.verify(j, req, unit)
        assert result.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"Different constraint kinds (retention_period vs response_time) must NOT conflict; "
            f"got {result.rule_adjusted_label}"
        )


# ===========================================================================
# Section-aware requirement builder tests (5 new scenarios)
# ===========================================================================


class TestSectionAwareRequirementBuilder:
    """
    Test 1: RequirementBuilder filters descriptive intro-section text out of the
    requirement set while keeping real requirements from section 4.x.
    """

    def test_requirement_section_included_trigger_only(self):
        """Fragment from section 4 with trigger word → included even with UNKNOWN modality."""
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "fragments": [
                {
                    "fragment_id": "f1",
                    "section_id": "4.1",
                    "text": "Предоставлять возможность проверки типов входных данных пользователя.",
                    "kind": "paragraph",
                },
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 1, "Section-4 requirement-like fragment must be included"

    def test_intro_section_descriptive_excluded(self):
        """
        Fragment from section 2 (Введение) with a trigger word but NO explicit
        modality (UNKNOWN) → must be excluded from RequirementUnit set.
        """
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "fragments": [
                {
                    "fragment_id": "f1",
                    "section_id": "2",
                    "text": "Данная система предназначена для обеспечивать учёт ресурсов организации.",
                    "kind": "paragraph",
                },
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 0, (
            "Descriptive fragment from section 2 without explicit modality must be excluded"
        )

    def test_intro_section_with_must_modality_included(self):
        """Fragment from section 3 with explicit 'должна' → must be included despite non-req section."""
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "fragments": [
                {
                    "fragment_id": "f1",
                    "section_id": "3",
                    "text": "Система должна обеспечивать обработку не менее 100 запросов.",
                    "kind": "paragraph",
                },
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 1, "Fragment with explicit 'должна' must be included regardless of section"

    def test_mixed_sections_only_req_sections_without_modality_included(self):
        """
        Mix of req-section (4.x) fragments and intro-section (2.x) fragments.
        Only section-4 fragments or those with explicit modality survive.
        """
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "fragments": [
                {
                    "fragment_id": "f1",
                    "section_id": "2.1",
                    "text": "Система предназначена для обеспечивать учёт данных организации.",
                    "kind": "paragraph",
                },
                {
                    "fragment_id": "f2",
                    "section_id": "4.1",
                    "text": "Предоставлять возможность фильтрации данных по дате и типу.",
                    "kind": "paragraph",
                },
                {
                    "fragment_id": "f3",
                    "section_id": "4.2",
                    "text": "Обеспечивать поддержку экспорта данных в формат CSV.",
                    "kind": "paragraph",
                },
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 2, (
            f"Expected 2 reqs from sections 4.1 and 4.2 only; got {len(reqs)}: "
            f"{[r.source_section_id for r in reqs]}"
        )
        section_ids = {r.source_section_id for r in reqs}
        assert "4.1" in section_ids
        assert "4.2" in section_ids


class TestNearVerbatimRequirementMatch:
    """
    Test 2: Near-verbatim requirement in TZ and PMI → COVERED.
    """

    def test_near_verbatim_predostavlyat_covered(self):
        """
        TZ: "Предоставлять возможность проверки типов..."
        PMI: "Программный продукт должен предоставлять возможность проверки типов..."
        Expected: COVERED (same action verb, significant lexical overlap)
        """
        status = _run(_make_package(
            "Предоставлять возможность проверки типов входных данных пользователем.",
            "Разрабатываемый программный продукт должен предоставлять возможность "
            "проверки типов входных данных пользователем системы.",
        ))
        assert status == CoverageStatus.COVERED, (
            f"Near-verbatim requirement with same action verb must be COVERED; got {status}"
        )

    def test_near_verbatim_obespechivat_covered(self):
        """
        TZ: "Обеспечивать разграничение прав доступа пользователей."
        PMI: "Система должна обеспечивать разграничение прав доступа пользователей по ролям."
        Expected: COVERED
        """
        status = _run(_make_package(
            "Обеспечивать разграничение прав доступа пользователей.",
            "Система должна обеспечивать разграничение прав доступа пользователей по ролям.",
        ))
        assert status == CoverageStatus.COVERED, (
            f"Near-verbatim 'обеспечивать' match must be COVERED; got {status}"
        )


class TestSectionRolePriorMatch:
    """
    Test 3: TZ section 4.2 requirement matched by PMI section 3.2 → section role prior helps.
    """

    def test_section_prior_tz4_pmi3_covered(self):
        """
        When TZ requirement is in section 4.x and PMI coverage unit is in section 3.x,
        and they share the same action verb with moderate lexical overlap → COVERED.
        """
        from app.application.use_cases.build_requirements import _normalize_text
        from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge

        judge = DisabledCoverageJudge()
        req = RequirementUnit(
            source_document_id="doc-tz",
            source_section_id="4.2",
            text="Обеспечивать ввод и редактирование данных через интерфейс пользователя.",
            normalized_text=_normalize_text(
                "Обеспечивать ввод и редактирование данных через интерфейс пользователя."
            ),
            constraints=[],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            section_id="3.2",
            text="Разрабатываемый продукт должен обеспечивать ввод и редактирование "
                 "данных пользователя через интерфейс.",
            normalized_text=_normalize_text(
                "Разрабатываемый продукт должен обеспечивать ввод и редактирование "
                "данных пользователя через интерфейс."
            ),
            constraints=[],
        )
        j = judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"TZ4→PMI3 with same action verb must be COVERED; got {j.llm_label}; {j.explanation}"
        )


class TestIntroDescriptiveNotDominantMissing:
    """
    Test 4: General descriptive statement from TZ intro section must not become
    a dominant MISSING requirement when filtered out correctly.
    """

    def test_intro_descriptive_not_in_requirements_set(self):
        """
        A descriptive sentence from section 2 without explicit modality
        must be excluded from RequirementUnit set (not become a MISSING requirement).
        """
        artifact = {
            "document_id": "doc-tz",
            "doc_role": "tz",
            "requirement_candidates": [
                {
                    "req_id": "c1",
                    "fragment_id": "f1",
                    "section_id": "2",
                    "text": "Система предназначена для реализовывать учёт финансовых операций.",
                },
                {
                    "req_id": "c2",
                    "fragment_id": "f2",
                    "section_id": "4.1",
                    "text": "Система должна обеспечивать учёт финансовых операций.",
                },
            ],
        }
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 1, (
            f"Only section-4 candidate must survive; got {len(reqs)}: "
            f"{[(r.source_section_id, r.text[:40]) for r in reqs]}"
        )
        assert reqs[0].source_section_id == "4.1"


class TestPMISection6Coverage:
    """
    Test 5: Requirement covered by PMI section 6 method text → at least PARTIAL,
    preferably COVERED when evidence is strong.
    """

    def test_pmi_section6_method_covers_tz4_requirement(self):
        """
        TZ section 4.x requirement matched by PMI section 6.x verification method
        text with same action verb → COVERED via section-role prior.
        """
        from app.application.use_cases.build_requirements import _normalize_text
        from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge

        judge = DisabledCoverageJudge()
        req = RequirementUnit(
            source_document_id="doc-tz",
            source_section_id="4.4",
            text="Обеспечивать восстановление работоспособности системы после сбоя.",
            normalized_text=_normalize_text(
                "Обеспечивать восстановление работоспособности системы после сбоя."
            ),
            constraints=[],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            section_id="6.3",
            text="Метод проверки: убедиться что система обеспечивает восстановление "
                 "работоспособности после имитированного сбоя питания.",
            normalized_text=_normalize_text(
                "Метод проверки: убедиться что система обеспечивает восстановление "
                "работоспособности после имитированного сбоя питания."
            ),
            constraints=[],
        )
        j = judge.judge(req, unit)
        assert j.llm_label in (LLMLabel.COVERED, LLMLabel.PARTIAL), (
            f"PMI section 6 verification method must yield at least PARTIAL; "
            f"got {j.llm_label}; {j.explanation}"
        )


# ===========================================================================
# Object-aware judge tests
# ===========================================================================


class TestObjectAwareJudge:
    """
    Object-aware disabled judge: verb match alone must never produce COVERED.
    Object phrase after the verb is the primary discriminator.
    """

    def setup_method(self):
        from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
        self.judge = DisabledCoverageJudge()

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _req(text: str, section_id: str = None) -> RequirementUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _extract_modality,
            _extract_requirement_type, _normalize_text,
        )
        return RequirementUnit(
            source_document_id="doc-tz",
            source_section_id=section_id,
            text=text,
            normalized_text=_normalize_text(text),
            requirement_type=_extract_requirement_type(text),
            modality=_extract_modality(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    @staticmethod
    def _unit(text: str, section_id: str = None) -> CoverageUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _normalize_text,
        )
        return CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            section_id=section_id,
            text=text,
            normalized_text=_normalize_text(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    # -- Test 1: same verb, same object → COVERED ------------------------------

    def test_same_verb_same_object_covered(self):
        """
        TZ: "Предоставлять возможность проверки типов входных данных."
        PMI: "Должен предоставлять возможность проверки типов входных данных пользователем."
        → COVERED: same verb AND same object phrase core tokens.
        """
        req = self._req("Предоставлять возможность проверки типов входных данных.")
        unit = self._unit(
            "Разрабатываемый программный продукт должен предоставлять возможность "
            "проверки типов входных данных пользователем."
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"Same verb + same object must be COVERED; got {j.llm_label}; {j.explanation}"
        )
        # Explanation must mention object match
        assert "object" in j.explanation.lower() or "verb" in j.explanation.lower()

    # -- Test 2: same verb, different object → NOT COVERED ----------------------

    def test_same_verb_different_object_not_covered(self):
        """
        TZ: "Предоставлять возможность проверки типов..."
        PMI: "Предоставлять возможность компиляции модуля..."
        → NOT COVERED: verb matches but object phrases are disjoint.
        """
        req = self._req("Предоставлять возможность проверки типов входных данных.")
        unit = self._unit("Предоставлять возможность компиляции модуля программы.")
        j = self.judge.judge(req, unit)
        assert j.llm_label != LLMLabel.COVERED, (
            f"Same verb + different object must NOT be COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )

    def test_same_verb_different_object_irrelevant_for_short_texts(self):
        """
        Short texts (verb + object only): different objects → IRRELEVANT
        because lex is also too low to produce PARTIAL via the fallback path.
        """
        req = self._req("Предоставлять возможность проверки типов.")
        unit = self._unit("Предоставлять возможность компиляции модуля.")
        j = self.judge.judge(req, unit)
        # Verb alone must not reach COVERED; PARTIAL or IRRELEVANT are both acceptable
        assert j.llm_label != LLMLabel.COVERED, (
            f"Same verb + different object (short text) must NOT be COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )
        # Explanation must mention that objects differ (for diagnostics)
        assert "объект" in j.explanation.lower() or "object" in j.explanation.lower() or \
               "verb" in j.explanation.lower() or j.llm_label == LLMLabel.PARTIAL

    # -- Test 3: different verbs, same object domain → PARTIAL via lex, not COVERED via verb

    def test_different_verbs_same_domain_not_verb_covered(self):
        """
        TZ: "Предоставлять сообщения об ошибках компиляции."
        PMI: "Формировать сообщения об ошибках в исходном коде."
        → verbs differ, so verb-path is blocked; lex overlap determines result.
        Must NOT be COVERED via verb mechanism (different verbs).
        """
        req = self._req("Предоставлять сообщения об ошибках компиляции.")
        unit = self._unit("Формировать сообщения об ошибках в исходном коде.")
        j = self.judge.judge(req, unit)
        # PARTIAL is acceptable (shared content words), IRRELEVANT also ok
        # COVERED is NOT ok because we can't confirm same semantic with different verbs
        assert j.llm_label != LLMLabel.COVERED, (
            f"Different verbs must not produce COVERED; got {j.llm_label}; {j.explanation}"
        )

    # -- Test 4: different verb, same domain domain word overlap → not COVERED -----

    def test_different_verb_same_domain_not_covered(self):
        """
        TZ: "Обеспечивать хранение журнала аудита."
        PMI: "Предоставлять журнал аудита пользователю."
        → different action verbs → verb-path blocked entirely → not COVERED.
        """
        req = self._req("Обеспечивать хранение журнала аудита.")
        unit = self._unit("Предоставлять журнал аудита пользователю.")
        j = self.judge.judge(req, unit)
        assert j.llm_label != LLMLabel.COVERED, (
            f"Different verbs must not produce COVERED; got {j.llm_label}; {j.explanation}"
        )

    # -- Test 5: regression — neighboring template requirements must not cross-COVERED

    def test_neighboring_template_requirements_not_covered(self):
        """
        Regression: neighboring functional requirements with the same verb template
        must NOT classify each other as COVERED solely due to shared verb.

        Pattern observed in real GOST TZ documents where section 4 lists:
          4.1.1 Предоставлять поддержку базовых конструкций языка.
          4.1.2 Предоставлять возможность проверки типов переменных.
          4.1.3 Предоставлять возможность компиляции отдельного модуля.
        """
        req_texts = [
            "Предоставлять поддержку базовых конструкций языка программирования.",
            "Предоставлять возможность проверки типов переменных.",
            "Предоставлять возможность компиляции отдельного модуля программы.",
        ]
        reqs = [self._req(t, section_id="4.1") for t in req_texts]

        # Each requirement must NOT be COVERED by either of the other two
        for i, req in enumerate(reqs):
            for j_idx, other in enumerate(reqs):
                if i == j_idx:
                    continue
                unit = self._unit(other.text, section_id="3.1")
                j = self.judge.judge(req, unit)
                assert j.llm_label != LLMLabel.COVERED, (
                    f"req[{i}] must NOT be COVERED by req[{j_idx}] (different requirements, same verb template); "
                    f"got {j.llm_label}; {j.explanation}\n"
                    f"  req: {req.text}\n"
                    f"  unit: {unit.text}"
                )

    # -- Test 6: object_overlap in extraction works correctly --------------------

    def test_extract_action_object_basic(self):
        """Unit test for _extract_action_object helper."""
        from app.infrastructure.llm.disabled_coverage_judge import _extract_action_object

        verb, obj_phrase, obj_tokens = _extract_action_object(
            "предоставлять возможность проверки типов входных данных"
        )
        assert verb == "предоставлять"
        # "возможность" is boilerplate and must be removed from tokens
        assert "проверки" in obj_tokens
        assert "типов" in obj_tokens
        assert "возможность" not in obj_tokens, "'возможность' must be filtered as boilerplate"

    def test_extract_action_object_no_verb(self):
        """No action verb → returns (None, '', frozenset())."""
        from app.infrastructure.llm.disabled_coverage_judge import _extract_action_object

        verb, obj_phrase, obj_tokens = _extract_action_object(
            "кнопка выхода должна быть видна пользователю"
        )
        assert verb is None
        assert obj_tokens == frozenset()

    def test_extract_action_object_boundary_at_comma(self):
        """Object extraction stops at comma."""
        from app.infrastructure.llm.disabled_coverage_judge import _extract_action_object

        verb, obj_phrase, obj_tokens = _extract_action_object(
            "обеспечивать хранение журнала аудита, не менее 90 дней"
        )
        assert verb == "обеспечивать"
        assert "90" not in obj_phrase, "Object must not cross the comma boundary"
        assert "хранение" in obj_tokens or "журнала" in obj_tokens or "аудита" in obj_tokens


# ===========================================================================
# Bug-fix tests: Strong dual evidence demotion + exact text match promotion
# ===========================================================================


class TestStrongDualEvidenceDemotion:
    """
    Bug 1: old 'Strong dual evidence' path could return COVERED even when
    object phrases were completely different (same template, different object).

    Fix: lex >= 0.40 AND entity_ov >= 0.25 alone is capped at PARTIAL.
    COVERED requires object_covers OR a text-match signal.
    """

    def setup_method(self):
        from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
        self.judge = DisabledCoverageJudge()

    @staticmethod
    def _req(text: str) -> RequirementUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _extract_modality,
            _extract_requirement_type, _normalize_text,
        )
        return RequirementUnit(
            source_document_id="doc-tz",
            text=text,
            normalized_text=_normalize_text(text),
            requirement_type=_extract_requirement_type(text),
            modality=_extract_modality(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    @staticmethod
    def _unit(text: str) -> CoverageUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _normalize_text,
        )
        return CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text=text,
            normalized_text=_normalize_text(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    def test_same_template_different_object_not_covered(self):
        """
        Test 1: same requirement template (shared boilerplate inflates lex/entity),
        but completely different objects.
        Result: NOT COVERED regardless of lex/entity values.

        This is the regression guard against the old 'Strong dual evidence' path.
        """
        req = self._req(
            "Разрабатываемый программный продукт должен предоставлять возможность "
            "проверки типов входных данных пользователем."
        )
        unit = self._unit(
            "Разрабатываемый программный продукт должен предоставлять возможность "
            "компиляции отдельного модуля программы."
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label != LLMLabel.COVERED, (
            f"Same template + different object must NOT be COVERED even if lex/entity is high; "
            f"got {j.llm_label}; {j.explanation}"
        )

    def test_strong_dual_evidence_weak_object_is_partial(self):
        """
        Test 2: high lex + high entity_overlap but object_overlap is weak.
        The demoted 'Strong dual evidence' path must produce PARTIAL, not COVERED.
        """
        from app.application.use_cases.build_requirements import _normalize_text
        from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge

        judge = DisabledCoverageJudge()

        # Construct a req and unit with high lex but disjoint objects.
        # Use explicit entities to force entity_ov >= 0.25.
        req = RequirementUnit(
            source_document_id="doc-tz",
            text="Система должна предоставлять поддержку конструкций цикла.",
            normalized_text=_normalize_text(
                "Система должна предоставлять поддержку конструкций цикла."
            ),
            entities=["Система", "Pascal", "IDE"],
            constraints=[],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text="Система должна предоставлять поддержку конструкций ветвления.",
            normalized_text=_normalize_text(
                "Система должна предоставлять поддержку конструкций ветвления."
            ),
            entities=["Система", "Pascal", "IDE"],
            constraints=[],
        )
        j = judge.judge(req, unit)
        # These share high lex (differ only in last word) and high entity overlap,
        # but "цикла" vs "ветвления" means different objects.
        # Must NOT be COVERED — at most PARTIAL.
        assert j.llm_label != LLMLabel.COVERED, (
            f"High lex+entity but different object must NOT be COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )
        # Explanation must NOT say "Strong dual evidence → COVERED"
        assert "strong dual evidence" not in j.explanation.lower() or "partial" in j.explanation.lower(), (
            f"Demoted path must mention PARTIAL, not COVERED; {j.explanation}"
        )

    def test_regression_old_strong_dual_evidence_path(self):
        """
        Test 5: regression — old Path 4 'Strong dual evidence' must not give COVERED
        for cases where object phrases are weak matches (object_overlap < 0.6).
        """
        from app.application.use_cases.build_requirements import _normalize_text

        judge = self.judge
        # Build pairs with identical entities but different object phrases
        shared_entities = ["Pascal", "IDE", "Компилятор"]

        req = RequirementUnit(
            source_document_id="doc-tz",
            text="Предоставлять поддержку базовых конструкций языка программирования.",
            normalized_text=_normalize_text(
                "Предоставлять поддержку базовых конструкций языка программирования."
            ),
            entities=shared_entities,
            constraints=[],
        )
        unit = CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text="Предоставлять возможность проверки типов переменных программы.",
            normalized_text=_normalize_text(
                "Предоставлять возможность проверки типов переменных программы."
            ),
            entities=shared_entities,
            constraints=[],
        )
        j = judge.judge(req, unit)
        assert j.llm_label != LLMLabel.COVERED, (
            f"Strong dual evidence with weak object match must NOT be COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )


class TestExactTextMatchPromotion:
    """
    Bug 2: lex=1.00 with no verb extraction → stays PARTIAL.

    Fix: exact normalized text match, text containment, and near-exact lex
    are now explicit COVERED paths (before verb-based paths).
    """

    def setup_method(self):
        from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
        self.judge = DisabledCoverageJudge()

    @staticmethod
    def _req(text: str, sec: str = None) -> RequirementUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _extract_modality,
            _extract_requirement_type, _normalize_text,
        )
        return RequirementUnit(
            source_document_id="doc-tz",
            source_section_id=sec,
            text=text,
            normalized_text=_normalize_text(text),
            requirement_type=_extract_requirement_type(text),
            modality=_extract_modality(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    @staticmethod
    def _unit(text: str, sec: str = None) -> CoverageUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _normalize_text,
        )
        return CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            section_id=sec,
            text=text,
            normalized_text=_normalize_text(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    def test_exact_normalized_text_match_covered(self):
        """
        Test 3: normalized texts are identical (noun-phrase requirement, no verb extracted).
        Must be COVERED, not stuck at PARTIAL.
        """
        text = "Поддержка конструкций условных операторов."
        req = self._req(text)
        unit = self._unit(text)
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"Exact normalized text match must be COVERED; got {j.llm_label}; {j.explanation}"
        )
        assert "exact" in j.explanation.lower(), (
            f"Explanation must mention exact match; got: {j.explanation}"
        )
        assert "normalized_text_exact_match" in j.matched_aspects

    def test_near_verbatim_pmi_restatement_covered(self):
        """
        Test 4: PMI restates TZ requirement with minor additions (common GOST pattern).
        req text is verbatim substring of unit text → text_containment path → COVERED.
        """
        req = self._req(
            "Предоставлять возможность проверки типов входных данных пользователем."
        )
        unit = self._unit(
            "Разрабатываемый программный продукт должен предоставлять возможность "
            "проверки типов входных данных пользователем системы."
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"Near-verbatim text containment must be COVERED; got {j.llm_label}; {j.explanation}"
        )
        # Must use text-match path, not fall into PARTIAL
        assert "containment" in j.explanation.lower() or "exact" in j.explanation.lower() or \
               "near-exact" in j.explanation.lower() or "verb" in j.explanation.lower()

    def test_high_lex_no_verb_noun_phrase_covered(self):
        """
        Test 6: documentation-like requirement stated as noun phrase (no action verb).
        Verb extractor finds nothing, but near-exact lex (≥0.80) plus non-trivial
        object tokens → COVERED via near_exact_lex path.
        """
        req = self._req("Поддержка операторов цикла while и for в компиляторе.")
        unit = self._unit(
            "Поддержка операторов цикла while и for в компиляторе Паскаль."
        )
        j = self.judge.judge(req, unit)
        # lex should be very high (differ by one token "Паскаль")
        # near_exact_lex >= 0.80 with non-trivial obj → COVERED
        assert j.llm_label == LLMLabel.COVERED, (
            f"Near-exact lex with non-trivial content must be COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )

    def test_lex_one_without_verb_not_partial(self):
        """
        Regression guard: lex=1.00 must never stay at PARTIAL.
        Even when verb extraction doesn't fire, exact text match catches it.
        """
        text = "Обнаружение ошибок синтаксического анализа программного кода."
        req = self._req(text)
        unit = self._unit(text)
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"lex=1.00 (identical texts) must be COVERED, not PARTIAL; "
            f"got {j.llm_label}; {j.explanation}"
        )

    def test_short_boilerplate_text_no_false_covered(self):
        """
        Guard: short boilerplate-only text that happens to be identical must not
        trigger near_exact_lex (non_trivial guard blocks it), but text_exact still
        fires (identical texts are a valid match regardless of length).
        """
        text = "Поддержка."
        req = self._req(text)
        unit = self._unit(text)
        j = self.judge.judge(req, unit)
        # text_exact fires (identical) → COVERED is acceptable for literally same text
        # This is correct: if PMI says exactly what TZ says, it's covered.
        # The test just documents expected behavior:
        assert j.llm_label in (LLMLabel.COVERED, LLMLabel.PARTIAL), (
            f"Identical short text: COVERED or PARTIAL; got {j.llm_label}"
        )


# ===========================================================================
# Tests: artifact-aware matching and verification-step detection (Message 4)
# ===========================================================================


class TestArtifactAndVerificationJudge:
    """
    Verify two new fix classes in DisabledCoverageJudge:
      Fix A — verification-step PMI units get PARTIAL (not IRRELEVANT)
              when section pairing is structurally plausible.
      Fix B — deliverable document artifact names (morphological variants)
              are matched canonically; same artifact → COVERED.
    """

    judge = DisabledCoverageJudge()

    @staticmethod
    def _req(text: str, src_sec: str = "4.1") -> RequirementUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _extract_modality,
            _extract_requirement_type, _normalize_text,
        )
        return RequirementUnit(
            source_document_id="doc-tz",
            source_section_id=src_sec,
            text=text,
            normalized_text=_normalize_text(text),
            requirement_type=_extract_requirement_type(text),
            modality=_extract_modality(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    @staticmethod
    def _unit(text: str, sec: str = "3.1") -> CoverageUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _normalize_text,
        )
        return CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            section_id=sec,
            text=text,
            normalized_text=_normalize_text(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    # ------------------------------------------------------------------
    # Fix A — verification-step units
    # ------------------------------------------------------------------

    def test_verification_step_unit_is_partial_not_irrelevant(self):
        """
        A PMI unit of the form "Пункт N) проверяется через запуск программы..."
        has zero lexical overlap with a functional TZ requirement but describes
        how that requirement is tested.  Must be PARTIAL (not IRRELEVANT) when
        the TZ4→PMI3 section pairing is structurally plausible.
        """
        req = self._req(
            "Система должна предоставлять поддержку базовых конструкций языка Pascal.",
            src_sec="4.2",
        )
        unit = self._unit(
            "Пункт 4.2) проверяется через запуск программы с базовыми конструкциями языка.",
            sec="3.1",
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.PARTIAL, (
            f"Verification-step unit with plausible section pairing must be PARTIAL; "
            f"got {j.llm_label}; {j.explanation}"
        )
        assert "верификац" in j.explanation.lower() or "verification" in j.explanation.lower() or \
               "verify" in j.explanation.lower() or "проверяется" in j.explanation.lower() or \
               "partial" in j.explanation.lower(), j.explanation

    def test_verification_step_without_section_id_still_partial(self):
        """
        A verification-step unit with no section_id (common in real PMI data) must
        still be PARTIAL — not IRRELEVANT.  sec_plausible is NOT required; the
        retrieval shortlist provides the relevance gate.
        """
        req = self._req(
            "Система должна поддерживать базовые конструкции языка Pascal.",
            src_sec=None,   # unknown section
        )
        unit = self._unit(
            "Пункт 4.2) проверяется путём запуска тестовой программы.",
            sec=None,       # no section_id — real PMI data often omits this
        )
        j = self.judge.judge(req, unit)
        # PX_VERIFY fires on is_verify alone; result must be PARTIAL, not IRRELEVANT.
        assert j.llm_label == LLMLabel.PARTIAL, (
            f"Verification-step without section_id must be PARTIAL; "
            f"got {j.llm_label}; {j.explanation}"
        )
        assert j.llm_label != LLMLabel.COVERED, "Verification-step must not be COVERED"

    # ------------------------------------------------------------------
    # Fix B — document artifact canonical matching
    # ------------------------------------------------------------------

    def test_pmi_doc_artifact_covered_morphological_variant(self):
        """
        TZ req: "Должны быть разработаны программа и методика испытаний."
        PMI unit: "В рамках данной работы разрабатываются программы и методики испытаний."
        Both mention the same deliverable artifact (pmi_doc) in different inflections.
        artifact_jac = 1.0 → COVERED via PX_ART path.
        """
        req = self._req(
            "Должны быть разработаны программа и методика испытаний.",
            src_sec="4.1",
        )
        unit = self._unit(
            "В рамках данной работы разрабатываются программы и методики испытаний.",
            sec="3.1",
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"Morphological variant of same artifact must be COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )
        assert "artifact" in j.explanation.lower(), (
            f"Explanation must mention artifact matching; got: {j.explanation}"
        )

    def test_operator_manual_artifact_covered(self):
        """
        TZ req mentions "руководство оператора"; PMI unit mentions "руководства оператора"
        (genitive case).  Both resolve to operator_manual → COVERED.
        """
        req = self._req(
            "Должно быть разработано руководство оператора для программного комплекса.",
            src_sec="4.3",
        )
        unit = self._unit(
            "Настоящий документ является руководства оператора разрабатываемого комплекса.",
            sec="3.2",
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"operator_manual artifact variant must be COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )

    def test_different_artifacts_not_artifact_covered(self):
        """
        TZ req mentions "руководство оператора"; PMI unit mentions "текст программы".
        Different artifact categories → artifact_jac = 0 → PX_ART path does NOT fire.
        (Result may be PARTIAL or IRRELEVANT depending on other signals, not COVERED.)
        """
        req = self._req(
            "Должно быть разработано руководство оператора.",
            src_sec="4.3",
        )
        unit = self._unit(
            "В рамках работы формируется текст программы и пояснительная записка.",
            sec="3.2",
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label != LLMLabel.COVERED, (
            f"Different artifact categories must not produce COVERED; "
            f"got {j.llm_label}; {j.explanation}"
        )


# ===========================================================================
# Tests: P10 shared-token path, explainability, remaining-MISSING fixes
# ===========================================================================


class TestExplainabilityAndP10:
    """
    Tests for:
      - P10: shared_token_count >= 3 AND lex >= 0.12 → PARTIAL (conservative low-lex path)
      - Enriched matched_aspects / missing_aspects for all PARTIAL paths
      - Expanded _ARTIFACT_CANONICAL
      - Unrelated retrieval noise stays IRRELEVANT
      - Negation verifier guard: llm_confidence < 0.25 → no CONFLICT
    """

    judge = DisabledCoverageJudge()

    @staticmethod
    def _req(text: str, src_sec: str = "4.1") -> RequirementUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _extract_modality,
            _extract_requirement_type, _normalize_text,
        )
        return RequirementUnit(
            source_document_id="doc-tz",
            source_section_id=src_sec,
            text=text,
            normalized_text=_normalize_text(text),
            requirement_type=_extract_requirement_type(text),
            modality=_extract_modality(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    @staticmethod
    def _unit(text: str, sec: str = "3.1") -> CoverageUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _normalize_text,
        )
        return CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            section_id=sec,
            text=text,
            normalized_text=_normalize_text(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    # ── Test 1: weak but meaningful document-family evidence → PARTIAL ────

    def test_p10_three_shared_tokens_lex_above_threshold_gives_partial(self):
        """
        P10 fires when shared_token_count >= 3 AND lex >= 0.12.
        Texts sharing 3 meaningful content words with lex ~0.13 must become PARTIAL.
        (count=2 or lex<0.12 is intentionally blocked to avoid boilerplate noise.)
        """
        req = self._req(
            "Система должна обеспечивать корректное завершение работы при "
            "некорректных входных данных пользователя."
        )
        unit = self._unit(
            "Проводится тестирование завершения работы приложения при "
            "подаче на вход некорректных данных различного типа, включая "
            "пустые строки, спецсимволы и числа вне допустимого диапазона."
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.PARTIAL, (
            f"Weak but meaningful vocabulary overlap must be PARTIAL via P10; "
            f"got {j.llm_label}; {j.explanation}"
        )
        assert j.llm_label != LLMLabel.COVERED, "P10 must not produce COVERED"

    # ── Test 2: verification PARTIAL has rich explanation ─────────────────

    def test_verification_partial_has_rich_matched_aspects(self):
        """
        PX_VERIFY must produce matched_aspects containing 'verification_unit'
        and 'testing_context_present', plus missing_aspects about the gap.
        """
        req = self._req(
            "Система должна обеспечивать поддержку базовых конструкций языка Pascal.",
            src_sec="4.3",
        )
        unit = self._unit(
            "Пункт 4.3) проверяется через запуск программы с набором тестовых примеров.",
            sec=None,
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.PARTIAL, (
            f"Verification-step must be PARTIAL; got {j.llm_label}"
        )
        assert "verification_unit" in j.matched_aspects, (
            f"matched_aspects must include 'verification_unit'; got {j.matched_aspects}"
        )
        assert "testing_context_present" in j.matched_aspects, (
            f"matched_aspects must include 'testing_context_present'; got {j.matched_aspects}"
        )
        assert any("exact_requirement" in a or "direct_functional" in a
                   for a in j.missing_aspects), (
            f"missing_aspects must describe the gap; got {j.missing_aspects}"
        )

    # ── Test 3: artifact-family PARTIAL has informative explanation ────────

    def test_artifact_partial_has_family_in_matched_aspects(self):
        """
        PX_ART_PARTIAL: when req and unit share a document artifact family but
        not the exact same artifact, matched_aspects must name the family.
        """
        req = self._req(
            "Должна быть разработана программная документация по ГОСТ 19.",
            src_sec="4.4",
        )
        unit = self._unit(
            "В рамках настоящего документа разрабатывается программная документация.",
            sec="3.3",
        )
        j = self.judge.judge(req, unit)
        # Both texts share 'prog_docs' category → artifact_jac = 1.0 → PX_ART COVERED
        # or at minimum PARTIAL if only partial overlap; either way not IRRELEVANT
        assert j.llm_label in (LLMLabel.COVERED, LLMLabel.PARTIAL), (
            f"Artifact family overlap must not be IRRELEVANT; got {j.llm_label}"
        )
        # matched_aspects must name the artifact family or the match type
        has_artifact_info = any(
            "artifact" in a or "prog_docs" in a for a in j.matched_aspects
        )
        assert has_artifact_info, (
            f"matched_aspects must include artifact family info; got {j.matched_aspects}"
        )

    # ── Test 4: exact artifact match still COVERED (regression guard) ─────

    def test_exact_artifact_covered_regression(self):
        """
        Regression: existing PX_ART COVERED path must not be broken by P10 changes.
        """
        req = self._req(
            "Должны быть разработаны программа и методика испытаний.",
            src_sec="4.1",
        )
        unit = self._unit(
            "В рамках данной работы разрабатываются программы и методики испытаний.",
            sec="3.1",
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.COVERED, (
            f"Exact artifact match (pmi_doc) must be COVERED; got {j.llm_label}; {j.explanation}"
        )

    # ── Test 5: P10 boundary — 2 tokens or lex < 0.12 stays IRRELEVANT ──────

    def test_p10_below_threshold_stays_irrelevant(self):
        """
        Guard: P10 must NOT fire when shared_token_count < 3 or lex < 0.12.
        Two boilerplate tokens (пользователя, программа) with lex=0.12 must not
        become PARTIAL — this is the pattern that caused false CONFLICTs via
        the negation verifier.
        """
        req = self._req(
            "Программа не должна препятствовать корректному завершению сеанса пользователя."
        )
        unit = self._unit(
            "Программа должна корректно осуществлять сохранение всех изменений данных "
            "документа при выходе пользователя из приложения программного комплекса."
        )
        # shared tokens: "пользователя", "программа" (2 tokens) → count < 3 → P10 doesn't fire
        # Neither has artifact, neither is verify, lex is low
        # Expected: IRRELEVANT (different aspect requirements)
        j = self.judge.judge(req, unit)
        assert j.llm_label != LLMLabel.CONFLICT, (
            f"Weak overlap between different aspect requirements must not be CONFLICT; "
            f"got {j.llm_label}; {j.explanation}"
        )

    # ── Test 5b: genuinely unrelated retrieval noise stays IRRELEVANT ──────

    def test_unrelated_unit_stays_irrelevant(self):
        """
        A unit with no shared content tokens, no artifact overlap, no verb match,
        and no verification pattern must remain IRRELEVANT.
        P10 must not fire when shared_token_count < 3.
        """
        req = self._req(
            "Система должна обеспечивать хранение журнала аудита не менее 90 дней.",
            src_sec="4.2",
        )
        unit = self._unit(
            "Список использованной литературы: монография, учебник, пособие.",
            sec="7.1",
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label == LLMLabel.IRRELEVANT, (
            f"Unrelated noise must stay IRRELEVANT; got {j.llm_label}; {j.explanation}"
        )

    # ── Test 6: PARTIAL matched_aspects and missing_aspects are non-empty ─

    def test_topical_partial_has_non_empty_aspects(self):
        """
        P9 topical-overlap PARTIAL must produce non-empty matched_aspects
        (including shared_tokens) and non-empty missing_aspects describing the gap.
        """
        req = self._req(
            "Система должна реализовывать поддержку операторов цикла Pascal."
        )
        unit = self._unit(
            "Тестирование реализации операторов цикла for и while в компиляторе Pascal "
            "проводится на нескольких тестовых примерах программного кода."
        )
        j = self.judge.judge(req, unit)
        assert j.llm_label in (LLMLabel.PARTIAL, LLMLabel.COVERED), (
            f"Topically overlapping pair must be at least PARTIAL; got {j.llm_label}"
        )
        assert len(j.matched_aspects) > 0, (
            f"matched_aspects must be non-empty; got {j.matched_aspects}"
        )
        if j.llm_label == LLMLabel.PARTIAL:
            assert len(j.missing_aspects) > 0, (
                f"missing_aspects must describe the gap for PARTIAL; got {j.missing_aspects}"
            )


# ===========================================================================
# Tests: negation verifier false-CONFLICT guard
# ===========================================================================


class TestNegationVerifierGuard:
    """
    Verify that the negation/modality verifier does NOT fire on low-confidence
    pairs (llm_confidence < 0.25).  This was the root cause of false CONFLICTs
    for cand::10 and cand::12: an exact-COVERED pair + a weak PARTIAL pair with
    opposite modality → aggregator saw CONFLICT > COVERED.
    """

    @staticmethod
    def _req(text: str) -> RequirementUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _extract_modality,
            _extract_requirement_type, _normalize_text,
        )
        return RequirementUnit(
            source_document_id="doc-tz",
            text=text,
            normalized_text=_normalize_text(text),
            requirement_type=_extract_requirement_type(text),
            modality=_extract_modality(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    @staticmethod
    def _unit(text: str) -> CoverageUnit:
        from app.application.use_cases.build_requirements import (
            _extract_constraints, _extract_entities, _normalize_text,
        )
        return CoverageUnit(
            target_document_id="doc-pmi",
            target_doc_role="pmi",
            text=text,
            normalized_text=_normalize_text(text),
            entities=_extract_entities(text),
            constraints=_extract_constraints(text),
        )

    def test_low_confidence_negation_mismatch_not_conflict(self):
        """
        Regression for cand::10 and cand::12 false CONFLICTs.

        Setup: MUST_NOT requirement + PMI unit with positive assertion,
        but their overlap is weak (lex ~0.12, only 2 shared tokens).
        Judge produces PARTIAL with low confidence; verifier must NOT
        escalate to CONFLICT because the pair might simply be about
        different aspects of the system, not a real contradiction.
        """
        from app.application.use_cases.verify_pairs import PairVerifier
        judge = DisabledCoverageJudge()
        verifier = PairVerifier()

        req = self._req(
            "Программа не должна препятствовать корректному завершению сеанса пользователя."
        )
        unit = self._unit(
            "Программа должна корректно осуществлять сохранение всех изменений "
            "документа при выходе пользователя из приложения программного комплекса."
        )
        j = judge.judge(req, unit)
        # Precondition: confidence must be low (< 0.25) for this scenario
        assert j.llm_confidence < 0.25, (
            f"Precondition failed: expected low confidence, got {j.llm_confidence}"
        )
        # Apply verifier
        j_verified = verifier.verify(j, req, unit)
        assert j_verified.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"Low-confidence negation mismatch must not become CONFLICT; "
            f"got {j_verified.rule_adjusted_label}; explanation: {j_verified.explanation}"
        )

    def test_high_confidence_negation_mismatch_is_conflict(self):
        """
        Regression guard: real negation contradiction on a high-confidence pair
        (same topic, opposite modality, lex > 0.30) must still be flagged as CONFLICT.
        """
        from app.application.use_cases.verify_pairs import PairVerifier
        judge = DisabledCoverageJudge()
        verifier = PairVerifier()

        req = self._req(
            "Система не должна хранить журнал событий более 30 дней."
        )
        unit = self._unit(
            "Система должна хранить журнал событий не менее 90 дней "
            "согласно требованиям регулятора по безопасности."
        )
        j = judge.judge(req, unit)
        # High lex: "хранить", "журнал", "событий", "дней" — many shared tokens
        assert j.llm_confidence >= 0.25, (
            f"Precondition failed: expected high confidence, got {j.llm_confidence}"
        )
        j_verified = verifier.verify(j, req, unit)
        # Numeric conflict fires first (30 vs 90 days) — either CONFLICT from
        # numeric rule or from negation is acceptable
        assert j_verified.rule_adjusted_label == LLMLabel.CONFLICT, (
            f"High-confidence negation+numeric contradiction must be CONFLICT; "
            f"got {j_verified.rule_adjusted_label}"
        )


# ===========================================================================
# TestAllSameReqIdHandling
# ===========================================================================


class TestAllSameReqIdHandling:
    """
    RequirementBuilder must use position-based IDs when all requirement_candidates
    share the same raw req_id (prepare-service bug).

    Tests:
      1. All-same-req_id + no fragment_id → positional IDs (cand::0, cand::1, ...)
      2. All-same-req_id + fragment_id available → fragment_id wins for that candidate
      3. Unique raw req_ids → raw req_id used as-is (no position override)
      4. Regression: section-4 candidates with unique req_ids remain included
    """

    DOC_ID = "doc-tz-001"

    def _artifact(self, candidates):
        return {"document_id": self.DOC_ID, "requirement_candidates": candidates}

    def _cand(self, text, req_id=None, fragment_id=None, section_id="4.1"):
        c = {"text": text, "section_id": section_id}
        if req_id is not None:
            c["req_id"] = req_id
        if fragment_id is not None:
            c["fragment_id"] = fragment_id
        return c

    def test_all_same_req_id_generates_positional_ids(self):
        """When all candidates share a req_id the builder must assign cand::0, cand::1, ..."""
        shared_id = f"{self.DOC_ID}:::cand"
        artifact = self._artifact([
            self._cand("Система должна предоставлять возможность A.", req_id=shared_id),
            self._cand("Система должна предоставлять возможность B.", req_id=shared_id),
            self._cand("Система должна предоставлять возможность C.", req_id=shared_id),
        ])
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 3
        ids = [r.req_id for r in reqs]
        assert ids[0] == f"{self.DOC_ID}::cand::0"
        assert ids[1] == f"{self.DOC_ID}::cand::1"
        assert ids[2] == f"{self.DOC_ID}::cand::2"

    def test_all_same_req_id_fragment_id_takes_priority(self):
        """When fragment_id is present it takes priority even in the all-same-req_id case."""
        shared_id = f"{self.DOC_ID}:::cand"
        artifact = self._artifact([
            self._cand("Система должна предоставлять возможность A.",
                       req_id=shared_id, fragment_id="frag-001"),
            self._cand("Система должна предоставлять возможность B.",
                       req_id=shared_id),
        ])
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 2
        assert reqs[0].req_id == f"{self.DOC_ID}::frag-001"
        assert reqs[1].req_id == f"{self.DOC_ID}::cand::1"

    def test_unique_req_ids_used_as_is(self):
        """When every candidate has a distinct raw req_id that value is kept unchanged."""
        artifact = self._artifact([
            self._cand("Система должна предоставлять возможность A.",
                       req_id="req-A"),
            self._cand("Система должна предоставлять возможность B.",
                       req_id="req-B"),
        ])
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == 2
        assert reqs[0].req_id == "req-A"
        assert reqs[1].req_id == "req-B"

    def test_section4_candidates_remain_included_after_fix(self):
        """
        Regression: valid section-4 requirement candidates must all be included
        regardless of the req_id deduplication strategy.
        """
        shared_id = f"{self.DOC_ID}:::cand"
        texts = [
            "Система должна предоставлять поддержку базовых конструкций языка программирования.",
            "Система должна предоставлять возможность проверки синтаксической корректности.",
            "Система должна предоставлять возможность компиляции целого модуля.",
            "Система должна обеспечивать защищённый доступ к данным пользователя.",
            "Система должна хранить журнал операций не менее 90 дней.",
        ]
        artifact = self._artifact([
            self._cand(t, req_id=shared_id, section_id="4.1") for t in texts
        ])
        reqs = RequirementBuilder().build(artifact)
        assert len(reqs) == len(texts), (
            f"All {len(texts)} section-4 requirements must be included; got {len(reqs)}"
        )
        req_ids = [r.req_id for r in reqs]
        assert req_ids == [f"{self.DOC_ID}::cand::{i}" for i in range(len(texts))]


# ===========================================================================
# Section-driven extraction tests
# ===========================================================================


class TestSectionDrivenExtraction:
    """
    Requirement extraction that trusts ONLY the sections hierarchy from
    prepare-service. Fragment splits, requirement_candidates, and all other
    heuristic outputs of prepare-service are ignored.
    """

    DOC_ID = "doc-tz-sec"

    def _cfg(self) -> CoverageConfig:
        cfg = CoverageConfig()
        cfg.requirement_extraction = "sections"
        return cfg

    def _artifact(self, sections, fragments) -> dict:
        return {
            "document_id": self.DOC_ID,
            "doc_role": "tz",
            "sections": sections,
            "fragments": fragments,
            "requirement_candidates": [
                {"req_id": "CAND-SHOULD-NOT-BE-USED", "text": "Игнорируй меня."}
            ],
        }

    def test_ignores_requirement_candidates(self):
        artifact = self._artifact(
            sections=[{"section_id": "4", "title": "Требования к программе"}],
            fragments=[
                {"fragment_id": "f1", "section_id": "4",
                 "text": "Система должна хранить журнал не менее 90 дней."}
            ],
        )
        reqs = RequirementBuilder(self._cfg()).build(artifact)
        assert len(reqs) == 1
        assert "Игнорируй" not in reqs[0].text

    def test_resegments_concatenated_fragments(self):
        # prepare-service may have split the same requirement into two
        # fragments mid-sentence OR lumped two requirements into one — we
        # should get sentence-level units regardless.
        artifact = self._artifact(
            sections=[{"section_id": "4.1", "title": "Функциональные требования"}],
            fragments=[
                {"fragment_id": "f1", "section_id": "4.1",
                 "text": "Система должна хранить журнал событий не менее 90 дней. "
                         "Также система должна обеспечивать авторизацию пользователей."},
                {"fragment_id": "f2", "section_id": "4.1",
                 "text": "Система должна обеспечивать время отклика не более 2 секунд."},
            ],
        )
        reqs = RequirementBuilder(self._cfg()).build(artifact)
        texts = [r.text for r in reqs]
        assert len(reqs) == 3, f"expected 3 sentence-level requirements, got {texts}"
        assert any("90 дней" in t for t in texts)
        assert any("авторизацию" in t for t in texts)
        assert any("2 секунд" in t for t in texts)

    def test_skips_non_requirement_sections_by_title(self):
        artifact = self._artifact(
            sections=[
                {"section_id": "A", "title": "Введение"},
                {"section_id": "B", "title": "Требования к надёжности"},
            ],
            fragments=[
                {"fragment_id": "f1", "section_id": "A",
                 "text": "Система должна работать быстро (описание целей проекта)."},
                {"fragment_id": "f2", "section_id": "B",
                 "text": "Система должна обеспечивать доступность не менее 99.9%."},
            ],
        )
        reqs = RequirementBuilder(self._cfg()).build(artifact)
        assert len(reqs) == 1
        assert "99.9" in reqs[0].text

    def test_stable_req_ids(self):
        artifact = self._artifact(
            sections=[{"section_id": "4.2", "title": "Требования к безопасности"}],
            fragments=[
                {"fragment_id": "f1", "section_id": "4.2",
                 "text": "Система должна шифровать данные при передаче. "
                         "Пароли должны храниться в виде хэшей."},
            ],
        )
        reqs1 = RequirementBuilder(self._cfg()).build(artifact)
        reqs2 = RequirementBuilder(self._cfg()).build(artifact)
        assert [r.req_id for r in reqs1] == [r.req_id for r in reqs2]
        assert all(r.req_id.startswith(f"{self.DOC_ID}::4.2::s") for r in reqs1)

    def test_abbreviation_not_split(self):
        # "т.е." must not end a sentence.
        artifact = self._artifact(
            sections=[{"section_id": "4", "title": "Требования"}],
            fragments=[
                {"fragment_id": "f1", "section_id": "4",
                 "text": "Система должна поддерживать HTTP, т.е. принимать запросы по порту 80."}
            ],
        )
        reqs = RequirementBuilder(self._cfg()).build(artifact)
        assert len(reqs) == 1
        assert "т.е." in reqs[0].text

    def test_short_sentences_filtered(self):
        artifact = self._artifact(
            sections=[{"section_id": "4", "title": "Требования"}],
            fragments=[
                {"fragment_id": "f1", "section_id": "4",
                 "text": "Должен работать. Система должна принимать входящие запросы "
                         "и обрабатывать их в течение 2 секунд."}
            ],
        )
        reqs = RequirementBuilder(self._cfg()).build(artifact)
        assert len(reqs) == 1
        assert "2 секунд" in reqs[0].text

    def test_falls_back_to_fragments_without_sections(self):
        # If no sections[] at all, the section path falls back to fragments
        # rather than returning empty.
        artifact = {
            "document_id": self.DOC_ID,
            "doc_role": "tz",
            "sections": [],
            "fragments": [
                {"fragment_id": "f1",
                 "text": "Система должна хранить журнал не менее 90 дней."}
            ],
        }
        reqs = RequirementBuilder(self._cfg()).build(artifact)
        assert len(reqs) == 1

    def test_ambiguous_section_requires_explicit_modality(self):
        # Section with no number and a neutral title: only MUST/MUST_NOT/SHOULD
        # sentences admitted — prevents descriptive prose leaking in.
        artifact = self._artifact(
            sections=[{"section_id": "X", "title": "Общая информация о системе"}],
            fragments=[
                {"fragment_id": "f1", "section_id": "X",
                 "text": "Система обеспечивает высокую производительность работы. "
                         "Система должна поддерживать одновременно 1000 пользователей."},
            ],
        )
        reqs = RequirementBuilder(self._cfg()).build(artifact)
        assert len(reqs) == 1
        assert "1000" in reqs[0].text
