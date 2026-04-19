from __future__ import annotations

from pathlib import Path
from typing import List

from app.application.use_cases.build_report import build_traceability_report_payload
from app.application.use_cases.build_trace_matrix import TraceMatrixBuilder
from app.application.use_cases.detect_orphan_tests import OrphanTestDetector
from app.application.use_cases.evaluate_requirement_coverage import RequirementCoverageEvaluator
from app.core.config import ServiceConfig
from app.core.logging import get_logger
from app.domain.entities import Requirement, TestCase
from app.infrastructure.embeddings.base import EmbeddingBackend
from app.infrastructure.embeddings.simple import BagOfWordsEmbeddingBackend
from app.infrastructure.embeddings.transformer import TransformerEmbeddingBackend
from app.infrastructure.llm.base import LLMJudge
from app.infrastructure.llm.noop import DisabledLLMJudge
from app.infrastructure.rules.conflict_detector import RuleBasedConflictDetector


class TraceabilityService:
    def __init__(self, config: ServiceConfig):
        self._config = config
        self._logger = get_logger(self.__class__.__name__)
        self._project_root = Path(__file__).resolve().parents[2]

    def evaluate(self, requirements: List[Requirement], test_cases: List[TestCase]) -> dict:
        self._logger.info(
            "Starting traceability evaluation for %s requirements and %s test cases",
            len(requirements),
            len(test_cases),
        )
        test_cases_by_id = {test_case.id: test_case for test_case in test_cases}
        trace_matrix = TraceMatrixBuilder(
            config=self._config.retrieval,
            embedding_backend=self._build_embedding_backend(),
        ).build(requirements, test_cases)
        self._logger.info("Candidate generation completed")

        requirement_findings = RequirementCoverageEvaluator(
            rules=RuleBasedConflictDetector(self._config.rules),
            llm_judge=self._build_llm_judge(),
            scoring_config=self._config.scoring,
        ).evaluate(requirements, test_cases_by_id, trace_matrix)
        self._logger.info("Requirement coverage evaluation completed")

        orphan_findings = OrphanTestDetector(self._config.scoring).detect(test_cases, requirement_findings)
        self._logger.info("Orphan test detection completed")

        return build_traceability_report_payload(requirement_findings, orphan_findings, self._config)

    def _build_embedding_backend(self) -> EmbeddingBackend | None:
        if not self._config.retrieval.use_embeddings:
            return BagOfWordsEmbeddingBackend()
        model_path = self._config.resolve_embedding_model_path(self._project_root)
        if model_path and model_path.exists():
            try:
                return TransformerEmbeddingBackend(model_path)
            except Exception as exc:
                self._logger.warning("Falling back to bag-of-words embeddings: %s", exc)
        return BagOfWordsEmbeddingBackend()

    @staticmethod
    def _build_llm_judge() -> LLMJudge:
        return DisabledLLMJudge()
