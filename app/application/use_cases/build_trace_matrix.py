from __future__ import annotations
from typing import Dict, List, Optional

from app.core.config import RetrievalConfig
from app.core.text import tokenize_content, tokenize_raw
from app.domain.entities import CandidateTestCase, Requirement, TestCase
from app.infrastructure.embeddings.base import EmbeddingBackend


class TraceMatrixBuilder:
    def __init__(self, config: RetrievalConfig, embedding_backend: Optional[EmbeddingBackend] = None):
        self._config = config
        self._embedding_backend = embedding_backend

    def build(self, requirements: List[Requirement], test_cases: List[TestCase]) -> Dict[str, List[CandidateTestCase]]:
        candidate_map: Dict[str, List[CandidateTestCase]] = {}
        test_texts = [self._full_test_text(test_case) for test_case in test_cases]

        for requirement in requirements:
            embedding_scores = self._embedding_scores(requirement.text, test_texts)
            candidates: List[CandidateTestCase] = []
            for index, test_case in enumerate(test_cases):
                lexical_score = self._lexical_score(requirement.text, self._full_test_text(test_case))
                section_score = self._section_score(requirement.section, test_case.section)
                metadata_score = self._metadata_score(requirement.metadata, test_case.metadata)
                embedding_score = embedding_scores[index]
                retrieval_score = self._combine_scores(
                    lexical_score=lexical_score,
                    section_score=section_score,
                    metadata_score=metadata_score,
                    embedding_score=embedding_score,
                )
                if retrieval_score < self._config.min_retrieval_score:
                    continue
                candidates.append(
                    CandidateTestCase(
                        test_case_id=test_case.id,
                        retrieval_score=round(retrieval_score, 4),
                        lexical_score=round(lexical_score, 4),
                        section_score=round(section_score, 4),
                        metadata_score=round(metadata_score, 4),
                        embedding_score=round(embedding_score, 4),
                    )
                )

            candidates.sort(key=lambda item: item.retrieval_score, reverse=True)
            candidate_map[requirement.id] = candidates[: self._config.top_k]
        return candidate_map

    def _embedding_scores(self, query_text: str, test_texts: List[str]) -> List[float]:
        if not test_texts:
            return []
        if not self._embedding_backend or not self._config.use_embeddings:
            return [0.0] * len(test_texts)
        return [max(0.0, min(1.0, score)) for score in self._embedding_backend.similarity(query_text, test_texts)]

    @staticmethod
    def _full_test_text(test_case: TestCase) -> str:
        return " ".join(part for part in [test_case.text, test_case.expected_result or ""] if part)

    @staticmethod
    def _lexical_score(requirement_text: str, test_text: str) -> float:
        left = tokenize_content(requirement_text)
        right = tokenize_content(test_text)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left)

    @staticmethod
    def _section_score(requirement_section: str, test_section: str) -> float:
        if not requirement_section or not test_section:
            return 0.0
        requirement_tokens = tokenize_raw(requirement_section)
        test_tokens = tokenize_raw(test_section)
        if not requirement_tokens or not test_tokens:
            return 0.0
        return 1.0 if requirement_tokens & test_tokens else 0.0

    @staticmethod
    def _metadata_score(requirement_metadata: dict, test_metadata: dict) -> float:
        if not requirement_metadata or not test_metadata:
            return 0.0
        shared_keys = set(requirement_metadata) & set(test_metadata)
        if not shared_keys:
            return 0.0
        shared_values = sum(1 for key in shared_keys if requirement_metadata[key] == test_metadata[key])
        return shared_values / len(shared_keys)

    def _combine_scores(
        self,
        *,
        lexical_score: float,
        section_score: float,
        metadata_score: float,
        embedding_score: float,
    ) -> float:
        weighted_sum = (
            self._config.lexical_weight * lexical_score
            + self._config.section_weight * section_score
            + self._config.metadata_weight * metadata_score
            + self._config.embedding_weight * embedding_score
        )
        normalization = (
            self._config.lexical_weight
            + self._config.section_weight
            + self._config.metadata_weight
            + self._config.embedding_weight
        )
        return 0.0 if normalization == 0 else weighted_sum / normalization
