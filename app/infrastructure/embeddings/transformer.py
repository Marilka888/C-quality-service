from __future__ import annotations

from pathlib import Path
from typing import List

from app.infrastructure.embeddings.base import EmbeddingBackend
from app.retrieval.retriever import CRetriever


class TransformerEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_path: Path):
        self._retriever = CRetriever(model_path=str(model_path))

    def similarity(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        ranked = self._retriever.retrieve(query, candidates, k=len(candidates))
        scores = [0.0] * len(candidates)
        for index, score in ranked:
            scores[index] = float(score)
        return scores
