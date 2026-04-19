from __future__ import annotations

from typing import List


class EmbeddingBackend:
    def similarity(self, query: str, candidates: List[str]) -> List[float]:
        raise NotImplementedError
