from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

from app.infrastructure.embeddings.base import EmbeddingBackend

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


class BagOfWordsEmbeddingBackend(EmbeddingBackend):
    def similarity(self, query: str, candidates: List[str]) -> List[float]:
        query_counter = Counter(_tokenize(query))
        return [self._cosine(query_counter, Counter(_tokenize(candidate))) for candidate in candidates]

    @staticmethod
    def _cosine(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(left[token] * right[token] for token in set(left) & set(right))
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
