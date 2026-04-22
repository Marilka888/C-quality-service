"""
Common interface for pair rerankers.

A reranker takes (query, candidate_texts) and returns a relevance score per
candidate. Unlike the retrieval embedder it sees the pair jointly — this
is why cross-encoders beat bi-encoders on short technical Russian.

Implementations are expected to be batch-friendly (one forward pass per
N candidates) and to tolerate being called with empty candidate lists.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, candidates: List[str]) -> List[float]:
        """Return one relevance score per candidate. Higher = more relevant.

        Scores may live in any range (logits for some models, 0..1 for
        others); only the ORDER is guaranteed meaningful across a single
        call. Callers that need comparable scores across queries should
        apply their own normalisation.
        """


class NoopReranker(Reranker):
    """Pass-through reranker — keeps retrieval score ordering.

    Used when the reranker is disabled by config so the pipeline can be
    constructed uniformly without branching on `if reranker is None`.
    """

    def score(self, query: str, candidates: List[str]) -> List[float]:
        # Returning equal scores means the upstream retrieval order is
        # preserved (Python's sort is stable).
        return [0.0] * len(candidates)
