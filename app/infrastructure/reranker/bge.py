"""
Cross-encoder reranker backed by BGE (BAAI/bge-reranker-v2-m3).

Architecture: a single transformer that sees (query, passage) jointly
and emits one relevance logit. This is the opposite of bi-encoder
embeddings — much slower per pair but roughly +10–20 percentage points
of nDCG on typical IR tasks, because the model can attend across the
pair (crucial on short technical Russian where individual word overlap
is a weak signal).

Role in the pipeline
--------------------
    retrieve top-N (hybrid score) → rerank → take top-K

N is `top_k_before_rerank` (default 20), K is `top_k`. 20 is a reasonable
compromise: a cross-encoder at batch 20 takes ~100 ms on a GPU 1650, a
few seconds on CPU — acceptable per requirement, and much better
ordering than the bi-encoder alone.

Model: `BAAI/bge-reranker-v2-m3` is a 568M parameter multilingual
cross-encoder. Downloads to HF cache on first use (~2.3 GB).

Design notes
------------
    - **Lazy load.** Same pattern as the classifier: don't touch torch
      until someone actually calls `score`. Keeps tests fast and lets
      the service start without the weights on disk.
    - **Per-path singleton.** Constructor arguments map to a cache key
      so rebuilding the pipeline per request does not reload the model.
    - **Graceful fallback.** On any failure (import error, HF download
      issue, OOM) the caller is expected to fall back to retrieval
      order via `NoopReranker`; we raise so the error is visible.
"""
from __future__ import annotations

import threading
from typing import List, Tuple

from app.core.logging import get_logger
from app.infrastructure.reranker.base import Reranker

logger = get_logger(__name__)

_LOAD_LOCK = threading.Lock()
_CACHE: dict = {}  # {(model_name, device): (tokenizer, model)}


def _get_cross_encoder(model_name: str, device: str) -> Tuple:
    """Return (tokenizer, model) cached by (model_name, device)."""
    key = (model_name, device)
    if key in _CACHE:
        return _CACHE[key]
    with _LOAD_LOCK:
        if key in _CACHE:
            return _CACHE[key]
        try:
            import torch  # noqa: F401 — ensure torch is available
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BGE reranker requires torch + transformers."
            ) from exc
        logger.info("Loading cross-encoder reranker %r on %s", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(device).eval()
        _CACHE[key] = (tokenizer, model)
        return _CACHE[key]


class BGEReranker(Reranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        max_len: int = 512,
        batch_size: int = 16,
    ) -> None:
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._model_name = model_name
        self._device = device
        self._max_len = int(max_len)
        self._batch_size = int(batch_size)

    def score(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        import torch

        tokenizer, model = _get_cross_encoder(self._model_name, self._device)
        pairs = [(query, c) for c in candidates]
        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self._batch_size):
                batch = pairs[i : i + self._batch_size]
                enc = tokenizer(
                    [p[0] for p in batch],
                    [p[1] for p in batch],
                    padding=True,
                    truncation=True,
                    max_length=self._max_len,
                    return_tensors="pt",
                ).to(self._device)
                # BGE reranker emits a single relevance logit per pair.
                # Apply sigmoid so the returned score lands in (0, 1) —
                # the rest of the pipeline (AdaptiveCandidateSelector,
                # evidence_strength binning, evidence_floor gate) all
                # assume this range. Without sigmoid, raw logits like
                # -3.0 / -5.0 for irrelevant pairs propagate downstream
                # and trigger over-aggressive skip_llm_below_floor on
                # every non-trivial requirement (Polyakov re-run with
                # raw-logit output: 29 pair(s) skipped vs an expected
                # 5-10).
                logits = model(**enc).logits.view(-1)
                probs = torch.sigmoid(logits)
                scores.extend(probs.cpu().tolist())
        return scores
