"""
Multilingual E5 embedding backend.

Wraps `sentence-transformers` loading of the E5 family (e.g.
`intfloat/multilingual-e5-base`) with two E5-specific behaviours:

  - **Prompt prefixes.** E5 was trained with "query:" and "passage:"
    markers; embeddings come out measurably better when you use them.
    Our pairs are (requirement, coverage_unit) — the requirement acts
    as a query and each unit is a passage.
  - **Cosine similarity via normalised dot product.** E5 vectors come
    out L2-normalised from the model, so a plain matrix product gives
    the cosine and we avoid redundant math.

Loads lazily (sentence-transformers import is heavy) and caches the
model per path so rebuilding the pipeline per request reuses the same
instance.

Why E5 specifically (vs. the generic mBERT in ./model): E5 was fine-tuned
on a 1B+ pair dataset for semantic similarity across 100 languages,
with Russian in the top tier. On short technical Russian sentences like
"время отклика" vs "латентность запроса", E5-base cosine is ~0.85;
the generic mBERT used to give ~0.5, which was not a useful retrieval
signal.
"""
from __future__ import annotations

import threading
from typing import List

import numpy as np

from app.core.logging import get_logger
from app.infrastructure.embeddings.base import EmbeddingBackend

logger = get_logger(__name__)

_LOAD_LOCK = threading.Lock()
_CACHE: dict = {}  # {model_name: sentence_transformers.SentenceTransformer}


def _get_model(model_name: str, device: str):
    """Return a cached SentenceTransformer instance for `model_name`."""
    key = (model_name, device)
    if key in _CACHE:
        return _CACHE[key]
    with _LOAD_LOCK:
        if key in _CACHE:
            return _CACHE[key]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "E5 backend requires sentence-transformers; install it or "
                "switch to backend='bow'."
            ) from exc
        logger.info("Loading E5 embedding model %r on %s", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        _CACHE[key] = model
        return model


class E5EmbeddingBackend(EmbeddingBackend):
    """Semantic similarity via a multilingual E5 cross-lingual encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default "intfloat/multilingual-e5-base")
        or a local path. First load downloads ~1 GB to the HF cache.
    device : str
        "cuda" or "cpu"; default autodetects.
    batch_size : int
        Encoding batch size.
    """

    _QUERY_PREFIX = "query: "
    _PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._model_name = model_name
        self._device = device
        self._batch_size = int(batch_size)

    # ------------------------------------------------------------------

    def similarity(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        model = _get_model(self._model_name, self._device)
        q_vec = model.encode(
            [self._QUERY_PREFIX + (query or "")],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        c_vec = model.encode(
            [self._PASSAGE_PREFIX + (c or "") for c in candidates],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        # Vectors are L2-normalised → dot product == cosine
        sims = (c_vec @ q_vec.T).reshape(-1)
        return [float(s) for s in sims.astype(np.float32)]
