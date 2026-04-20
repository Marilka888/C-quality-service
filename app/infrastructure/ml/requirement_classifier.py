"""
Binary "is-requirement" sentence classifier.

Wraps a fine-tuned HuggingFace `BertForSequenceClassification` checkpoint
(see `model/req_classifier/`, produced by `scripts/train_req_classifier.py`
and the v5 dataset build) behind a narrow interface so the pipeline does
not need to know about torch/transformers at call time.

Design choices:

  - **Lazy load.** Torch + transformers are heavy imports (~seconds) and
    pin 1+ GB RAM for the weights. Loading is deferred until the first
    call to `predict_proba` so tests / services that do not use the ML
    path pay no cost.

  - **Singleton per path.** Typical deployment uses one fine-tuned
    checkpoint; repeated constructions with the same path return the
    same underlying model (avoids double-load when the pipeline is
    rebuilt per-request).

  - **CPU by default, CUDA if available.** No configuration required.

  - **Batch inference.** Requirement extraction feeds the classifier
    hundreds of sentences at a time; per-sentence forward passes would
    dominate end-to-end latency.

  - **Soft-fail.** If torch/transformers are missing or the checkpoint
    directory is invalid, the classifier raises at construction time,
    so callers can fall back to rule-based extraction rather than
    silently returning empty predictions.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

_LOAD_LOCK = threading.Lock()
_CACHE: dict = {}  # {abs_path: (tokenizer, model, device)}


class RequirementClassifier:
    """Batch sentence-level binary classifier.

    Parameters
    ----------
    model_path : str
        Directory containing a HuggingFace sequence-classification
        checkpoint (config.json, model.safetensors, tokenizer.json).
    max_len : int
        Max tokens per sentence; 192 mirrors the length used in training.
    batch_size : int
        Batch size for the forward pass.
    """

    def __init__(
        self,
        model_path: str,
        max_len: int = 192,
        batch_size: int = 32,
    ) -> None:
        path = Path(model_path).resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"Requirement classifier checkpoint not found at {path}. "
                f"Expected config.json inside."
            )
        self._path = str(path)
        self._max_len = int(max_len)
        self._batch_size = int(batch_size)

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._path in _CACHE:
            return
        with _LOAD_LOCK:
            if self._path in _CACHE:
                return
            # Imports are deferred so the service can run without torch
            # when requirement_extraction != "model".
            try:
                import torch
                from transformers import (  # type: ignore
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Requirement classifier requires torch + transformers; "
                    "install them or set requirement_extraction to a "
                    "non-model mode."
                ) from exc

            logger.info("Loading requirement classifier from %s", self._path)
            tokenizer = AutoTokenizer.from_pretrained(self._path)
            model = AutoModelForSequenceClassification.from_pretrained(self._path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            _CACHE[self._path] = (tokenizer, model, device)
            logger.info("Requirement classifier loaded on %s", device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, texts: List[str]) -> List[float]:
        """Return P(label=1) for each text. Preserves input order."""
        if not texts:
            return []
        self._ensure_loaded()
        import torch  # safe here — _ensure_loaded has already succeeded

        tokenizer, model, device = _CACHE[self._path]
        probs_pos: List[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self._batch_size):
                chunk = texts[i : i + self._batch_size]
                enc = tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self._max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                probs_pos.extend(probs[:, 1].tolist())
        return probs_pos

    def predict_is_requirement(
        self, texts: List[str], threshold: float = 0.5
    ) -> List[bool]:
        return [p >= threshold for p in self.predict_proba(texts)]
