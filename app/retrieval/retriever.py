import os
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np

# Optional dependency: sentence-transformers (preferred if the model is an SBERT export)
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

import torch
from transformers import AutoModel, AutoTokenizer


class _HFMeanPoolEncoder:
    """HF encoder with mean pooling (no SentenceTransformer export required).

    This is a robust fallback for cases where ./model is a plain HuggingFace checkpoint
    (config.json + tokenizer + model weights) or when a broken SBERT export is present.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)

            last_hidden = out.last_hidden_state  # (bs, seq, dim)
            attn = enc.get("attention_mask")
            if attn is None:
                pooled = last_hidden.mean(dim=1)
            else:
                mask = attn.unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = summed / counts

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)


class CRetriever:
    """Retrieve Top-K PMI tests for a TZ requirement.

    Supports TWO model formats:
      1) SentenceTransformer export folder (contains modules.json or 0_Transformer/)
      2) Plain HuggingFace checkpoint folder (config.json + tokenizer + weights)

    The second format is important because many training notebooks accidentally save
    HF checkpoints rather than a full SentenceTransformers export.
    """

    def __init__(self, model_path: str = "./model", top_k: int = 5, device: Optional[str] = None):
        base_dir = Path(__file__).resolve().parents[2]
        self.model_path = str((base_dir / model_path).resolve()) if not os.path.isabs(model_path) else model_path
        self.top_k = int(top_k)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._backend: str = "hf"
        self._model: Union[_HFMeanPoolEncoder, "SentenceTransformer"] = self._load_model(self.model_path)

    # ---------- public API ----------

    def retrieve(self, query_text: str, candidates: List[str], k: Optional[int] = None) -> List[Tuple[int, float]]:
        """Return list of (candidate_index, cosine_score) sorted desc."""
        if not candidates:
            return []
        k = self.top_k if k is None else int(k)

        q_vec = self._encode([query_text])  # (1, dim)
        c_vec = self._encode(candidates)    # (n, dim)

        # cosine similarity assuming vectors are L2-normalized
        scores = (c_vec @ q_vec.T).reshape(-1)
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    # Backward compatibility with earlier naming
    def topk(self, query: str, corpus: List[str], k: Optional[int] = None) -> List[Tuple[int, float]]:
        return self.retrieve(query, corpus, k=k)

    # ---------- internals ----------

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self._backend == "sbert":
            # type: ignore[union-attr]
            vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return vecs.astype(np.float32)
        # HF fallback
        return self._model.encode(texts)  # type: ignore[union-attr]

    def _looks_like_sbert_export(self, p: Path) -> bool:
        return (p / "modules.json").exists() or (p / "0_Transformer").exists() or (p / "1_Pooling").exists()

    def _looks_like_hf_checkpoint(self, p: Path) -> bool:
        # common HF artifacts
        return (p / "config.json").exists() and (
            (p / "model.safetensors").exists()
            or (p / "pytorch_model.bin").exists()
            or any(p.glob("model-*.safetensors"))
        )

    def _load_model(self, model_path: str) -> Union[_HFMeanPoolEncoder, "SentenceTransformer"]:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Retrieval model path not found: {model_path}")

        # 1) Try SBERT export (best)
        if self._looks_like_sbert_export(p) and SentenceTransformer is not None:
            try:
                self._backend = "sbert"
                return SentenceTransformer(str(p), device=self.device)
            except Exception:
                # fall through to HF fallback
                pass

        # 2) HF checkpoint fallback
        if self._looks_like_hf_checkpoint(p) or True:
            # Even if we can't positively detect HF layout, we try it — AutoModel will
            # fail fast with a readable error.
            self._backend = "hf"
            return _HFMeanPoolEncoder(str(p), device=self.device)

        raise RuntimeError(f"Cannot load retrieval model from: {model_path}")
