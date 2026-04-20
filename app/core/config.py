from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    min_retrieval_score: float = Field(default=0.2, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    section_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    metadata_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    embedding_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    use_embeddings: bool = False
    embedding_model_path: Optional[str] = "./model"


class RuleConfig(BaseModel):
    require_expected_result_for_assertive_tests: bool = True


class StatusThresholdConfig(BaseModel):
    inadequate_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    adequate_overlap_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    minimal_overlap_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    orphan_support_score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class ScoringConfig(BaseModel):
    status_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "ADEQUATE": 1.0,
            "PARTIAL": 0.5,
            "INADEQUATE": 0.25,
            "MISSING": 0.0,
            "CONFLICT": 0.0,
        }
    )
    orphan_penalty_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    thresholds: StatusThresholdConfig = Field(default_factory=StatusThresholdConfig)


class LLMConfig(BaseModel):
    enabled: bool = False
    backend: str = "disabled"
    backend_model_name: Optional[str] = None


class ServiceConfig(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rules: RuleConfig = Field(default_factory=RuleConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    random_seed: int = 42

    @classmethod
    def from_request_overrides(
        cls,
        top_k: Optional[int] = None,
        min_retrieval_score: Optional[float] = None,
        use_llm: Optional[bool] = None,
        use_embeddings: Optional[bool] = None,
    ) -> "ServiceConfig":
        config = cls()
        if top_k is not None:
            config.retrieval.top_k = top_k
        if min_retrieval_score is not None:
            config.retrieval.min_retrieval_score = min_retrieval_score
        if use_llm is not None:
            config.llm.enabled = use_llm
            config.llm.backend = "configured" if use_llm else "disabled"
        if use_embeddings is not None:
            config.retrieval.use_embeddings = use_embeddings
        return config

    def resolve_embedding_model_path(self, project_root: Path) -> Optional[Path]:
        path = self.retrieval.embedding_model_path
        if not path:
            return None
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (project_root / candidate).resolve()


# ---------------------------------------------------------------------------
# Coverage analysis config (C-quality-service package-level pipeline)
# ---------------------------------------------------------------------------


class CoverageRetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)
    min_retrieval_score: float = Field(default=0.05, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    constraint_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    section_prior_weight: float = Field(default=0.10, ge=0.0, le=1.0)


class CoverageLLMConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = False
    backend: str = "ollama"  # "ollama" | "disabled"
    model_name: str = "qwen2.5:3b"
    prompt_version: str = "v1"
    timeout: int = 120


class CoverageEmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    backend: str = "transformer"  # "transformer" | "bow"
    model_path: Optional[str] = "./model"


class RequirementModelConfig(BaseModel):
    """Settings for the fine-tuned requirement classifier.

    Only consulted when `CoverageConfig.requirement_extraction == "model"`.
    The model checkpoint is NOT loaded at config-construction time — the
    extractor loads it lazily on first use (see
    `app.infrastructure.ml.requirement_classifier.RequirementClassifier`).
    """

    model_config = ConfigDict(protected_namespaces=())

    model_path: str = "./model/req_classifier"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_len: int = Field(default=192, ge=32, le=512)
    batch_size: int = Field(default=32, ge=1, le=256)


class CoverageConfig(BaseModel):
    retrieval: CoverageRetrievalConfig = Field(default_factory=CoverageRetrievalConfig)
    llm: CoverageLLMConfig = Field(default_factory=CoverageLLMConfig)
    embedding: CoverageEmbeddingConfig = Field(default_factory=CoverageEmbeddingConfig)
    requirement_model: RequirementModelConfig = Field(
        default_factory=RequirementModelConfig
    )
    enable_rule_verification: bool = True
    # "auto"     — candidates → fragments → sections, first non-empty wins
    # "sections" — only trust sections hierarchy; re-segment text inside each
    #              requirement-section ourselves (prepare-service fragment
    #              splits are ignored)
    # "model"    — use the fine-tuned requirement_classifier to score each
    #              sentence in requirement-plausible sections
    # "candidates" / "fragments" — legacy paths for explicit control
    requirement_extraction: str = "auto"

    @classmethod
    def from_options(cls, options: Dict) -> "CoverageConfig":
        config = cls()
        if "top_k" in options:
            config.retrieval.top_k = int(options["top_k"])
        if "enable_llm_judge" in options:
            config.llm.enabled = bool(options["enable_llm_judge"])
            if config.llm.enabled:
                config.llm.backend = "ollama"
        if "enable_rule_verification" in options:
            config.enable_rule_verification = bool(options["enable_rule_verification"])
        if "min_retrieval_score" in options:
            config.retrieval.min_retrieval_score = float(options["min_retrieval_score"])
        if "requirement_extraction" in options:
            mode = str(options["requirement_extraction"]).lower()
            if mode in {"auto", "sections", "candidates", "fragments", "model"}:
                config.requirement_extraction = mode
        if "requirement_model_path" in options:
            config.requirement_model.model_path = str(options["requirement_model_path"])
        if "requirement_model_threshold" in options:
            config.requirement_model.threshold = float(
                options["requirement_model_threshold"]
            )
        return config
