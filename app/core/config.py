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
    # BUG-9: floor below which we never trust an LLM verdict. min_retrieval_score
    # gates the retrieval shortlist (a noisy but generous filter); evidence_floor
    # gates whether the resulting verdict is authoritative. If the highest
    # retrieval score in the shortlist for a (req, target) pair is below this,
    # the pipeline marks the result `low_confidence=True` regardless of what
    # the LLM judge said — protects against CONFLICT/COVERED produced from
    # weak retrieval (audit-time symptom: CONFLICT with max evidence score 0.37).
    #
    # PR-K P0: lowered default from 0.5 to 0.30. Real packages with BoW +
    # qwen2.5:3b consistently produce max retrieval 0.40-0.49 for valid
    # coverage (Polyakov 0.20::sent1 max=0.4367 was a perfect coverage that
    # the old floor=0.5 demoted to MISSING). 0.30 still rejects truly weak
    # retrieval (≤0.30 means almost no shared signal) while admitting
    # legitimate semantic-only matches. With the split low_confidence /
    # grounding_failed semantics in PairJudgment, below-floor judgments
    # no longer auto-demote COVERED — they only set low_confidence on the
    # row for UI dimming.
    evidence_floor: float = Field(default=0.30, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    constraint_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    section_prior_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    # When the reranker is enabled, the first-stage hybrid retrieval
    # returns this many candidates; the cross-encoder then reorders them
    # and the pipeline keeps `top_k` from the reordered list. 20 is the
    # best quality/time trade-off on our data; too small reduces the
    # rerank benefit, too large slows the pipeline without F1 gains.
    top_k_before_rerank: int = Field(default=20, ge=5, le=100)
    # ── PR-K: AdaptiveCandidateSelector ─────────────────────────────
    # Initial shortlist size BEFORE the adaptive selector trims to a
    # judge-ready k. Decoupled from `top_k` so we can score generously,
    # bin into evidence_strength, and let the selector pick how many
    # to actually send to the LLM. 10-20 is a good range.
    initial_top_n: int = Field(default=10, ge=1, le=50)
    # EvidenceStrength bin thresholds on retrieval_score (0..1).
    # See EvidenceStrength docstring.
    evidence_strength_strong_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    evidence_strength_medium_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    evidence_strength_weak_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    # Score-margin gap that lets the selector trust top-1 alone when
    # top-1 is STRONG. Below this the pipeline broadens to top-3.
    selector_strong_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    # Cap selected_k regardless of available candidates.
    selector_max_k: int = Field(default=5, ge=1, le=20)
    # PR-K post-fix (B): when True AND debug.enabled is False, the judge
    # call is skipped entirely for shortlists whose max retrieval_score is
    # below evidence_floor — saves ~15-20% of LLM calls on packages where
    # some requirements have no strong retrieval signal. The resulting row
    # is MISSING_NO_EVIDENCE (same as if the shortlist were empty). When
    # debug is enabled the judge IS still called so the rationale appears
    # in evidence_trace for investigation. Default False preserves the
    # pre-B behaviour (call judge + stamp low_confidence, useful for
    # debugging false-COVERED/PARTIAL rows from weak retrieval).
    skip_llm_below_floor: bool = True


class CoverageAggregatorConfig(BaseModel):
    """Confidence / evidence-strength thresholds the
    EvidenceBasedCoverageAggregator uses to decide a verdict."""
    model_config = ConfigDict(protected_namespaces=())

    # COVERED accepted only when this confidence is reached AND
    # grounding_passed AND evidence is at least medium_threshold.
    covered_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # CONFLICT accepted only when this confidence is reached AND
    # grounding_passed AND retrieval_score >= medium_threshold.
    conflict_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    # PARTIAL accepted only when LLM confidence is at least this value.
    # PR-K post-fix (B): raised from 0.50 to 0.65 (symmetric with
    # covered_confidence_threshold) to eliminate false PARTIALs from
    # small models (qwen2.5:3b, llama3.2:3b) that output PARTIAL with
    # conf=0.50–0.64 on off-topic evidence. Real-package symptom:
    # Поляков package had 4 false PARTIALs in PZ driven by sticky
    # generic units scored ≈0.44; all had conf ≤ 0.64. Genuine PARTIALs
    # from well-evidenced partial coverage consistently show conf ≥ 0.70
    # in manual review. Override via CoverageConfig.from_options() or
    # env-driven config if a higher recall of weak PARTIALs is needed.
    partial_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # Aggregator-side "medium" floor on retrieval_score for confident
    # COVERED / CONFLICT. Distinct from the retrieval-side
    # evidence_floor (which only stamps low_confidence).
    medium_retrieval_threshold: float = Field(default=0.30, ge=0.0, le=1.0)


class CoverageDebugConfig(BaseModel):
    """Controls the size and verbosity of evidence_trace on each
    RequirementCoverageResult. Disabled by default to keep the wire
    payload small for the UI and the orchestrator's report DTO."""
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = False
    max_candidates: int = Field(default=5, ge=1, le=50)
    include_discarded: bool = False


class CoverageLLMConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = False
    # "ollama"        — local Ollama HTTP API (llama3 etc.)
    # "litellm"       — unified multi-provider via LiteLLM. `model_name`
    #                   becomes a LiteLLM routing string, e.g.
    #                   "xai/grok-2-latest",
    #                   "groq/llama-3.3-70b-versatile",
    #                   "gemini/gemini-2.0-flash",
    #                   "openai/gpt-4o-mini",
    #                   "anthropic/claude-3-5-haiku-latest",
    #                   "cerebras/llama-3.1-70b",
    #                   "ollama/qwen2.5:7b". Each provider expects its
    #                   API key in env (XAI_API_KEY, GROQ_API_KEY,
    #                   GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, …).
    # "cross_encoder" — zero-shot BGE cross-encoder as judge (reuses the
    #                   reranker model; no training required)
    # "disabled"      — rule-based DisabledCoverageJudge fallback
    #
    # Default: litellm + OpenRouter Llama-3.3-70B-Instruct (free tier).
    #
    # Why OpenRouter and not Groq: Groq's free tier TPM throttles new
    # accounts hard (Polyakov re-runs got 6K TPM, not the advertised
    # 30K). OpenRouter's free tier is request-based (20 RPM) which
    # fits our 41-pair pipeline comfortably — no token bookkeeping
    # needed.
    #
    # Free-tier daily quota:
    #   * 50 requests/day with no credit on file
    #   * 1000 requests/day after a one-time $5 top-up (lifetime,
    #     not a subscription) — recommended if you'll run more than
    #     1 package per day.
    #
    # Setup:
    #   1. Register at https://openrouter.ai/
    #   2. https://openrouter.ai/keys → Create Key
    #   3. setx OPENROUTER_API_KEY "sk-or-v1-..."
    #   4. (optional) top up $5 to lift daily limit to 1000 reqs
    #
    # OpenRouter rotates :free model availability frequently — pinning
    # any specific :free id is fragile. Confirmed-working in this
    # operator's session (we saw real verdicts from it):
    #   meta-llama/llama-3.3-70b-instruct:free — works; occasionally
    #   gets upstream rate-limited under shared peak load. Our retry
    #   logic (24a8310 + 1cbc1c4) handles those by sleeping 30-60s
    #   instead of falling back to disabled judge.
    #
    # Confirmed REMOVED from free tier (DO NOT use):
    #   qwen/qwen-2.5-72b-instruct:free          — 404 (2026-05-07)
    #   deepseek/deepseek-chat-v3-0324:free      — 404 (2026-05-07)
    #   google/gemma-2-9b-it:free                — superseded by gemma-3
    #
    # Always check the live list before pinning:
    #   https://openrouter.ai/models?max_price=0
    #
    # Per-request override (recommended for trying alternatives):
    #   options.llm_model_name = "openrouter/google/gemma-3-27b-it:free"
    #   options.llm_model_name = "openrouter/meta-llama/llama-4-scout:free"
    #
    # Or stick with Groq: model_name = "groq/llama-3.1-8b-instant"
    # AND set CQUALITY_LITELLM_TPM=5500 to enable token-bucket throttle.
    #
    # Default reverted to local Ollama (qwen2.5:3b): no rate limits, no
    # API keys, no upstream rotation. Quality is lower than 70B-class
    # cloud models (qwen-3b sometimes ставит ложный COVERED on
    # near-verbatim PMI without methodology — but the deterministic
    # PMI-copy-without-methodology rule from 8bfd165 catches that),
    # but for iterative development on real packages predictable local
    # inference beats cloud-rate-limit-roulette.
    #
    # Switch to cloud per-request via options.llm_model_name, or change
    # backend = "litellm" to make it the global default.
    backend: str = "ollama"
    # P0 #8 (all 5 packages): pinned to a ≥ 7B model for the demo.
    # qwen2.5:3b produced unstable confidence ≥ 0.7 on marginal matches —
    # surface lexical overlap pushed it into confident COVERED on pairs
    # that 7B+ models correctly call PARTIAL/IRRELEVANT. The 7B variant
    # is the smallest size that consistently respects the prompt's
    # grounding contract (cited_phrases must be substrings of the
    # evidence) without the small-model echo-and-fabricate failure mode.
    # Override per-deployment with CQUALITY_LLM_MODEL_NAME or per-request
    # via options.llm_model_name.
    model_name: str = "qwen2.5:7b"
    prompt_version: str = "v1"
    # Polyakov-regression (2026-05-10): bumped 120 → 240 s. On Ollama
    # qwen2.5:7b with parallelism=1 the 90th-percentile pair latency is
    # ~110-180 s; the previous 120 s budget had ~15% timeout rate (57/60
    # on the May-10 Polyakov re-run). 240 s absorbs the long-prompt tail
    # without making median wall-clock worse. CQUALITY_JUDGE_TIMEOUT env
    # overrides at runtime in the Ollama wrapper.
    timeout: int = 240
    # Thresholds for cross_encoder backend. Calibrated for BAAI/bge-reranker-v2-m3
    # sigmoid output. Tune against a manually reviewed sample on real packages.
    cross_encoder_covered_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    cross_encoder_partial_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class CoverageEmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # "bow"        — lexical bag-of-words, no ML dependency
    # "transformer" — local HF checkpoint at model_path (default ./model)
    # "e5"         — multilingual-e5-base via HF hub (domain-optimised
    #                for semantic similarity, 278M params)
    backend: str = "transformer"
    model_path: Optional[str] = "./model"
    e5_model_name: str = "intfloat/multilingual-e5-base"


class CoverageRerankerConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Default ON: BGE reranker in conditional mode is a strict quality
    # win on real packages (Polyakov + others). When BGE isn't available
    # at startup the pipeline silently falls back (NoopReranker), so this
    # default is safe even on dev machines without sentence-transformers.
    # Override per-request via options.enable_reranker = false.
    enabled: bool = True
    backend: str = "bge"  # "bge" | "disabled"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_len: int = Field(default=512, ge=64, le=1024)
    batch_size: int = Field(default=16, ge=1, le=128)
    # PR-K: when reranker is enabled, choose between unconditional and
    # signal-driven application:
    #   "always"      — rerank every shortlist (legacy behaviour).
    #   "conditional" — rerank only when first-stage signals are weak
    #                   (top1 < strong threshold, top1-top2 < margin,
    #                    requirement critical, paraphrase indicated by
    #                    high semantic but low lexical, etc.).
    # If reranker.enabled is False the mode is irrelevant.
    mode: str = "conditional"
    # Conditional-mode thresholds.
    conditional_top1_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    conditional_min_margin: float = Field(default=0.08, ge=0.0, le=1.0)


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
    reranker: CoverageRerankerConfig = Field(default_factory=CoverageRerankerConfig)
    requirement_model: RequirementModelConfig = Field(
        default_factory=RequirementModelConfig
    )
    # PR-K: aggregator and explainability tuning.
    aggregator: CoverageAggregatorConfig = Field(default_factory=CoverageAggregatorConfig)
    debug: CoverageDebugConfig = Field(default_factory=CoverageDebugConfig)
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
    def from_env(cls) -> "CoverageConfig":
        """Build a CoverageConfig from defaults and apply env-var
        overrides. Honoured variables (each optional, defaults to the
        field default when unset / unparseable):

          * CQUALITY_MIN_RETRIEVAL_SCORE — float in [0, 1]; lowers or
            raises the per-shortlist retrieval gate.
          * CQUALITY_EVIDENCE_FLOOR     — float in [0, 1]; below this
            the LLM verdict is flagged low_confidence on the row.
          * CQUALITY_LLM_MODEL_NAME    — already honoured by the
            startup warmup (P0 #8); also propagated here so the
            LiteLLM-backed runs see the same model.

        Polyakov-regression motivation: real ВКР packages whose ПЗ
        paraphrases the ТЗ heavily often have max retrieval score
        0.20-0.34 — below the 0.05 default of min_retrieval_score
        the row is processed, but the dominant lexical bias drops
        many genuine paraphrases below the (downstream-applied)
        evidence_floor of 0.30. Lowering both via env vars at
        deployment time is the safest knob; the alternative is to
        hard-code a smaller default and risk false COVERED on
        unrelated lexically-similar pairs.
        """
        config = cls()
        import os

        def _float_env(name: str, lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
            raw = os.environ.get(name)
            if raw is None:
                return None
            try:
                v = float(raw)
            except ValueError:
                return None
            if v < lo or v > hi:
                return None
            return v

        v = _float_env("CQUALITY_MIN_RETRIEVAL_SCORE")
        if v is not None:
            config.retrieval.min_retrieval_score = v
        v = _float_env("CQUALITY_EVIDENCE_FLOOR")
        if v is not None:
            config.retrieval.evidence_floor = v
        model = os.environ.get("CQUALITY_LLM_MODEL_NAME")
        if model:
            config.llm.model_name = model
        return config

    @classmethod
    def from_options(cls, options: Dict) -> "CoverageConfig":
        config = cls()
        if "top_k" in options:
            config.retrieval.top_k = int(options["top_k"])
        if "enable_llm_judge" in options:
            config.llm.enabled = bool(options["enable_llm_judge"])
            # Legacy compat: when an orchestrator says "enable LLM" without
            # specifying a backend AND the service default is the special
            # "disabled" backend, fall back to ollama (the original
            # behaviour). When the service default is already a real
            # backend (litellm, ollama, cross_encoder), preserve it so
            # the operator's choice in config.py wins.
            if config.llm.enabled and config.llm.backend == "disabled":
                config.llm.backend = "ollama"
        if "judge_backend" in options:
            backend = str(options["judge_backend"]).lower()
            if backend in {"ollama", "cross_encoder", "disabled", "litellm"}:
                config.llm.backend = backend
                config.llm.enabled = backend != "disabled"
        # PR-J: explicit LLM model name override (e.g. when backend is
        # "litellm" we need the routing string "groq/llama-3.3-70b-versatile").
        if "llm_model_name" in options:
            config.llm.model_name = str(options["llm_model_name"])
        if "enable_rule_verification" in options:
            config.enable_rule_verification = bool(options["enable_rule_verification"])
        if "min_retrieval_score" in options:
            config.retrieval.min_retrieval_score = float(options["min_retrieval_score"])
        if "evidence_floor" in options:
            config.retrieval.evidence_floor = float(options["evidence_floor"])
        if "skip_llm_below_floor" in options:
            config.retrieval.skip_llm_below_floor = bool(
                options["skip_llm_below_floor"]
            )
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
        if "embedding_backend" in options:
            backend = str(options["embedding_backend"]).lower()
            if backend in {"bow", "transformer", "e5"}:
                config.embedding.backend = backend
        if "enable_reranker" in options:
            config.reranker.enabled = bool(options["enable_reranker"])
        if "top_k_before_rerank" in options:
            config.retrieval.top_k_before_rerank = int(options["top_k_before_rerank"])
        # PR-K options.
        if "initial_top_n" in options:
            config.retrieval.initial_top_n = int(options["initial_top_n"])
        if "reranker_mode" in options:
            mode = str(options["reranker_mode"]).lower()
            if mode in {"always", "conditional"}:
                config.reranker.mode = mode
        if "debug" in options:
            config.debug.enabled = bool(options["debug"])
        if "debug_max_candidates" in options:
            config.debug.max_candidates = int(options["debug_max_candidates"])
        if "debug_include_discarded" in options:
            config.debug.include_discarded = bool(options["debug_include_discarded"])
        if "covered_confidence_threshold" in options:
            config.aggregator.covered_confidence_threshold = float(
                options["covered_confidence_threshold"]
            )
        if "conflict_confidence_threshold" in options:
            config.aggregator.conflict_confidence_threshold = float(
                options["conflict_confidence_threshold"]
            )
        if "partial_confidence_threshold" in options:
            config.aggregator.partial_confidence_threshold = float(
                options["partial_confidence_threshold"]
            )
        return config
