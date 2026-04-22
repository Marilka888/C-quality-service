"""
Full coverage analysis pipeline:
  requirement_builder
    → coverage_unit_builder
    → retrieval (hybrid)
    → shortlist (top-K)
    → LLM judge
    → rule verifier
    → aggregator
    → report builder
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from app.application.use_cases.aggregate_coverage import CoverageAggregator
from app.application.use_cases.build_coverage_report import CoverageReportBuilder
from app.application.use_cases.build_coverage_units import CoverageUnitBuilder
from app.application.use_cases.build_requirements import RequirementBuilder
from app.application.use_cases.judge_pairs import PairJudgeService
from app.application.use_cases.retrieve_candidates import CandidateRetriever
from app.application.use_cases.verify_pairs import PairVerifier
from app.core.config import CoverageConfig
from app.core.logging import get_logger
from app.domain.c_quality_models import (
    CoverageAnalysisResult,
    CoverageUnit,
    PairJudgment,
    RequirementCoverageResult,
    RetrievedCandidate,
)
from app.infrastructure.embeddings.base import EmbeddingBackend
from app.infrastructure.embeddings.simple import BagOfWordsEmbeddingBackend
from app.infrastructure.llm.coverage_judge import CoverageJudge
from app.infrastructure.llm.disabled_coverage_judge import DisabledCoverageJudge
from app.infrastructure.llm.ollama_coverage_judge import OllamaCoverageJudge

logger = get_logger(__name__)


def _build_embedding_backend(config: CoverageConfig) -> EmbeddingBackend:
    backend = (config.embedding.backend or "transformer").lower()
    if backend == "e5":
        try:
            from app.infrastructure.embeddings.e5 import E5EmbeddingBackend
            return E5EmbeddingBackend(model_name=config.embedding.e5_model_name)
        except Exception as exc:
            logger.warning("E5EmbeddingBackend failed (%s); falling back to BoW", exc)
            return BagOfWordsEmbeddingBackend()
    if backend == "transformer":
        model_path = config.embedding.model_path or "./model"
        resolved = Path(model_path)
        if resolved.exists():
            try:
                from app.infrastructure.embeddings.transformer import TransformerEmbeddingBackend
                return TransformerEmbeddingBackend(str(resolved))
            except Exception as exc:
                logger.warning("TransformerEmbeddingBackend failed (%s); falling back to BoW", exc)
    return BagOfWordsEmbeddingBackend()


def _build_reranker(config: CoverageConfig):
    """Return a Reranker or None when reranking is disabled.

    None signals `CandidateRetriever` to use its internal NoopReranker, so
    the zero-dependency path stays clean.
    """
    if not config.reranker.enabled:
        return None
    backend = (config.reranker.backend or "bge").lower()
    if backend == "bge":
        try:
            from app.infrastructure.reranker.bge import BGEReranker
            return BGEReranker(
                model_name=config.reranker.model_name,
                max_len=config.reranker.max_len,
                batch_size=config.reranker.batch_size,
            )
        except Exception as exc:
            logger.warning("BGEReranker failed (%s); reranker disabled", exc)
            return None
    return None


def _build_judge(config: CoverageConfig) -> CoverageJudge:
    if config.llm.enabled and config.llm.backend == "ollama":
        return OllamaCoverageJudge(
            model_name=config.llm.model_name,
            timeout=config.llm.timeout,
        )
    return DisabledCoverageJudge()


class CoverageAnalysisPipeline:
    """Stateless orchestrator — create once, call run() per request."""

    def __init__(self, config: Optional[CoverageConfig] = None) -> None:
        self._config = config or CoverageConfig()
        self._req_builder = RequirementBuilder(self._config)
        self._unit_builder = CoverageUnitBuilder()
        self._embedding_backend = _build_embedding_backend(self._config)
        self._reranker = _build_reranker(self._config)
        self._retriever = CandidateRetriever(
            self._config.retrieval, self._embedding_backend, self._reranker,
        )
        self._judge_service = PairJudgeService(_build_judge(self._config))
        self._verifier = PairVerifier()
        self._aggregator = CoverageAggregator()
        self._report_builder = CoverageReportBuilder()

    # ------------------------------------------------------------------

    def run(self, request: dict) -> CoverageAnalysisResult:
        job_id = request.get("job_id") or str(uuid.uuid4())
        package_id = request.get("package_id", "unknown")
        source_role = (request.get("source_doc_role") or "tz").lower()
        target_roles = {r.lower() for r in (request.get("target_doc_roles") or ["pmi", "pz"])}
        documents = request.get("documents") or []
        options = request.get("options") or {}

        # Allow per-request config overrides
        config = CoverageConfig.from_options(options) if options else self._config

        # Judge must honor per-request enable_llm_judge. The default
        # _judge_service was built from self._config at construction time and
        # would always be DisabledCoverageJudge; rebuild here if the effective
        # config differs.
        if config.llm.enabled != self._config.llm.enabled or config.llm.backend != self._config.llm.backend:
            judge_service = PairJudgeService(_build_judge(config))
        else:
            judge_service = self._judge_service

        if config.requirement_extraction != self._config.requirement_extraction:
            req_builder = RequirementBuilder(config)
        else:
            req_builder = self._req_builder

        warnings: List[str] = []

        # ── Step 1: split documents by role ──────────────────────────────
        source_artifact: Optional[dict] = None
        target_artifacts: List[dict] = []

        for doc in documents:
            artifact = doc.get("prepared_artifact") or {}
            role = (doc.get("doc_role") or artifact.get("doc_role") or "").lower()
            if not artifact.get("document_id"):
                artifact["document_id"] = doc.get("document_id", str(uuid.uuid4()))
            if not artifact.get("doc_role"):
                artifact["doc_role"] = role

            if role == source_role:
                source_artifact = artifact
            elif role in target_roles:
                target_artifacts.append(artifact)

        if source_artifact is None:
            warnings.append(f"No document with role '{source_role}' found")
            return self._report_builder.build(
                job_id=job_id,
                package_id=package_id,
                source_document_id="unknown",
                requirement_results=[],
                warnings=warnings,
            )
        if not target_artifacts:
            warnings.append(f"No target documents found for roles {target_roles}")

        # ── Step 2: build RequirementUnits from TZ ────────────────────────
        requirements = req_builder.build(source_artifact)
        logger.info("[%s] Built %d requirements from TZ", job_id, len(requirements))
        if not requirements:
            warnings.append("No requirements extracted from source document")

        # ── Step 3: build CoverageUnits from each target document ─────────
        all_units: List[CoverageUnit] = []
        for artifact in target_artifacts:
            units = self._unit_builder.build(artifact)
            all_units.extend(units)
            logger.info(
                "[%s] Built %d coverage_units from %s (%s)",
                job_id,
                len(units),
                artifact.get("document_id"),
                artifact.get("doc_role"),
            )
            if not units:
                raw_frags_count = len(artifact.get("fragments") or artifact.get("sentences") or [])
                warnings.append(
                    f"No coverage units built from target document "
                    f"{artifact.get('document_id')} (role={artifact.get('doc_role')}). "
                    f"raw_fragments={raw_frags_count}. "
                    f"Possible causes: (1) fragments[] and sentences[] both empty in prepared_artifact; "
                    f"(2) all fragments have < 2 words (check prepare-service output)."
                )

        units_by_id: Dict[str, CoverageUnit] = {u.unit_id: u for u in all_units}
        logger.info("[%s] Total coverage_units across all targets: %d", job_id, len(all_units))

        # ── Steps 4-7: retrieval → judge → verify → aggregate ─────────────
        all_results: List[RequirementCoverageResult] = []
        all_judgments: List[PairJudgment] = []

        _log_sample_req_ids = [r.req_id for r in requirements[:3]]
        logger.info("[%s] Sample req_ids (first 3): %s", job_id, _log_sample_req_ids)

        _total_shortlisted = 0

        for req_i, req in enumerate(requirements):
            # Retrieve candidates per target document (keep them separate for per-doc reporting)
            doc_candidates: Dict[str, List[RetrievedCandidate]] = {}
            for artifact in target_artifacts:
                doc_id = artifact["document_id"]
                doc_units = [u for u in all_units if u.target_document_id == doc_id]
                candidates = self._retriever.retrieve(req, doc_units)
                doc_candidates[doc_id] = candidates
                if req_i < 3:
                    sample_scores = [round(c.retrieval_score, 3) for c in candidates[:3]]
                    logger.debug(
                        "[%s] req[%d]=%s → doc=%s shortlist=%d sample_scores=%s",
                        job_id, req_i, req.req_id[:12], doc_id,
                        len(candidates), sample_scores,
                    )

            for artifact in target_artifacts:
                doc_id = artifact["document_id"]
                doc_role = artifact.get("doc_role", "unknown")
                shortlist = doc_candidates.get(doc_id, [])
                _total_shortlisted += len(shortlist)

                if not shortlist:
                    logger.debug(
                        "[%s] No candidates for req=%s target=%s (all below threshold or no units)",
                        job_id, req.req_id[:12], doc_id,
                    )
                    all_results.append(
                        RequirementCoverageResult(
                            req_id=req.req_id,
                            source_document_id=req.source_document_id,
                            target_document_id=doc_id,
                            target_doc_role=doc_role,
                        )
                    )
                    continue

                # Judge
                judgments = judge_service.judge_shortlist(req, shortlist, units_by_id)

                # Verify
                if config.enable_rule_verification:
                    judgments = [
                        self._verifier.verify(j, req, units_by_id[j.unit_id])
                        for j in judgments
                        if j.unit_id in units_by_id
                    ]

                all_judgments.extend(judgments)

                # Aggregate
                candidates_by_unit_id = {c.unit_id: c for c in shortlist}
                result = self._aggregator.aggregate(
                    requirement=req,
                    judgments=judgments,
                    candidates_by_unit_id=candidates_by_unit_id,
                    units_by_id=units_by_id,
                    target_document_id=doc_id,
                    target_doc_role=doc_role,
                )
                all_results.append(result)

        logger.info(
            "[%s] Pipeline done: requirements=%d, coverage_units=%d, "
            "total_shortlisted=%d, pair_judgments=%d",
            job_id, len(requirements), len(all_units),
            _total_shortlisted, len(all_judgments),
        )

        if not all_judgments and all_results and requirements:
            target_doc_ids_str = ", ".join(a["document_id"] for a in target_artifacts)
            warnings.append(
                f"No candidate pairs survived retrieval for any of the {len(requirements)} requirements "
                f"(target_docs=[{target_doc_ids_str}], coverage_units={len(all_units)}, "
                f"min_retrieval_score={config.retrieval.min_retrieval_score}). "
                f"All {len(all_results)} results are MISSING. "
                f"Possible causes: threshold too high, empty PMI fragments, or vocabulary mismatch."
            )
            logger.warning(
                "[%s] No candidate pairs found across all requirements. "
                "coverage_units=%d, min_retrieval_score=%s",
                job_id, len(all_units), config.retrieval.min_retrieval_score,
            )

        source_doc_id = source_artifact.get("document_id", "unknown")
        return self._report_builder.build(
            job_id=job_id,
            package_id=package_id,
            source_document_id=source_doc_id,
            requirement_results=all_results,
            pair_judgments=all_judgments,
            warnings=warnings,
        )
