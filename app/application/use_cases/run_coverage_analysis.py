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

import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.application.use_cases.adaptive_candidate_selector import (
    SelectionResult,
    select_candidates,
)
from app.application.use_cases.aggregate_coverage import CoverageAggregator
from app.application.use_cases.applicability import (
    applicability_for,
    coverage_requirement_level_for,
    severity_for,
    should_affect_critical,
    should_affect_grade,
)
from app.application.use_cases.build_coverage_report import CoverageReportBuilder
from app.application.use_cases.build_coverage_units import CoverageUnitBuilder
from app.application.use_cases.build_requirements import RequirementBuilder
from app.application.use_cases.judge_pairs import PairJudgeService
from app.application.use_cases.retrieve_candidates import CandidateRetriever
from app.application.use_cases.verify_pairs import PairVerifier
from app.core.config import CoverageConfig
from app.core.logging import get_logger
from app.domain.c_quality_enums import (
    Applicability,
    CoverageRequirementLevel,
    CoverageStatus,
    RequirementType,
)
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

# ── Cross-requirement parallelism (PR-K post-fix B) ─────────────────────────
# CQUALITY_REQ_CONCURRENCY controls how many requirement-workers run in
# parallel. Each worker handles one requirement through the full
# retrieval → judge → verify → aggregate chain independently.
# Combined with per-pair judge concurrency (CQUALITY_JUDGE_CONCURRENCY)
# the total concurrent LLM calls = req_workers × judge_workers. For a
# typical single-GPU Ollama setup, req=3 + judge=1 (or req=2 + judge=2)
# is the sweet spot.
#
# Thread-safety notes for callers of this path:
#   * CandidateRetriever / PairVerifier / CoverageAggregator — stateless,
#     thread-safe with no extra work.
#   * OllamaCoverageJudge.unavailable_count — incremented without a lock,
#     but it is purely a telemetry counter (no correctness risk); Python
#     GIL protects the int object from corruption.
#   * JudgmentCache — already uses per-call SQLite connections with WAL
#     mode + a write lock. Safe for concurrent use from multiple threads.
_REQ_CONCURRENCY_HARD_CAP = 8
_REQ_CONCURRENCY_DEFAULT = 3
_FLOOR_SKIP_EXEMPT_TYPES = {
    RequirementType.SECURITY,
    RequirementType.PERFORMANCE,
    RequirementType.RELIABILITY,
}


def _can_skip_llm_below_floor(req: RequirementUnit) -> bool:
    return req.requirement_type not in _FLOOR_SKIP_EXEMPT_TYPES and not req.constraints


def _resolve_req_concurrency() -> int:
    """Read CQUALITY_REQ_CONCURRENCY, validate, clamp to [1, cap]."""
    raw = os.environ.get("CQUALITY_REQ_CONCURRENCY", str(_REQ_CONCURRENCY_DEFAULT)).strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "CQUALITY_REQ_CONCURRENCY=%r is not an int; using 1", raw,
        )
        return 1
    if n < 1:
        return 1
    if n > _REQ_CONCURRENCY_HARD_CAP:
        logger.warning(
            "CQUALITY_REQ_CONCURRENCY=%d > hard cap %d; clamping",
            n, _REQ_CONCURRENCY_HARD_CAP,
        )
        return _REQ_CONCURRENCY_HARD_CAP
    return n


# Markers we expect in any normative TZ text. Used by the sanity guard
# to estimate whether the extracted-count looks suspiciously low.
_REQUIREMENT_MARKER_RE = __import__("re").compile(
    r"\b(должн[аоы]|должен|необходим[ао]?|требуется|следует|обязан\w*|"
    r"не\s+должн[аоы]|не\s+должен|обеспечивать|реализовывать)\b",
    __import__("re").IGNORECASE | __import__("re").UNICODE,
)


def _build_extraction_diagnostics(
    source_artifact: dict,
    requirements: List,
) -> dict:
    """Compute extraction-coverage diagnostics for the TZ source.

    The sanity check: count "must/shall"-class markers in the source
    text and compare to the number of RequirementUnit objects the
    builder produced. Returns a dict with:
      * extracted_count            — len(requirements)
      * marker_count               — number of regex hits in raw text
      * sections_seen              — total sections in artifact
      * requirement_sections_seen  — sections whose category looks
                                     requirement-bearing (best-effort,
                                     based on `category` metadata)
      * sections_per_extracted_req — distribution by source_section_id
      * low_extraction_coverage    — True when extracted_count is
                                     << marker_count
      * suspected_reason           — short Russian hint
    """
    sections = source_artifact.get("sections") or []
    fragments = source_artifact.get("fragments") or []
    candidates = source_artifact.get("requirement_candidates") or []

    raw_texts: List[str] = []
    for s in sections:
        t = s.get("text") if isinstance(s, dict) else getattr(s, "text", None)
        if t:
            raw_texts.append(str(t))
    for f in fragments:
        t = f.get("text") if isinstance(f, dict) else getattr(f, "text", None)
        if t:
            raw_texts.append(str(t))
    if not raw_texts and candidates:
        for c in candidates:
            t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if t:
                raw_texts.append(str(t))

    full_text = "\n".join(raw_texts)
    marker_count = len(_REQUIREMENT_MARKER_RE.findall(full_text))
    extracted_count = len(requirements)

    requirement_sections_seen = 0
    for s in sections:
        cat = s.get("category") if isinstance(s, dict) else getattr(s, "category", None)
        if cat in {"requirements", "test_methods", "input_output", "environment"}:
            requirement_sections_seen += 1

    by_section: dict[str, int] = {}
    for r in requirements:
        sid = r.source_section_id or "(none)"
        by_section[sid] = by_section.get(sid, 0) + 1

    # Heuristic: low coverage if marker_count >= 10 and extracted_count
    # is below 25% of markers (a typical TZ extracts ≥30% of markers
    # as candidates after de-duplication and filtering).
    low_coverage = marker_count >= 10 and extracted_count * 4 < marker_count

    reason = ""
    if low_coverage:
        if requirement_sections_seen <= 1 and extracted_count <= 5:
            reason = (
                "Главный раздел требований ТЗ не распознан как "
                "requirement-bearing — вероятно, заголовки оформлены "
                "не Word-стилем, а нумерованным списком/обычным "
                "параграфом."
            )
        else:
            reason = (
                "Высокая концентрация requirement-маркеров при низком "
                "числе извлечённых требований. Проверьте section_category "
                "и фильтр кандидатов."
            )

    return {
        "extracted_count": extracted_count,
        "marker_count": marker_count,
        "sections_seen": len(sections),
        "requirement_sections_seen": requirement_sections_seen,
        "sections_per_extracted_req": by_section,
        "low_extraction_coverage": low_coverage,
        "suspected_reason": reason,
    }


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


def _build_judge(config: CoverageConfig, reranker=None) -> CoverageJudge:
    if config.llm.enabled and config.llm.backend == "ollama":
        return OllamaCoverageJudge(
            model_name=config.llm.model_name,
            timeout=config.llm.timeout,
        )
    if config.llm.enabled and config.llm.backend == "litellm":
        # PR-J: unified multi-provider judge. `model_name` is a LiteLLM
        # routing string ("groq/llama-3.3-70b-versatile" /
        # "gemini/gemini-2.0-flash" / "openai/gpt-4o-mini" / etc.). API
        # keys come from the environment per LiteLLM's convention.
        from app.infrastructure.llm.litellm_coverage_judge import LiteLLMCoverageJudge
        return LiteLLMCoverageJudge(
            model_name=config.llm.model_name,
            timeout=config.llm.timeout,
        )
    if config.llm.enabled and config.llm.backend == "cross_encoder":
        # Reuse the already-built reranker when available to avoid loading
        # BGE twice. If the reranker isn't available (rerank disabled by
        # config), build a fresh BGE instance purely for the judge.
        judge_reranker = reranker
        if judge_reranker is None:
            try:
                from app.infrastructure.reranker.bge import BGEReranker
                judge_reranker = BGEReranker(
                    model_name=config.reranker.model_name,
                    max_len=config.reranker.max_len,
                    batch_size=config.reranker.batch_size,
                )
            except Exception as exc:
                logger.warning(
                    "cross_encoder judge: BGE unavailable (%s); falling back to DisabledCoverageJudge",
                    exc,
                )
                return DisabledCoverageJudge()
        from app.infrastructure.llm.cross_encoder_coverage_judge import (
            CrossEncoderCoverageJudge,
        )
        return CrossEncoderCoverageJudge(
            reranker=judge_reranker,
            covered_threshold=config.llm.cross_encoder_covered_threshold,
            partial_threshold=config.llm.cross_encoder_partial_threshold,
        )
    return DisabledCoverageJudge()


class CoverageAnalysisPipeline:
    """Stateless orchestrator — create once, call run() per request."""

    def __init__(self, config: Optional[CoverageConfig] = None) -> None:
        # Default constructor honours env-var overrides — this is the
        # only way a deployment can lower min_retrieval_score /
        # evidence_floor without per-request flags or code changes.
        # Polyakov-class packages need 0.15-0.20 thresholds on the BoW
        # retriever to stop dropping real paraphrases below the floor.
        self._config = config or CoverageConfig.from_env()
        self._req_builder = RequirementBuilder(self._config)
        self._unit_builder = CoverageUnitBuilder()
        self._embedding_backend = _build_embedding_backend(self._config)
        self._reranker = _build_reranker(self._config)
        self._retriever = CandidateRetriever(
            self._config.retrieval,
            self._embedding_backend,
            self._reranker,
            reranker_config=self._config.reranker,
        )
        self._judge_service = PairJudgeService(
            _build_judge(self._config, reranker=self._reranker)
        )
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
            judge_service = PairJudgeService(_build_judge(config, reranker=self._reranker))
        else:
            judge_service = self._judge_service

        if config.requirement_extraction != self._config.requirement_extraction:
            req_builder = RequirementBuilder(config)
        else:
            req_builder = self._req_builder

        warnings: List[str] = []

        # ── Step 1: split documents by role ──────────────────────────────
        # B3: defensive normalisation. The FastAPI request schema already
        # rejects unknown roles with a 422 (DocRole Literal), so this branch
        # only fires when the use case is invoked directly (tests, scripts,
        # internal callers that bypass the API). In that case we coerce
        # unknown strings to "unknown" and emit a warning so the run is
        # observable rather than silently dropping the document.
        _ALLOWED_ROLES = {"tz", "pmi", "pz", "unknown"}
        source_artifact: Optional[dict] = None
        target_artifacts: List[dict] = []

        for doc in documents:
            artifact = doc.get("prepared_artifact") or {}
            raw_role = (doc.get("doc_role") or artifact.get("doc_role") or "").strip().lower()
            if raw_role and raw_role not in _ALLOWED_ROLES:
                warnings.append(
                    f"INVALID_DOC_ROLE: document {doc.get('document_id', '?')} has "
                    f"unrecognised doc_role={raw_role!r}; treating as 'unknown' and "
                    f"excluding from C-quality matching"
                )
                logger.warning(
                    "[%s] invalid doc_role=%r for doc=%s; coerced to 'unknown'",
                    job_id, raw_role, doc.get("document_id", "?"),
                )
                raw_role = "unknown"
            role = raw_role
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

        # ── Step 1b: drop targets that point at the same document as the
        # source (self-comparison guard, BUG-03). Comparing a TZ with itself
        # returns artificially high coverage and is never meaningful. We use
        # document_id as the stable identifier and fall back to a
        # (filename, doc_role) tuple for legacy payloads that don't propagate
        # document_id through prepared_artifact. Role match is required for
        # the fallback so that a TZ↔PMI pair where filenames happen to match
        # (rare but possible) is not falsely dropped.
        def _self_target_key(art: dict) -> tuple:
            doc_id = art.get("document_id") or ""
            filename = art.get("filename") or ""
            role = (art.get("doc_role") or "").lower()
            return (doc_id, filename, role)

        src_key = _self_target_key(source_artifact)
        deduped_targets: List[dict] = []
        for art in target_artifacts:
            tgt_key = _self_target_key(art)
            same_id = bool(src_key[0]) and src_key[0] == tgt_key[0]
            same_fallback = (
                not src_key[0]
                and not tgt_key[0]
                and src_key[1]
                and src_key[1] == tgt_key[1]
                and src_key[2] == tgt_key[2]
            )
            if same_id or same_fallback:
                warnings.append(
                    f"SELF_COMPARISON_SKIPPED: target document_id={tgt_key[0] or '?'} "
                    f"role={tgt_key[2] or '?'} matches source — pair dropped to avoid TZ↔TZ "
                    f"self-comparison"
                )
                logger.warning(
                    "[%s] self-comparison guard dropped target doc_id=%s role=%s "
                    "(matches source doc_id=%s)",
                    job_id, tgt_key[0] or "?", tgt_key[2] or "?", src_key[0] or "?",
                )
                continue
            deduped_targets.append(art)
        target_artifacts = deduped_targets

        if not target_artifacts:
            warnings.append(f"No target documents found for roles {target_roles}")

        # ── Step 2: build RequirementUnits from TZ ────────────────────────
        requirements = req_builder.build(source_artifact)
        logger.info("[%s] Built %d requirements from TZ", job_id, len(requirements))
        if not requirements:
            warnings.append("No requirements extracted from source document")

        # PR-I sanity guard: when the TZ source text contains many
        # requirement markers but few units made it through extraction,
        # the pipeline has likely missed the main "Требования к
        # программе" section. This is the audit-time symptom on the
        # «Череухо» package — 39 "должн*" markers in source, only 3
        # candidates extracted because Word headings were inconsistent
        # and 12 KB of body text landed in a heading-less preamble.
        # We don't fail the run — the orchestrator may still want a
        # report — but we emit a structured warning + diagnostics so
        # the UI can flag the package for manual review and downstream
        # severity rollups can stay conservative.
        try:
            extr_diag = _build_extraction_diagnostics(source_artifact, requirements)
        except Exception as exc:
            logger.warning("extraction diagnostics failed: %s", exc)
            extr_diag = {}
        if extr_diag.get("low_extraction_coverage"):
            warnings.append(
                f"LOW_REQUIREMENT_EXTRACTION_COVERAGE: TZ source contains "
                f"{extr_diag.get('marker_count', 0)} requirement markers "
                f"('должн*' / 'необходимо' / 'требуется') but only "
                f"{extr_diag.get('extracted_count', 0)} requirements were "
                f"extracted. {extr_diag.get('suspected_reason', '')} "
                f"Coverage results may be incomplete; manual review "
                f"recommended."
            )

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
        req_concurrency = _resolve_req_concurrency()
        workers = min(req_concurrency, len(requirements)) if requirements else 1

        if workers <= 1 or len(requirements) <= 1:
            # ── Serial path (unchanged behaviour) ────────────────────────
            for req_i, req in enumerate(requirements):
                _, req_results, req_judgments, req_shortlisted = (
                    self._process_one_requirement(
                        req_i, req, target_artifacts, all_units, units_by_id,
                        judge_service, config, job_id,
                    )
                )
                all_results.extend(req_results)
                all_judgments.extend(req_judgments)
                _total_shortlisted += req_shortlisted
        else:
            # ── Parallel path (PR-K post-fix B) ──────────────────────────
            # Fan-out across requirement workers. Each worker runs the full
            # retrieval → select → judge → verify → aggregate chain for one
            # requirement. Results are merged in original input order so the
            # report is deterministic.
            logger.info(
                "[%s] Parallel requirement processing: %d requirements, "
                "%d workers (CQUALITY_REQ_CONCURRENCY=%s)",
                job_id, len(requirements), workers,
                os.environ.get("CQUALITY_REQ_CONCURRENCY", str(_REQ_CONCURRENCY_DEFAULT)),
            )
            indexed: List[Optional[Tuple]] = [None] * len(requirements)
            # Polyakov-regression: track per-worker failures so the
            # final report carries a WARN about silently-excluded
            # requirements. Without this, exceptions inside
            # _process_one_requirement disappear into ERROR-level logs
            # only and the user sees a 31-requirements ТЗ collapse to
            # 3 results with no on-report explanation.
            worker_failures: list[tuple[int, str, str]] = []
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="req-worker",
            ) as pool:
                future_to_idx = {
                    pool.submit(
                        self._process_one_requirement,
                        req_i, req, target_artifacts, all_units, units_by_id,
                        judge_service, config, job_id,
                    ): req_i
                    for req_i, req in enumerate(requirements)
                }
                for fut in as_completed(future_to_idx):
                    req_i = future_to_idx[fut]
                    try:
                        _, req_results, req_judgments, req_shortlisted = fut.result()
                        indexed[req_i] = (req_results, req_judgments, req_shortlisted)
                    except Exception as exc:
                        req_id_short = (
                            requirements[req_i].req_id[:24]
                            if req_i < len(requirements) else "?"
                        )
                        worker_failures.append(
                            (req_i, req_id_short, f"{type(exc).__name__}: {exc}")
                        )
                        logger.error(
                            "[%s] req-worker[%d] failed — requirement excluded "
                            "from results: %s",
                            job_id, req_i, exc, exc_info=True,
                        )
                        indexed[req_i] = ([], [], 0)
            if worker_failures:
                # Surface the failure count in a single WARN entry so
                # the orchestrator / UI can show "N requirements
                # silently excluded due to worker error" and the user
                # can investigate via the C-quality service logs. Cap
                # the per-failure detail at 5 entries so the warning
                # stays readable; the rest live only in the log stream.
                detail_lines = [
                    f"req[{i}] {rid}: {err}"
                    for i, rid, err in worker_failures[:5]
                ]
                tail = (
                    f" (+ {len(worker_failures) - 5} more — see C-quality logs)"
                    if len(worker_failures) > 5 else ""
                )
                warnings.append(
                    f"WORKER_EXCLUSIONS: {len(worker_failures)} requirement(s) "
                    f"silently excluded due to per-worker exceptions in "
                    f"_process_one_requirement. The corresponding (req × target) "
                    f"rows are missing from the result set; coverage_rate / "
                    f"criticalCount are computed without them. Investigate the "
                    f"C-quality service ERROR logs for stack traces. "
                    f"First failures: {'; '.join(detail_lines)}{tail}"
                )
            # Merge in input order for stable reporting.
            for slot in indexed:
                if slot is not None:
                    req_results, req_judgments, req_shortlisted = slot
                    all_results.extend(req_results)
                    all_judgments.extend(req_judgments)
                    _total_shortlisted += req_shortlisted

        # Telemetry: count rows where the LLM was skipped via the selector
        # (NO_EVIDENCE / below-floor) or applicability gate. Useful for
        # operators tuning evidence_floor / skip_llm_below_floor.
        skipped_llm_rows = 0
        applicability_skipped_rows = 0
        for r in all_results:
            subcode = (r.status_subcode or "")
            if subcode in ("NOT_APPLICABLE", "OUT_OF_SCOPE"):
                applicability_skipped_rows += 1
            elif subcode in ("MISSING_NO_EVIDENCE", "OPTIONAL_NOT_FOUND"):
                # Best-effort: these subcodes fire when the selector skipped
                # the LLM (no shortlist or NO_EVIDENCE / below-floor) AND
                # nothing else upgraded the row.
                skipped_llm_rows += 1

        logger.info(
            "[%s] Pipeline done: requirements=%d, coverage_units=%d, "
            "total_shortlisted=%d, pair_judgments=%d, "
            "rows_skipped_llm=%d, rows_skipped_applicability=%d",
            job_id, len(requirements), len(all_units),
            _total_shortlisted, len(all_judgments),
            skipped_llm_rows, applicability_skipped_rows,
        )

        # Surface savings as a non-fatal info-level warning so the
        # orchestrator / UI can show it. Helps reviewers understand why a
        # package has many MISSING_NO_EVIDENCE rows — those rows didn't
        # cost LLM calls.
        if skipped_llm_rows or applicability_skipped_rows:
            warnings.append(
                f"LLM_CALL_BUDGET: skipped {skipped_llm_rows} pair(s) via "
                f"retrieval gates (NO_EVIDENCE / below evidence_floor) and "
                f"{applicability_skipped_rows} pair(s) via applicability "
                f"matrix (NOT_APPLICABLE / OUT_OF_SCOPE). Inspect "
                f"evidence_trace on affected rows for the selection_reason."
            )

        # BUG-09 fix: surface Ollama / LLM unavailability to the user.
        # If the active judge silently fell back to DisabledCoverageJudge for
        # any pair, every affected pair was labelled IRRELEVANT, which
        # propagates as MISSING to the user. Without this warning the report
        # looks like "nothing is covered" with no explanation.
        # When llm.enabled is False, the judge is DisabledCoverageJudge by
        # design — we do NOT emit a warning in that case.
        if config.llm.enabled:
            inner_judge = getattr(judge_service, "_judge", None)
            consume = getattr(inner_judge, "consume_unavailability", None)
            if callable(consume):
                fail_count, last_err = consume()
                if fail_count > 0:
                    warnings.append(
                        f"LLM_UNAVAILABLE: judge backend errored on {fail_count} "
                        f"pair(s); last_error={last_err!r}. "
                        f"Affected pairs were marked NOT_JUDGED → status UNKNOWN "
                        f"(excluded from criticalCount and C-grade denominator). "
                        f"C-quality results are incomplete for these requirements: "
                        f"re-run after restoring the LLM service to obtain a "
                        f"definitive verdict."
                    )
                    logger.warning(
                        "[%s] LLM_UNAVAILABLE: %d fallback events; last_error=%s",
                        job_id, fail_count, last_err,
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

        # BUG-14: dedup defense-in-depth. Each (req_id, target_document_id)
        # must yield exactly one RequirementCoverageResult. Upstream req_id
        # collision-avoidance (RequirementBuilder) already enforces unique
        # req_ids per document, but if a future refactor or a bypass path
        # re-introduces a collision we want the orchestrator/UI to see one
        # row per pair, not two rows with contradictory rationales (the
        # audit-time symptom). Tiebreak: keep the row with the highest
        # status priority; if tied, keep the one with non-empty evidence;
        # finally, prefer the first occurrence for stability.
        from app.application.use_cases.aggregate_coverage import _STATUS_RANK
        seen: Dict[tuple, RequirementCoverageResult] = {}
        n_dedup = 0
        for r in all_results:
            key = (r.req_id, r.target_document_id)
            prev = seen.get(key)
            if prev is None:
                seen[key] = r
                continue
            n_dedup += 1
            prev_rank = _STATUS_RANK.get(prev.status, 0)
            curr_rank = _STATUS_RANK.get(r.status, 0)
            if curr_rank > prev_rank:
                seen[key] = r
            elif curr_rank == prev_rank and not prev.evidence and r.evidence:
                seen[key] = r
            # else: keep prev (stable, and at least as good)
        if n_dedup > 0:
            warnings.append(
                f"DUPLICATE_PAIRS: collapsed {n_dedup} duplicate "
                "(req_id, target_document_id) pair(s) into a single result each. "
                "Upstream req_id collision-avoidance should prevent this; "
                "investigate the RequirementBuilder if it recurs."
            )
            logger.warning("[%s] dedup collapsed %d duplicate result(s)", job_id, n_dedup)
        all_results = list(seen.values())

        source_doc_id = source_artifact.get("document_id", "unknown")
        return self._report_builder.build(
            job_id=job_id,
            package_id=package_id,
            source_document_id=source_doc_id,
            requirement_results=all_results,
            pair_judgments=all_judgments,
            warnings=warnings,
        )

    # ------------------------------------------------------------------

    def _process_one_requirement(
        self,
        req_i: int,
        req: "RequirementUnit",
        target_artifacts: List[dict],
        all_units: List["CoverageUnit"],
        units_by_id: Dict[str, "CoverageUnit"],
        judge_service: PairJudgeService,
        config: CoverageConfig,
        job_id: str,
    ) -> Tuple[int, List["RequirementCoverageResult"], List["PairJudgment"], int]:
        """Process one requirement through retrieval → select → judge → verify → aggregate.

        Called from the serial loop or from a ThreadPoolExecutor worker
        (PR-K post-fix B).  All callees are stateless or thread-safe; see
        the module-level note for the thread-safety reasoning.

        Returns
        -------
        (req_i, results_for_this_req, judgments_for_this_req, total_shortlisted)
        """
        req_results: List[RequirementCoverageResult] = []
        req_judgments: List[PairJudgment] = []
        total_shortlisted = 0

        # ── 1. Retrieve candidates per target document ───────────────────
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

        # ── 2. Per-document: select → judge → verify → aggregate ─────────
        for artifact in target_artifacts:
            doc_id = artifact["document_id"]
            doc_role = artifact.get("doc_role", "unknown")
            shortlist = doc_candidates.get(doc_id, [])
            total_shortlisted += len(shortlist)

            # PR-K: applicability-aware skip — OUT_OF_SCOPE / NOT_APPLICABLE
            # rows never get an LLM call; the aggregator fills the row.
            req_type = req.requirement_type or RequirementType.OTHER
            applicability = applicability_for(req_type, doc_role)
            cov_level = coverage_requirement_level_for(req_type, doc_role)

            if applicability != Applicability.APPLICABLE:
                result = self._aggregator.aggregate(
                    requirement=req,
                    judgments=[],
                    candidates_by_unit_id={c.unit_id: c for c in shortlist},
                    units_by_id=units_by_id,
                    target_document_id=doc_id,
                    target_doc_role=doc_role,
                    selection_result=SelectionResult(
                        selected=[], discarded=list(shortlist),
                        selection_reason=(
                            f"applicability={applicability.value}; "
                            f"requirement_type={req_type.value} not "
                            f"checked in target role '{doc_role}'."
                        ),
                        skip_llm=True,
                        skip_reason=f"applicability={applicability.value}",
                    ),
                    coverage_requirement_level=cov_level,
                    debug_cfg=config.debug,
                    aggregator_cfg=config.aggregator,
                )
                req_results.append(result)
                continue

            if not shortlist:
                logger.debug(
                    "[%s] No candidates for req=%s target=%s (all below threshold or no units)",
                    job_id, req.req_id[:12], doc_id,
                )
                result = self._aggregator.aggregate(
                    requirement=req,
                    judgments=[],
                    candidates_by_unit_id={},
                    units_by_id=units_by_id,
                    target_document_id=doc_id,
                    target_doc_role=doc_role,
                    selection_result=SelectionResult(
                        selection_reason="no candidates above min_retrieval_score",
                        skip_llm=True,
                        skip_reason="empty shortlist after retrieval",
                    ),
                    coverage_requirement_level=cov_level,
                    debug_cfg=config.debug,
                    aggregator_cfg=config.aggregator,
                )
                req_results.append(result)
                continue

            # PR-K: AdaptiveCandidateSelector — pick how many candidates
            # go to the LLM (NO_EVIDENCE skip; STRONG+wide margin → k=1;
            # ambiguous → k=3; critical/numeric → up to selector_max_k).
            selection: SelectionResult = select_candidates(
                req, shortlist, config.retrieval,
            )

            # BUG-9: evidence floor.
            max_score = max((c.retrieval_score for c in shortlist), default=0.0)
            low_conf_floor = max_score < config.retrieval.evidence_floor

            # PR-K post-fix (B): skip judge when below floor and not
            # in debug mode, to avoid wasting LLM calls on retrievals
            # that the aggregator will flag low_confidence anyway.
            skip_due_to_floor = (
                low_conf_floor
                and config.retrieval.skip_llm_below_floor
                and _can_skip_llm_below_floor(req)
                and not config.debug.enabled
            )

            if selection.skip_llm or not selection.selected or skip_due_to_floor:
                # Skip LLM — aggregator still runs to produce the row
                # with proper subcode / level / trace.
                judgments: List[PairJudgment] = []
            else:
                # Judge — only the selected slice
                judgments = judge_service.judge_shortlist(
                    req, selection.selected, units_by_id,
                )

                # Verify
                if config.enable_rule_verification:
                    judgments = [
                        self._verifier.verify(j, req, units_by_id[j.unit_id])
                        for j in judgments
                        if j.unit_id in units_by_id
                    ]

                # Mirror the judge label / confidence / grounding onto
                # the candidate so the trace can render the full story.
                cand_by_unit = {c.unit_id: c for c in selection.selected}
                for j in judgments:
                    c = cand_by_unit.get(j.unit_id)
                    if c is None:
                        continue
                    c.judge_label = j.rule_adjusted_label.value
                    c.judge_confidence = float(j.llm_confidence or 0.0)
                    # PR-K P0: grounding_passed reflects ONLY actual
                    # citation grounding (grounding_failed flag), not
                    # below-floor retrieval.
                    c.grounding_passed = not bool(getattr(j, "grounding_failed", False))

                # BUG-9: stamp low_confidence on below-floor judgments.
                # `grounding_failed` deliberately NOT set — the LLM's
                # citation may be perfectly grounded in the (low-retrieval)
                # evidence text.
                if low_conf_floor:
                    for j in judgments:
                        j.low_confidence = True

                req_judgments.extend(judgments)

            # Aggregate
            candidates_by_unit_id = {c.unit_id: c for c in shortlist}
            result = self._aggregator.aggregate(
                requirement=req,
                judgments=judgments,
                candidates_by_unit_id=candidates_by_unit_id,
                units_by_id=units_by_id,
                target_document_id=doc_id,
                target_doc_role=doc_role,
                selection_result=selection,
                coverage_requirement_level=cov_level,
                debug_cfg=config.debug,
                aggregator_cfg=config.aggregator,
            )
            req_results.append(result)

        return req_i, req_results, req_judgments, total_shortlisted
