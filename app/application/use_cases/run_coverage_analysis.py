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
        self._config = config or CoverageConfig()
        self._req_builder = RequirementBuilder(self._config)
        self._unit_builder = CoverageUnitBuilder()
        self._embedding_backend = _build_embedding_backend(self._config)
        self._reranker = _build_reranker(self._config)
        self._retriever = CandidateRetriever(
            self._config.retrieval, self._embedding_backend, self._reranker,
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

                # BUG-9: evidence floor. If the strongest retrieval score in
                # the shortlist is below the configured floor, we don't trust
                # the LLM verdict regardless of label — the audit-time
                # CONFLICT/COVERED rows came from retrieval below 0.45 with
                # judge confidence 0.8, which is exactly this failure mode.
                # We still call the judge so its rationale lands in evidence
                # (useful for debugging), but we flag every produced judgment
                # as low_confidence and force the aggregator-side flag.
                max_score = max((c.retrieval_score for c in shortlist), default=0.0)
                low_conf_floor = max_score < config.retrieval.evidence_floor

                # Judge
                judgments = judge_service.judge_shortlist(req, shortlist, units_by_id)

                # Verify
                if config.enable_rule_verification:
                    judgments = [
                        self._verifier.verify(j, req, units_by_id[j.unit_id])
                        for j in judgments
                        if j.unit_id in units_by_id
                    ]

                # BUG-9: stamp low_confidence on every judgment from a
                # below-floor shortlist. Aggregator OR-merges the flag into
                # the result.
                if low_conf_floor:
                    for j in judgments:
                        j.low_confidence = True

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
                        f"LLM_UNAVAILABLE: judge backend fell back to disabled mode for "
                        f"{fail_count} pair(s); last_error={last_err!r}. "
                        f"Affected pairs were judged IRRELEVANT (→ MISSING). "
                        f"Coverage results may be incomplete; verify the LLM service."
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
