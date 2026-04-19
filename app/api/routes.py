from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import EvaluateTraceabilityRequest, EvaluateTraceabilityResponse
from app.core.config import ServiceConfig
from app.domain.entities import Requirement, TestCase
from app.service.traceability import TraceabilityService

router = APIRouter()


@router.post("/traceability/evaluate", response_model=EvaluateTraceabilityResponse)
def evaluate_traceability(req: EvaluateTraceabilityRequest) -> EvaluateTraceabilityResponse:
    config = ServiceConfig.from_request_overrides(
        top_k=req.top_k,
        min_retrieval_score=req.min_retrieval_score,
        use_llm=req.use_llm,
        use_embeddings=req.use_embeddings,
    )
    service = TraceabilityService(config=config)
    report = service.evaluate(
        requirements=[Requirement(**item.model_dump()) for item in req.requirements],
        test_cases=[TestCase(**item.model_dump()) for item in req.test_cases],
    )
    return EvaluateTraceabilityResponse(report=report)


@router.post("/judge", response_model=EvaluateTraceabilityResponse)
def judge(req: EvaluateTraceabilityRequest) -> EvaluateTraceabilityResponse:
    return evaluate_traceability(req)


# ---------------------------------------------------------------------------
# C-quality: package-level coverage analysis
# ---------------------------------------------------------------------------

from app.api.c_quality_schemas import CoverageAnalysisRequest, CoverageAnalysisResponse  # noqa: E402
from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline  # noqa: E402

_coverage_pipeline = CoverageAnalysisPipeline()


@router.post("/coverage/analyze", response_model=CoverageAnalysisResponse)
def analyze_coverage(req: CoverageAnalysisRequest) -> CoverageAnalysisResponse:
    payload = req.model_dump()
    result = _coverage_pipeline.run(payload)
    return CoverageAnalysisResponse(result=result.model_dump(mode="json"))
