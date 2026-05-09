"""
Polyakov-regression: silent worker exclusions.

Real-data observation on the Polyakov ТЗ run: the document has ~30
product requirements but only 3 reach the C-quality result set. The
parallel requirement-processing loop catches per-worker exceptions,
sets the slot to ([], [], 0) and continues — the user sees no signal
that 28 requirements failed silently, just an empty ТЗ↔ПЗ comparison
table.

Contract pinned: worker exceptions must populate a WORKER_EXCLUSIONS
warning in the final report so the orchestrator / UI can surface the
failure count and the user knows to check the C-quality service logs.
"""
from __future__ import annotations

from unittest.mock import patch

from app.application.use_cases.run_coverage_analysis import (
    CoverageAnalysisPipeline,
)


def _build_request(n_reqs: int) -> dict:
    """Construct a minimal package-shaped request with `n_reqs` TZ
    requirement candidates and one PMI target. The PMI target is empty
    so retrieval / LLM are short-circuited — the parallel branch is
    still entered for n_reqs > 1."""
    candidates = [
        {
            "fragment_id": f"sec::sent{i}",
            "section_id": "sec",
            "text": f"Система должна обеспечивать функцию номер {i}.",
            "metadata": {"sectionCategory": "requirements"},
        }
        for i in range(n_reqs)
    ]
    return {
        "job_id": f"test-job-{n_reqs}",
        "package_id": "test-pkg",
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi"],
        "documents": [
            {
                "document_id": "doc-tz",
                "doc_role": "tz",
                "prepared_artifact": {
                    "document_id": "doc-tz",
                    "doc_role": "tz",
                    "requirement_candidates": candidates,
                    "fragments": [],
                    "sections": [
                        {"section_id": "sec", "title": "Требования к функциям"}
                    ],
                },
            },
            {
                "document_id": "doc-pmi",
                "doc_role": "pmi",
                "prepared_artifact": {
                    "document_id": "doc-pmi",
                    "doc_role": "pmi",
                    "fragments": [
                        {"fragment_id": "p1", "text": "Тестовая методика.",
                         "kind": "paragraph"}
                    ],
                },
            },
        ],
        "options": {
            "enable_llm_judge": False,
            "enable_rule_verification": False,
        },
    }


def test_worker_exceptions_surface_as_warning_polyakov() -> None:
    pipeline = CoverageAnalysisPipeline()

    # Patch the worker entry-point to throw for every requirement.
    def boom(self, req_i, *args, **kwargs):
        raise RuntimeError(f"synthetic failure for req {req_i}")

    with patch.dict("os.environ", {"CQUALITY_REQ_CONCURRENCY": "4"}, clear=False), \
         patch.object(CoverageAnalysisPipeline, "_process_one_requirement", boom):
        result = pipeline.run(_build_request(8))

    warnings_text = "\n".join(result.warnings or [])
    assert "WORKER_EXCLUSIONS" in warnings_text, (
        f"WORKER_EXCLUSIONS warning missing — silent worker exclusion bug "
        f"would let 28-of-31 requirement losses go unreported. Warnings: "
        f"{result.warnings!r}"
    )
    assert "8 requirement" in warnings_text


def test_worker_exceptions_cap_detail_lines_at_5_polyakov() -> None:
    pipeline = CoverageAnalysisPipeline()

    def boom(self, req_i, *args, **kwargs):
        raise ValueError("synthetic")

    with patch.dict("os.environ", {"CQUALITY_REQ_CONCURRENCY": "4"}, clear=False), \
         patch.object(CoverageAnalysisPipeline, "_process_one_requirement", boom):
        result = pipeline.run(_build_request(20))

    warnings_text = "\n".join(result.warnings or [])
    assert "WORKER_EXCLUSIONS" in warnings_text
    assert "20 requirement" in warnings_text
    assert "(+ 15 more" in warnings_text


def test_no_warning_when_workers_succeed_polyakov() -> None:
    pipeline = CoverageAnalysisPipeline()

    def success(self, req_i, *args, **kwargs):
        return (req_i, [], [], 0)

    with patch.dict("os.environ", {"CQUALITY_REQ_CONCURRENCY": "4"}, clear=False), \
         patch.object(CoverageAnalysisPipeline, "_process_one_requirement", success):
        result = pipeline.run(_build_request(5))

    warnings_text = "\n".join(result.warnings or [])
    assert "WORKER_EXCLUSIONS" not in warnings_text
