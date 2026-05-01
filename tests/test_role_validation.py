"""
B3 regression tests: doc_role accepts only the strict enum at the API layer
(DocumentInput / PreparedArtifact / CoverageAnalysisRequest), and the
internal use case treats unrecognised roles as "unknown" with a warning.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.c_quality_schemas import (
    ALLOWED_DOC_ROLES,
    CoverageAnalysisRequest,
    DocumentInput,
    PreparedArtifact,
)
from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline


def _frag(fid: str, text: str, kind: str = "paragraph") -> dict:
    return {"fragment_id": fid, "text": text, "kind": kind}


# ── DTO-level validation (FastAPI 422 on bad role) ─────────────────────


@pytest.mark.parametrize("role", ["tz", "pmi", "pz", "unknown"])
def test_prepared_artifact_accepts_allowed_roles(role: str):
    art = PreparedArtifact(document_id="d1", doc_role=role)
    assert art.doc_role == role


@pytest.mark.parametrize("role", ["TZ", " pmi ", "Pz"])
def test_prepared_artifact_normalises_case_and_whitespace(role: str):
    """Existing callers that send 'TZ' or trim-able strings must keep working."""
    art = PreparedArtifact(document_id="d1", doc_role=role)
    assert art.doc_role in ALLOWED_DOC_ROLES


@pytest.mark.parametrize("role", ["foobar", "", "TZZ", "spec", "manual"])
def test_prepared_artifact_rejects_unknown_role(role: str):
    with pytest.raises(ValidationError):
        PreparedArtifact(document_id="d1", doc_role=role)


def test_document_input_rejects_unknown_role():
    with pytest.raises(ValidationError):
        DocumentInput(
            document_id="d1",
            doc_role="invalid",
            prepared_artifact={"document_id": "d1", "doc_role": "tz"},
        )


def test_coverage_request_rejects_unknown_source_role():
    with pytest.raises(ValidationError):
        CoverageAnalysisRequest(
            package_id="p1",
            source_doc_role="not-a-role",
            documents=[
                DocumentInput(
                    document_id="d1",
                    doc_role="tz",
                    prepared_artifact={"document_id": "d1", "doc_role": "tz"},
                )
            ],
        )


def test_coverage_request_rejects_unknown_target_role():
    with pytest.raises(ValidationError):
        CoverageAnalysisRequest(
            package_id="p1",
            source_doc_role="tz",
            target_doc_roles=["pmi", "garbage"],
            documents=[
                DocumentInput(
                    document_id="d1",
                    doc_role="tz",
                    prepared_artifact={"document_id": "d1", "doc_role": "tz"},
                )
            ],
        )


def test_coverage_request_normalises_uppercase_roles():
    req = CoverageAnalysisRequest(
        package_id="p1",
        source_doc_role="TZ",
        target_doc_roles=["PMI", "Pz"],
        documents=[
            DocumentInput(
                document_id="d1",
                doc_role="TZ",
                prepared_artifact={"document_id": "d1", "doc_role": "TZ"},
            )
        ],
    )
    assert req.source_doc_role == "tz"
    assert req.target_doc_roles == ["pmi", "pz"]
    assert req.documents[0].doc_role == "tz"


# ── Internal defensive path: unknown roles in raw dict ─────────────────


def test_use_case_treats_unknown_role_as_unknown_with_warning():
    """If a script bypasses the API and feeds an unknown-role doc directly
    into the pipeline, the document must be coerced to 'unknown' (excluded
    from default target_roles) and a warning surfaced.
    """
    package = {
        "package_id": "pkg-bad-role",
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi"],
        "documents": [
            {
                "document_id": "doc-tz",
                "doc_role": "tz",
                "prepared_artifact": {
                    "document_id": "doc-tz",
                    "doc_role": "tz",
                    "fragments": [_frag("f1", "Система должна хранить журнал.")],
                },
            },
            {
                "document_id": "doc-x",
                "doc_role": "manual",   # not in DocRole literal
                "prepared_artifact": {
                    "document_id": "doc-x",
                    "doc_role": "manual",
                    "fragments": [_frag("f2", "free text")],
                },
            },
        ],
        "options": {
            "top_k": 3,
            "enable_llm_judge": False,
            "enable_rule_verification": False,
            "min_retrieval_score": 0.0,
        },
    }

    pipeline = CoverageAnalysisPipeline()
    result = pipeline.run(package)

    invalid_warnings = [w for w in result.warnings if "INVALID_DOC_ROLE" in w]
    assert invalid_warnings, (
        f"expected INVALID_DOC_ROLE warning for 'manual' role, got: {result.warnings!r}"
    )
    # The doc-x must NOT appear in target document_ids — it was coerced to
    # 'unknown', which is not in default target_roles.
    target_ids = result.target_document_ids or []
    assert "doc-x" not in target_ids, (
        f"unknown-role doc must be excluded from targets, got: {target_ids}"
    )


def test_use_case_does_not_warn_on_valid_roles():
    package = {
        "package_id": "pkg-clean",
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi"],
        "documents": [
            {
                "document_id": "doc-tz",
                "doc_role": "tz",
                "prepared_artifact": {
                    "document_id": "doc-tz",
                    "doc_role": "tz",
                    "fragments": [_frag("f1", "Система должна хранить журнал.")],
                },
            },
            {
                "document_id": "doc-pmi",
                "doc_role": "pmi",
                "prepared_artifact": {
                    "document_id": "doc-pmi",
                    "doc_role": "pmi",
                    "fragments": [_frag("f2", "Проверить хранение журнала.")],
                },
            },
        ],
        "options": {
            "top_k": 3,
            "enable_llm_judge": False,
            "enable_rule_verification": False,
            "min_retrieval_score": 0.0,
        },
    }
    pipeline = CoverageAnalysisPipeline()
    result = pipeline.run(package)
    invalid = [w for w in result.warnings if "INVALID_DOC_ROLE" in w]
    assert not invalid, f"unexpected warning on valid roles: {invalid}"
