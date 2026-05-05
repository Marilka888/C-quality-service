"""
Tests for summary tally correctness in CoverageReportBuilder / _tally.

Bug: NOT_APPLICABLE and OUT_OF_SCOPE rows have status=MISSING and were
counted in summary.missing + summary.total_requirements, dragging down
coverage_rate and inflating the missing count.

Fix: rows with applicability != APPLICABLE are excluded from
total_requirements and all status buckets; tracked in not_applicable.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.build_coverage_report import (
    CoverageReportBuilder,
    _tally,
)
from app.domain.c_quality_enums import (
    Applicability,
    CoverageStatus,
    RequirementType,
)
from app.domain.c_quality_models import RequirementCoverageResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def _result(
    status: CoverageStatus,
    applicability: Applicability = Applicability.APPLICABLE,
    req_type: RequirementType = RequirementType.FUNCTIONAL,
    target_doc_role: str = "pmi",
    req_id: str = "r1",
    target_document_id: str = "doc-pmi",
) -> RequirementCoverageResult:
    return RequirementCoverageResult(
        req_id=req_id,
        source_document_id="doc-tz",
        target_document_id=target_document_id,
        target_doc_role=target_doc_role,
        status=status,
        applicability=applicability,
        requirement_type=req_type,
    )


# ── Unit tests for _tally ─────────────────────────────────────────────────────

class TestTallyApplicabilityExclusion:

    def test_not_applicable_excluded_from_missing_and_total(self):
        """A NOT_APPLICABLE row must not appear in missing or total_requirements."""
        results = [
            _result(CoverageStatus.MISSING, Applicability.NOT_APPLICABLE, req_id="r1"),
        ]
        s = _tally(results)
        assert s.total_requirements == 0, (
            f"NOT_APPLICABLE must not count toward total; got {s.total_requirements}"
        )
        assert s.missing == 0, (
            f"NOT_APPLICABLE must not count toward missing; got {s.missing}"
        )
        assert s.not_applicable == 1

    def test_out_of_scope_excluded_from_missing_and_total(self):
        """An OUT_OF_SCOPE row must not appear in missing or total_requirements."""
        results = [
            _result(CoverageStatus.MISSING, Applicability.OUT_OF_SCOPE, req_id="r1"),
        ]
        s = _tally(results)
        assert s.total_requirements == 0
        assert s.missing == 0
        assert s.not_applicable == 1

    def test_applicable_missing_still_counted(self):
        """APPLICABLE MISSING rows must still appear in missing."""
        results = [
            _result(CoverageStatus.MISSING, Applicability.APPLICABLE, req_id="r1"),
        ]
        s = _tally(results)
        assert s.total_requirements == 1
        assert s.missing == 1
        assert s.not_applicable == 0

    def test_mixed_rows_correct_counts(self):
        """Mixed APPLICABLE + NOT_APPLICABLE rows: only APPLICABLE ones
        contribute to total / status buckets."""
        results = [
            _result(CoverageStatus.COVERED,  Applicability.APPLICABLE,     req_id="r1"),
            _result(CoverageStatus.COVERED,  Applicability.APPLICABLE,     req_id="r2"),
            _result(CoverageStatus.PARTIAL,  Applicability.APPLICABLE,     req_id="r3"),
            _result(CoverageStatus.MISSING,  Applicability.APPLICABLE,     req_id="r4"),
            _result(CoverageStatus.MISSING,  Applicability.NOT_APPLICABLE, req_id="r5"),
            _result(CoverageStatus.MISSING,  Applicability.OUT_OF_SCOPE,   req_id="r6"),
        ]
        s = _tally(results)
        assert s.total_requirements == 4
        assert s.covered == 2
        assert s.partial == 1
        assert s.missing == 1
        assert s.conflict == 0
        assert s.not_applicable == 2

    def test_coverage_rate_excludes_not_applicable(self):
        """coverage_rate must use only APPLICABLE rows as denominator."""
        results = [
            _result(CoverageStatus.COVERED,  Applicability.APPLICABLE,     req_id="r1"),
            _result(CoverageStatus.COVERED,  Applicability.APPLICABLE,     req_id="r2"),
            _result(CoverageStatus.MISSING,  Applicability.NOT_APPLICABLE, req_id="r3"),
            _result(CoverageStatus.MISSING,  Applicability.NOT_APPLICABLE, req_id="r4"),
        ]
        s = _tally(results)
        assert s.total_requirements == 2
        assert s.covered == 2
        assert abs(s.coverage_rate - 1.0) < 1e-9, (
            f"coverage_rate should be 1.0 when all APPLICABLE rows are COVERED; "
            f"got {s.coverage_rate}"
        )

    def test_all_not_applicable_zero_rate(self):
        """When every row is NOT_APPLICABLE, total is 0 and coverage_rate is 0."""
        results = [
            _result(CoverageStatus.MISSING, Applicability.NOT_APPLICABLE, req_id="r1"),
            _result(CoverageStatus.MISSING, Applicability.OUT_OF_SCOPE,   req_id="r2"),
        ]
        s = _tally(results)
        assert s.total_requirements == 0
        assert s.coverage_rate == 0.0
        assert s.not_applicable == 2


# ── Integration: CoverageReportBuilder per-doc tally ─────────────────────────

class TestReportBuilderDocTally:

    def setup_method(self):
        self.builder = CoverageReportBuilder()

    def _build(self, results):
        return self.builder.build(
            job_id="j1",
            package_id="pkg1",
            source_document_id="doc-tz",
            requirement_results=results,
        )

    def test_per_doc_not_applicable_excluded(self):
        """DocumentCoverageReport.missing must not include NOT_APPLICABLE rows."""
        results = [
            _result(CoverageStatus.COVERED,  Applicability.APPLICABLE,     req_id="r1"),
            _result(CoverageStatus.MISSING,  Applicability.NOT_APPLICABLE, req_id="r2"),
        ]
        report = self._build(results)
        doc = report.document_reports[0]
        assert doc.total_requirements == 1
        assert doc.covered == 1
        assert doc.missing == 0
        assert doc.not_applicable == 1

    def test_per_doc_multiple_docs(self):
        """NOT_APPLICABLE exclusion works per document independently."""
        results = [
            _result(CoverageStatus.COVERED,  Applicability.APPLICABLE,     req_id="r1",
                    target_document_id="doc-pmi", target_doc_role="pmi"),
            _result(CoverageStatus.MISSING,  Applicability.NOT_APPLICABLE, req_id="r1",
                    target_document_id="doc-pz",  target_doc_role="pz"),
        ]
        report = self._build(results)
        by_doc = {d.target_document_id: d for d in report.document_reports}

        pmi = by_doc["doc-pmi"]
        assert pmi.total_requirements == 1
        assert pmi.covered == 1
        assert pmi.missing == 0
        assert pmi.not_applicable == 0

        pz = by_doc["doc-pz"]
        assert pz.total_requirements == 0
        assert pz.missing == 0
        assert pz.not_applicable == 1

    def test_summary_tally_matches_per_doc_fix(self):
        """result.summary must also exclude NOT_APPLICABLE from total/missing."""
        results = [
            _result(CoverageStatus.COVERED,  Applicability.APPLICABLE,     req_id="r1"),
            _result(CoverageStatus.PARTIAL,  Applicability.APPLICABLE,     req_id="r2"),
            _result(CoverageStatus.MISSING,  Applicability.NOT_APPLICABLE, req_id="r3"),
            _result(CoverageStatus.MISSING,  Applicability.OUT_OF_SCOPE,   req_id="r4"),
        ]
        result = self._build(results)
        s = result.summary
        assert s.total_requirements == 2
        assert s.covered == 1
        assert s.partial == 1
        assert s.missing == 0
        assert s.not_applicable == 2
