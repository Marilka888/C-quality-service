from __future__ import annotations

from app.core.config import ServiceConfig
from app.domain.entities import RequirementFinding, TraceLink, TraceabilityReport
from app.domain.enums import LinkStatus
from app.reporting.serializers import serialize_report
from app.scoring.aggregation import build_aggregated_metrics, build_summary_metrics


def build_traceability_report(
    requirement_findings: list[RequirementFinding],
    orphan_findings: list,
    config: ServiceConfig,
) -> TraceabilityReport:
    summary = build_summary_metrics(requirement_findings, orphan_findings)
    aggregated = build_aggregated_metrics(summary, orphan_findings, config.scoring)
    accepted_links: list[TraceLink] = []
    conflicting_links: list[TraceLink] = []
    uncovered_requirements: list[str] = []

    for finding in requirement_findings:
        if finding.selected_best_match is None:
            uncovered_requirements.append(finding.requirement_id)
            continue
        if finding.final_status == LinkStatus.CONFLICT:
            conflicting_links.append(finding.selected_best_match)
        elif finding.final_status in {LinkStatus.ADEQUATE, LinkStatus.PARTIAL, LinkStatus.INADEQUATE}:
            accepted_links.append(finding.selected_best_match)
        elif finding.final_status == LinkStatus.MISSING:
            uncovered_requirements.append(finding.requirement_id)

    return TraceabilityReport(
        summary=summary,
        aggregated_metrics=aggregated,
        accepted_links=accepted_links,
        conflicting_links=conflicting_links,
        uncovered_requirements=uncovered_requirements,
        orphan_test_cases=orphan_findings,
        detailed_findings=requirement_findings,
    )


def build_traceability_report_payload(
    requirement_findings: list[RequirementFinding],
    orphan_findings: list,
    config: ServiceConfig,
) -> dict:
    return serialize_report(build_traceability_report(requirement_findings, orphan_findings, config))
