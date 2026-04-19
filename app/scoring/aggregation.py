from __future__ import annotations

from typing import List

from app.core.config import ScoringConfig
from app.domain.entities import AggregatedMetrics, OrphanTestFinding, RequirementFinding, SummaryMetrics
from app.domain.enums import LinkStatus


def build_summary_metrics(
    requirement_findings: List[RequirementFinding],
    orphan_findings: List[OrphanTestFinding],
) -> SummaryMetrics:
    statuses = [finding.final_status for finding in requirement_findings]
    return SummaryMetrics(
        total_requirements=len(requirement_findings),
        adequate_count=sum(status == LinkStatus.ADEQUATE for status in statuses),
        partial_count=sum(status == LinkStatus.PARTIAL for status in statuses),
        inadequate_count=sum(status == LinkStatus.INADEQUATE for status in statuses),
        missing_count=sum(status == LinkStatus.MISSING for status in statuses),
        conflict_count=sum(status == LinkStatus.CONFLICT for status in statuses),
        orphan_test_count=len(orphan_findings),
    )


def build_aggregated_metrics(
    summary: SummaryMetrics,
    orphan_findings: List[OrphanTestFinding],
    config: ScoringConfig,
) -> AggregatedMetrics:
    total = max(summary.total_requirements, 1)
    orphan_base = max(summary.total_requirements + len(orphan_findings), 1)
    base_score = (
        summary.adequate_count * config.status_weights["ADEQUATE"]
        + summary.partial_count * config.status_weights["PARTIAL"]
        + summary.inadequate_count * config.status_weights["INADEQUATE"]
        + summary.missing_count * config.status_weights["MISSING"]
        + summary.conflict_count * config.status_weights["CONFLICT"]
    ) / total
    orphan_penalty = config.orphan_penalty_weight * (len(orphan_findings) / orphan_base)
    score_c = max(0.0, min(1.0, base_score - orphan_penalty))

    return AggregatedMetrics(
        adequate_coverage_rate=summary.adequate_count / total,
        partial_coverage_rate=summary.partial_count / total,
        missing_rate=summary.missing_count / total,
        conflict_rate=summary.conflict_count / total,
        orphan_test_rate=len(orphan_findings) / orphan_base,
        score_c=score_c,
    )
