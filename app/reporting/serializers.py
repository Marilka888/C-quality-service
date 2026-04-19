from __future__ import annotations

from app.domain.entities import TraceabilityReport


def serialize_report(report: TraceabilityReport) -> dict:
    return report.model_dump(mode="json")
