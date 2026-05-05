"""
Post-mortem diagnostics for a docAI / C-quality-service package report.

Reads a final report JSON and prints three signals that explain WHY a
package has many MISSING / low-confidence rows:

  1. retrieval-quality distribution per evidence_strength bin --
     answers "is the embedding backend strong enough?"
  2. low_confidence rate over APPLICABLE rows --
     answers "is retrieval below evidence_floor for too many pairs?"
  3. status_subcode breakdown --
     answers "where does the pipeline lose verdicts?"

Usage:
  py -m scripts.diag_package <report.json>
  cat report.json | py -m scripts.diag_package -

Designed for quick CLI inspection during package triage. No external
deps beyond the stdlib.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


# -- helpers --------------------------------------------------------------


def _load(path: str) -> Dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_c_requirements(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """The report shape exposes C-quality rows under either key --
    accept both (legacy `c.requirements` and newer `c.byDocument[*].rows`)."""
    c = report.get("c") or {}
    if isinstance(c.get("requirements"), list):
        yield from c["requirements"]
        return
    for doc in c.get("byDocument") or []:
        for row in doc.get("rows") or []:
            yield row


def _bar(label: str, n: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return f"  {label:30s} {n:4d}"
    pct = n / total
    filled = int(round(pct * width))
    # ASCII-safe so cp1251 / cp866 stdouts on Windows don't crash.
    bar = "#" * filled + "." * (width - filled)
    return f"  {label:30s} {n:4d} ({pct:5.1%}) {bar}"


# -- analyses -------------------------------------------------------------


def evidence_strength_dist(rows: List[Dict[str, Any]]) -> Counter:
    """Count rows by their best-evidence strength (STRONG / MEDIUM / WEAK
    / NO_EVIDENCE). Falls back to inferring from `evidence[0].score`."""
    counter: Counter = Counter()
    for r in rows:
        ev = r.get("evidence") or []
        if not ev:
            counter["NO_EVIDENCE"] += 1
            continue
        top_score = float(ev[0].get("score") or 0.0)
        if top_score >= 0.45:
            counter["STRONG"] += 1
        elif top_score >= 0.25:
            counter["MEDIUM"] += 1
        elif top_score >= 0.12:
            counter["WEAK"] += 1
        else:
            counter["NO_EVIDENCE"] += 1
    return counter


def low_confidence_rate(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    applicable = [r for r in rows if r.get("applicability") == "APPLICABLE"]
    low = sum(1 for r in applicable if r.get("lowConfidence") is True)
    return {
        "applicable_total": len(applicable),
        "low_confidence_count": low,
        "low_confidence_pct": (low / len(applicable)) if applicable else 0.0,
    }


def subcode_breakdown(rows: List[Dict[str, Any]]) -> Counter:
    return Counter(r.get("statusSubcode") or "(none)" for r in rows)


def role_split(rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    by_role: Dict[str, Counter] = {}
    for r in rows:
        role = (r.get("targetRole") or "?").lower()
        status = r.get("status") or "?"
        by_role.setdefault(role, Counter())[status] += 1
    return by_role


# -- reporting ------------------------------------------------------------


def render(report: Dict[str, Any]) -> str:
    rows = list(_iter_c_requirements(report))
    if not rows:
        return "No C-quality rows in report.\n"

    out: List[str] = []
    pkg = report.get("package") or {}
    out.append(f"Package: {pkg.get('name', '?')}  ({pkg.get('id', '?')})")
    summary = (report.get("summary") or {})
    out.append(
        f"Grade={summary.get('grade', '?')}  "
        f"criticalCount={summary.get('criticalCount', '?')}  "
        f"status={summary.get('status', '?')}"
    )
    out.append(f"C-quality rows: {len(rows)}")
    out.append("")

    # 1. evidence strength distribution
    out.append("--- 1. Retrieval quality (by best-evidence score) -----------")
    es = evidence_strength_dist(rows)
    for label in ("STRONG", "MEDIUM", "WEAK", "NO_EVIDENCE"):
        out.append(_bar(label, es.get(label, 0), len(rows)))
    no_ev_pct = es.get("NO_EVIDENCE", 0) / len(rows)
    weak_or_below = (es.get("WEAK", 0) + es.get("NO_EVIDENCE", 0)) / len(rows)
    if no_ev_pct > 0.30:
        out.append(
            "  !>30% NO_EVIDENCE -- embedding backend likely too weak "
            "(BoW?) or fragments split too small in prepare-service."
        )
    elif weak_or_below > 0.50:
        out.append(
            "  !>50% WEAK+ -- reranker probably disabled or first-stage "
            "embedding lacks recall. Try `enable_reranker: true`."
        )
    out.append("")

    # 2. low_confidence rate
    out.append("--- 2. Low-confidence rate (APPLICABLE rows only) -----------")
    lc = low_confidence_rate(rows)
    out.append(
        f"  applicable={lc['applicable_total']}  "
        f"low_confidence={lc['low_confidence_count']} "
        f"({lc['low_confidence_pct']:.1%})"
    )
    if lc["low_confidence_pct"] > 0.40:
        out.append(
            "  !>40% low_confidence -- most retrieval is below evidence_floor. "
            "Either floor too strict (lower it) or retrieval genuinely weak."
        )
    out.append("")

    # 3. subcode breakdown
    out.append("--- 3. Status subcode breakdown ------------------------------")
    sb = subcode_breakdown(rows)
    for code, n in sb.most_common():
        out.append(_bar(code, n, len(rows)))
    out.append("")

    # 4. status × role
    out.append("--- 4. Status x target role ---------------------------------")
    rs = role_split(rows)
    for role, counts in sorted(rs.items()):
        total = sum(counts.values())
        line = f"  {role.upper():4s} (n={total}):  "
        line += "  ".join(
            f"{s}={counts.get(s, 0)}"
            for s in ("COVERED", "PARTIAL", "MISSING", "CONFLICT")
        )
        out.append(line)
    out.append("")

    # 5. tunable hints
    out.append("--- 5. Suggestions ------------------------------------------")
    suggestions: List[str] = []
    if no_ev_pct > 0.20:
        suggestions.append(
            "Install numpy/torch and switch embedding_backend to 'transformer' "
            "or 'e5' (current run likely fell back to BoW)."
        )
    if weak_or_below > 0.40:
        suggestions.append(
            "Enable reranker (enable_reranker=true) -- first-stage signal is weak."
        )
    pz_missing = rs.get("pz", Counter()).get("MISSING", 0)
    pz_total = sum(rs.get("pz", Counter()).values())
    if pz_total and pz_missing / pz_total > 0.70:
        suggestions.append(
            "PZ has >70% MISSING -- verify functional/data_io are routed as "
            "OPTIONAL in PZ (commit ce269eb). If still REQUIRED, criticalCount "
            "is inflated."
        )
    if not suggestions:
        suggestions.append("(none -- package looks healthy)")
    for s in suggestions:
        out.append(f"  -{s}")

    return "\n".join(out) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "report",
        help="Path to report.json, or '-' for stdin",
    )
    args = parser.parse_args(argv)
    try:
        report = _load(args.report)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read report: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
