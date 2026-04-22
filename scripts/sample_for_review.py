"""
Sample N requirements per package for manual quality review.

Runs the coverage pipeline in "model" mode on each package, then for each
one picks a stratified sample across statuses (COVERED / PARTIAL / MISSING
/ CONFLICT). Writes a UTF-8 text file where every sample block contains:

    package      : pkg_XXXX
    status       : COVERED
    req_text     : <the extracted requirement sentence>
    classifier_p : 0.93                (confidence of the classifier)
    evidence     :
        [score=0.xx label=...] <PMI unit text>

You then read the file and answer two questions per sample:
    1. Is this actually a requirement?
    2. Is the status correct given the evidence?

Tally the answers and we know whether the model's output is trustworthy
without deploying to a full hand-labelling run.

Usage:
    python scripts/sample_for_review.py \
        --packages pkg_0002 pkg_0005 pkg_0008 pkg_0010 \
        --per-package 5 \
        --out .probe/review_samples.txt
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.application.use_cases.build_coverage_units import CoverageUnitBuilder
from app.application.use_cases.build_requirements import RequirementBuilder
from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
from app.core.config import CoverageConfig


def _chunks_to_prepared(pkg: dict) -> dict:
    documents = []
    for doc in pkg.get("documents", []):
        doc_type = (doc.get("doc_type") or "").upper()
        role = {"TZ": "tz", "PMI": "pmi", "PZ": "pz"}.get(doc_type)
        if not role:
            continue
        fragments = [
            {
                "fragment_id": c.get("chunk_id") or f"frag-{i}",
                "text": c.get("text") or "",
                "kind": "paragraph",
                "section_id": None,
            }
            for i, c in enumerate(doc.get("chunks", []))
            if (c.get("text") or "").strip()
        ]
        artifact = {
            "document_id": doc.get("doc_id") or doc_type,
            "package_id": pkg.get("package_id"),
            "doc_role": role,
            "sections": [],
            "fragments": fragments,
        }
        documents.append({
            "document_id": artifact["document_id"],
            "doc_role": role,
            "prepared_artifact": artifact,
        })
    return {
        "package_id": pkg.get("package_id"),
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi", "pz"],
        "documents": documents,
        "options": {},
    }


def _stratified_sample(items: List[dict], per_bucket: int, key: str, rng: random.Random) -> List[dict]:
    """Round-robin sample up to `per_bucket` per distinct value of item[key]."""
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_bucket[it[key]].append(it)
    picked: List[dict] = []
    for bucket, group in by_bucket.items():
        rng.shuffle(group)
        picked.extend(group[:per_bucket])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", nargs="+",
                    default=["pkg_0002", "pkg_0005", "pkg_0008", "pkg_0010"])
    ap.add_argument("--per-package", type=int, default=5,
                    help="Samples to draw per PACKAGE (rounded up across statuses)")
    ap.add_argument("--per-status", type=int, default=2,
                    help="Samples to draw per STATUS bucket per package")
    ap.add_argument("--packages-dir", default="./data/packages")
    ap.add_argument("--out", default=".probe/review_samples.txt")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_lines: List[str] = []

    config = CoverageConfig()
    config.requirement_extraction = "model"
    config.requirement_model.threshold = args.threshold
    config.embedding.backend = "bow"

    # One shared pipeline — reuses loaded classifier across packages
    pipeline = CoverageAnalysisPipeline(config)
    req_builder = RequirementBuilder(config)
    unit_builder = CoverageUnitBuilder()

    total_samples = 0
    for pkg_name in args.packages:
        pkg_path = Path(args.packages_dir) / f"{pkg_name}.json"
        if not pkg_path.exists():
            print(f"[skip] {pkg_path} not found", file=sys.stderr)
            continue
        pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        request = _chunks_to_prepared(pkg)

        # Build req/unit look-ups once so we can join text onto the report
        # result which only carries IDs.
        source_artifact = next(
            (d["prepared_artifact"] for d in request["documents"] if d["doc_role"] == "tz"),
            None,
        )
        target_artifacts = [
            d["prepared_artifact"] for d in request["documents"] if d["doc_role"] in ("pmi", "pz")
        ]
        if not source_artifact:
            continue

        requirements = req_builder.build(source_artifact)
        units = []
        for art in target_artifacts:
            units.extend(unit_builder.build(art))
        reqs_by_id = {r.req_id: r for r in requirements}
        units_by_id = {u.unit_id: u for u in units}

        # Run the full pipeline and get a report
        report = pipeline.run(request)
        report_d = report.model_dump() if hasattr(report, "model_dump") else report
        results = report_d.get("requirement_results") or []
        pair_judgments = report_d.get("pair_judgments") or []
        judgments_by_req: Dict[str, list] = defaultdict(list)
        for j in pair_judgments:
            judgments_by_req[j["req_id"]].append(j)

        flat = [
            {
                "req_id": r["req_id"],
                "status": str(r["status"]).rsplit(".", 1)[-1].rstrip("'>"),
                "result": r,
            }
            for r in results
        ]
        sample = _stratified_sample(flat, args.per_status, "status", rng)

        out_lines.append(f"{'='*78}")
        out_lines.append(f"{pkg_name}  — total requirements: {len(results)}")
        out_lines.append(f"{'='*78}")

        for i, s in enumerate(sample, 1):
            req = reqs_by_id.get(s["req_id"])
            req_text = (req.text if req else "<text unavailable>").strip()
            score = (req.metadata.get("classifier_score") if req else None) or "?"
            out_lines.append(f"\n-- sample {i} --")
            out_lines.append(f"  package      : {pkg_name}")
            out_lines.append(f"  status       : {s['status']}")
            out_lines.append(f"  classifier_p : {score}")
            out_lines.append(f"  req_id       : {s['req_id']}")
            out_lines.append(f"  req_text     : {req_text}")

            # Top 3 evidence (pair judgments) by score — pulled from the
            # result's embedded evidence list, which already carries the
            # unit text alongside the judgment.
            evidence = s["result"].get("evidence") or []
            evidence = sorted(
                evidence,
                key=lambda e: (e.get("judgment") or {}).get("llm_confidence") or 0,
                reverse=True,
            )
            out_lines.append(f"  evidence ({len(evidence)} items):")
            for e in evidence[:3]:
                unit_text = (e.get("text") or "?")[:220]
                jud = e.get("judgment") or {}
                label = str(jud.get("rule_adjusted_label") or jud.get("llm_label") or "?").rsplit(".", 1)[-1].rstrip("'>")
                conf = jud.get("llm_confidence") or 0
                retr = e.get("retrieval_score") or 0
                out_lines.append(
                    f"      [label={label} conf={conf:.2f} retr={retr:.2f}] {unit_text}"
                )
                expl = (jud.get("explanation") or "").strip()
                if expl:
                    out_lines.append(f"         why: {expl[:180]}")
            out_lines.append(f"\n  Your review:")
            out_lines.append(f"    1) Is this really a requirement?  Y / N  →")
            out_lines.append(f"    2) Is the status correct?         Y / N  →")
            total_samples += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {total_samples} samples for review → {out_path}")


if __name__ == "__main__":
    main()
