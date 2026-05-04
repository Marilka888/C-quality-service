"""
Run PR-K C-quality on a real DOCX package and produce the full diagnostic
summary the human-review checklist expects.

Usage:
    cd C:/Users/Marilka/PycharmProjects/C-quality-service
    .venv/Scripts/python.exe scripts/run_pr_k_real_package.py PACKAGE_DIR [--package-id NAME] [--model qwen2.5:7b] [--out report.json]

PACKAGE_DIR must contain three .docx files; the script auto-detects each
file's role (TZ / PZ / PMI) by filename keywords (тз/pz/пз/пми/pmi/тех_зад
etc.). If detection is ambiguous, pass --tz / --pz / --pmi explicitly.

Output:
  * stdout — three-section summary (per-package metrics + selector / verifier
    distribution + a stratified 30-row sample with hex IDs you can trace
    back into the JSON dump)
  * --out path (default: scripts/_pr_k_real_<package_id>.json) — full
    pipeline result + 30 review rows pre-classified by suspected root cause

Env vars:
  CQUALITY_LLM_TIMEOUT  — seconds (default 180)
  OLLAMA_URL            — http://localhost:11434/api/generate by default

Why this script vs scripts/calibrate_pr_c.py:
  * calibrate_pr_c was hardcoded to «Cherevuyhho»; this one accepts any
    package directory.
  * Output schema matches the smoke-test summary plus 30-row review rows
    so the human reviewer can spend 1.5–2 hours pre-classifying instead of
    starting from raw evidence_trace dumps.
  * Doesn't touch production config — all options are passed through
    CoverageConfig.from_options() in-memory.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT.parent / "Prepare-service"
sys.path.insert(0, str(ROOT))

from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
from app.core.config import CoverageConfig


# Mirrors docback's c_job.go::candidateTypeToFragmentKind
_FRAGMENT_KIND = {
    "test_step": "test_step",
    "test_method": "test_step",
    "acceptance_criterion": "test_step",
    "test_conclusion": "test_step",
    "requirement_like": "paragraph",
    "documentation_requirement": "paragraph",
    "environment_requirement": "paragraph",
    "input_output_spec": "paragraph",
    "descriptive_context": "paragraph",
    "metadata": "paragraph",
    "noise": "paragraph",
}
_TZ_REQ_TYPES = {
    "requirement_like", "documentation_requirement", "environment_requirement",
}


# ── role auto-detection ─────────────────────────────────────────────────


_ROLE_KEYWORDS = {
    "tz": [
        re.compile(r"(?:^|[_\-\s.])(?:тз|tz|tech_zad|техническ|тех_зад)", re.I),
        re.compile(r"индивидуальн", re.I),
    ],
    "pz": [
        re.compile(r"(?:^|[_\-\s.])(?:пз|pz|пояснит|explain)", re.I),
        re.compile(r"проект_вкр|проект_вкр|вкр_", re.I),
    ],
    "pmi": [
        re.compile(r"(?:^|[_\-\s.])(?:пми|pmi|тест|test)", re.I),
        re.compile(r"мет(?:одик|оды)", re.I),
    ],
}


def _detect_role(filename: str) -> Optional[str]:
    name = filename.lower()
    matched = []
    for role, patterns in _ROLE_KEYWORDS.items():
        if any(p.search(name) for p in patterns):
            matched.append(role)
    if len(matched) == 1:
        return matched[0]
    return None  # ambiguous or none


# ── parse via Prepare-service in subprocess ─────────────────────────────


def _parse_doc_via_subprocess(path: Path) -> Dict[str, Any]:
    helper = ROOT / "scripts" / "_prep_doc_to_json.py"
    if not helper.exists():
        raise RuntimeError(
            f"missing {helper}; expected the prepare-side helper used by "
            f"scripts/calibrate_pr_c.py"
        )
    proc = subprocess.run(
        [sys.executable, str(helper), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"prepare subprocess failed for {path.name}: rc={proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}"
        )
    return json.loads(proc.stdout)


def _to_prepared_artifact(role: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    sections = [
        {
            "section_id": s["id"],
            "title": s["section_name"],
            "level": s["level"],
            "number": s.get("number"),
        }
        for s in parsed.get("sections", [])
    ]
    fragments = []
    requirement_candidates = []
    for c in parsed.get("bq_candidates", []):
        kind = _FRAGMENT_KIND.get(c["candidate_type"], "paragraph")
        meta = {
            "sectionId": c["source_section_id"],
            "fragmentId": c["candidate_id"],
            "candidateType": c["candidate_type"],
            "sectionCategory": c["section_category"],
        }
        fragments.append({
            "fragment_id": c["candidate_id"],
            "text": c["text"],
            "kind": kind,
            "section_id": c["source_section_id"],
            "metadata": meta,
        })
        if c["candidate_type"] in _TZ_REQ_TYPES:
            requirement_candidates.append({
                "req_id": c["candidate_id"],
                "text": c["text"],
                "section_id": c["source_section_id"],
                "fragment_id": c["candidate_id"],
                "metadata": meta,
            })
    return {
        "document_id": parsed["document_id"],
        "doc_role": role,
        "sections": sections,
        "fragments": fragments,
        "requirement_candidates": requirement_candidates,
    }


# ── Summary aggregation ─────────────────────────────────────────────────


@dataclass
class PackageSummary:
    package_id: str
    model: str
    elapsed_sec: float
    requirement_count: int
    pair_count: int                 # rows in result (req × target)
    llm_calls_executed: int         # selected_for_llm sum
    llm_calls_saved_by_skip: int    # rows where selector skip_llm=True
    by_status: Dict[str, int] = field(default_factory=dict)
    by_subcode: Dict[str, int] = field(default_factory=dict)
    selected_k_dist: Dict[int, int] = field(default_factory=dict)
    reranker_used_count: int = 0
    verifier_actions: Dict[str, int] = field(default_factory=dict)
    grounded_judgments: int = 0
    ungrounded_judgments: int = 0
    low_confidence_rows: int = 0
    warnings: List[str] = field(default_factory=list)


def _aggregate_summary(package_id: str, model: str, result, elapsed: float) -> PackageSummary:
    s = PackageSummary(
        package_id=package_id,
        model=model,
        elapsed_sec=round(elapsed, 2),
        requirement_count=len({r.req_id for r in result.requirement_results}),
        pair_count=len(result.requirement_results),
        llm_calls_executed=0,
        llm_calls_saved_by_skip=0,
    )
    for r in result.requirement_results:
        s.by_status[r.status.value] = s.by_status.get(r.status.value, 0) + 1
        sc = r.status_subcode or "(none)"
        s.by_subcode[sc] = s.by_subcode.get(sc, 0) + 1
        if r.low_confidence:
            s.low_confidence_rows += 1
        if not r.evidence_trace:
            continue
        sel = r.evidence_trace.get("selection") or {}
        if sel.get("skip_llm"):
            s.llm_calls_saved_by_skip += 1
        else:
            k = int(sel.get("selected_k") or 0)
            s.selected_k_dist[k] = s.selected_k_dist.get(k, 0) + 1
            s.llm_calls_executed += k
        for cand in r.evidence_trace.get("candidates", []):
            if cand.get("reranker_used"):
                s.reranker_used_count += 1
            for a in cand.get("verifier_actions", []) or []:
                s.verifier_actions[a] = s.verifier_actions.get(a, 0) + 1
            grp = cand.get("grounding_passed")
            if grp is True:
                s.grounded_judgments += 1
            elif grp is False:
                s.ungrounded_judgments += 1
    s.warnings = list(result.warnings or [])
    return s


# ── 30-row stratified sample ────────────────────────────────────────────


def _suggest_root_cause(row) -> str:
    """Pre-classify a likely root cause for the reviewer to confirm/correct.

    The categories here mirror the human-review taxonomy in the user's
    plan. The reviewer decides the final classification.
    """
    sc = row.status_subcode or ""
    has_evidence = bool(row.evidence)
    if sc == "OUT_OF_SCOPE":
        return "applicability_matrix"            # check req_type → role mapping
    if sc == "NOT_APPLICABLE":
        return "applicability_matrix"
    if sc == "OPTIONAL_NOT_FOUND" and not has_evidence:
        return "retrieval_miss_or_truly_missing"
    if sc == "MISSING_NO_EVIDENCE" and not has_evidence:
        return "retrieval_miss_or_truly_missing"
    if sc == "MISSING_LOW_GROUNDING":
        return "llm_hallucinated_citation"       # gate caught it
    if sc == "MISSING_LOW_CONFIDENCE":
        return "llm_unsure_or_threshold_too_strict"
    if sc == "CONFLICT_VERIFIED":
        # Inspect verifier_actions on the winning candidate.
        win_id = row.winning_candidate_id
        win_actions = []
        if row.evidence_trace:
            for c in row.evidence_trace.get("candidates", []):
                if c.get("unit_id") == win_id:
                    win_actions = c.get("verifier_actions") or []
                    break
        if any(a.startswith("conflict_confirmed_numeric") for a in win_actions):
            return "verifier_numeric_conflict"
        if any(a.startswith("conflict_confirmed_negation") for a in win_actions):
            return "verifier_negation_conflict"
        return "verifier_unknown_conflict_kind"
    if sc == "COVERED":
        return "llm_covered_grounded"
    if sc == "PARTIAL":
        return "llm_partial_or_verifier_demote"
    return "unclassified"


def _pick_review_rows(result, n: int = 30) -> List[Dict[str, Any]]:
    """Stratified 30-row sample matching the user's review checklist."""
    by_bucket: Dict[str, List[Any]] = {
        "conflict": [],
        "missing": [],
        "covered_or_partial": [],
        "not_applicable_or_optional": [],
    }
    for r in result.requirement_results:
        sc = r.status_subcode or r.status.value
        if sc == "CONFLICT_VERIFIED" or r.status.value == "CONFLICT":
            by_bucket["conflict"].append(r)
        elif sc in {"MISSING_NO_EVIDENCE", "MISSING_LOW_GROUNDING", "MISSING_LOW_CONFIDENCE"} or r.status.value == "MISSING":
            by_bucket["missing"].append(r)
        elif sc in {"NOT_APPLICABLE", "OUT_OF_SCOPE", "OPTIONAL_NOT_FOUND"}:
            by_bucket["not_applicable_or_optional"].append(r)
        elif r.status.value in {"COVERED", "PARTIAL"}:
            by_bucket["covered_or_partial"].append(r)

    quotas = {
        "conflict": min(10, len(by_bucket["conflict"])),
        "missing": min(10, len(by_bucket["missing"])),
        "covered_or_partial": min(10, len(by_bucket["covered_or_partial"])),
        "not_applicable_or_optional": min(5, len(by_bucket["not_applicable_or_optional"])),
    }
    rng = random.Random(42)
    out: List[Dict[str, Any]] = []
    for bucket, quota in quotas.items():
        sample = rng.sample(by_bucket[bucket], quota) if quota else []
        for r in sample:
            short_id = (r.winning_candidate_id or r.req_id)[:8]
            out.append({
                "bucket": bucket,
                "short_id": short_id,
                "req_text": (r.req_text or "")[:160],
                "req_section_title": r.req_section_title,
                "target_doc_role": r.target_doc_role,
                "requirement_type": r.requirement_type.value,
                "applicability": r.applicability.value,
                "coverage_requirement_level": r.coverage_requirement_level.value,
                "status": r.status.value,
                "status_subcode": r.status_subcode,
                "final_confidence": r.final_confidence,
                "winning_candidate_id": r.winning_candidate_id,
                "winning_evidence_text": (
                    next(
                        (e.text for e in r.evidence
                         if e.unit_id == (r.winning_candidate_id or "")),
                        None,
                    )
                    or (r.evidence[0].text if r.evidence else "")
                )[:240],
                "aggregation_reason": r.aggregation_reason,
                "suggested_root_cause": _suggest_root_cause(r),
                "human_verdict": "TODO_FILL",
                "human_root_cause": "TODO_FILL",
                "human_notes": "",
            })
    return out


# ── Pretty print ────────────────────────────────────────────────────────


def _print_summary(s: PackageSummary) -> None:
    print()
    print("=" * 80)
    print(f" PR-K · package={s.package_id}  model={s.model}".upper())
    print("=" * 80)
    print(f"  requirements    : {s.requirement_count}")
    print(f"  pairs (req × tgt): {s.pair_count}")
    print(f"  LLM calls done  : {s.llm_calls_executed}")
    print(f"  LLM calls saved : {s.llm_calls_saved_by_skip}  (selector skip)")
    print(f"  wall time       : {s.elapsed_sec}s")
    print(f"  status          : {s.by_status}")
    print(f"  subcode         : {s.by_subcode}")
    print(f"  selected_k dist : {s.selected_k_dist}")
    print(f"  reranker_used   : {s.reranker_used_count}")
    print(f"  verifier_actions: {s.verifier_actions}")
    print(f"  grounding       : passed={s.grounded_judgments}  failed={s.ungrounded_judgments}")
    print(f"  low-conf rows   : {s.low_confidence_rows}")
    if s.warnings:
        print("  warnings:")
        for w in s.warnings:
            print(f"    - {w}")


def _print_review_rows(rows: List[Dict[str, Any]]) -> None:
    print()
    print("-" * 80)
    print(f" 30-row stratified review sample ({len(rows)} rows)".upper())
    print("-" * 80)
    for i, r in enumerate(rows, 1):
        print(
            f"\n[{i:2d}/{len(rows)}] {r['bucket']:<28} | id={r['short_id']:<8} | "
            f"role={r['target_doc_role']:<3} | type={r['requirement_type']:<22} | "
            f"status={r['status']:<8} subcode={r['status_subcode'] or '-':<24}\n"
            f"        suggested root-cause: {r['suggested_root_cause']}"
        )
        if r['req_text']:
            print(f"        req: {r['req_text']}")
        if r['winning_evidence_text']:
            print(f"        evid: {r['winning_evidence_text']}")
        if r['aggregation_reason']:
            print(f"        decision: {r['aggregation_reason'][:160]}")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path,
                        help="directory containing TZ/PZ/PMI .docx files")
    parser.add_argument("--package-id", default=None,
                        help="package identifier (default: dir name)")
    parser.add_argument("--model", default="qwen2.5:7b",
                        help="Ollama model (default qwen2.5:7b)")
    parser.add_argument("--tz", type=Path, help="explicit TZ .docx path")
    parser.add_argument("--pz", type=Path, help="explicit PZ .docx path")
    parser.add_argument("--pmi", type=Path, help="explicit PMI .docx path")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON output path (default scripts/_pr_k_real_<id>.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse + classify but skip the LLM run "
                             "(use the rule-only DisabledCoverageJudge)")
    args = parser.parse_args()

    pkg_dir: Path = args.package_dir
    if not pkg_dir.is_dir():
        print(f"ERROR: {pkg_dir} is not a directory")
        return 2

    package_id = args.package_id or pkg_dir.name
    out_path = args.out or (ROOT / "scripts" / f"_pr_k_real_{package_id}.json")

    # ── role detection ──
    docs: Dict[str, Path] = {}
    if args.tz: docs["tz"] = args.tz
    if args.pz: docs["pz"] = args.pz
    if args.pmi: docs["pmi"] = args.pmi
    for f in pkg_dir.glob("*.docx"):
        if f.name.startswith("~"):
            continue  # office lock files
        if any(f == p for p in docs.values()):
            continue
        role = _detect_role(f.name)
        if role and role not in docs:
            docs[role] = f
    missing = [r for r in ("tz", "pz", "pmi") if r not in docs]
    if missing:
        print(f"ERROR: could not find .docx for role(s): {missing}.")
        print(f"  detected: {[(k, v.name) for k, v in docs.items()]}")
        print(f"  pass --tz / --pz / --pmi explicitly")
        return 2

    print(f"package: {package_id}")
    for role, p in docs.items():
        print(f"  {role}: {p.name}")
    print(f"model: {args.model}  (dry-run={args.dry_run})")

    # ── parse all 3 docs ──
    print("\nparsing docs via Prepare-service subprocess …")
    parsed: Dict[str, Dict[str, Any]] = {}
    for role, p in docs.items():
        t = time.perf_counter()
        try:
            parsed[role] = _parse_doc_via_subprocess(p)
        except Exception as exc:
            print(f"  {role}: FAILED — {exc}")
            return 3
        print(f"  {role}: parsed {p.name} in {time.perf_counter() - t:.1f}s "
              f"(sections={len(parsed[role].get('sections', []))} "
              f"candidates={len(parsed[role].get('bq_candidates', []))})")

    # ── build request ──
    options: Dict[str, Any] = {
        "min_retrieval_score": 0.05,
        "initial_top_n": 20,
        "debug": True,
        "debug_max_candidates": 5,
        "enable_rule_verification": True,
        "enable_reranker": False,        # BGE not assumed available
        "reranker_mode": "conditional",
    }
    if args.dry_run:
        options.update({"enable_llm_judge": False, "judge_backend": "disabled"})
    else:
        options.update({
            "enable_llm_judge": True,
            "judge_backend": "ollama",
            "llm_model_name": args.model,
        })

    request = {
        "job_id": f"pr-k-real-{package_id}",
        "package_id": package_id,
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi", "pz"],
        "documents": [
            {
                "document_id": parsed[role]["document_id"],
                "doc_role": role,
                "prepared_artifact": _to_prepared_artifact(role, parsed[role]),
            }
            for role in ("tz", "pmi", "pz") if role in parsed
        ],
        "options": options,
    }

    # ── run ──
    cfg = CoverageConfig.from_options(options)
    cfg.llm.timeout = int(os.environ.get("CQUALITY_LLM_TIMEOUT", "180"))
    pipeline = CoverageAnalysisPipeline(config=cfg)
    print("\nrunning pipeline … (this may take a while)")
    t0 = time.perf_counter()
    result = pipeline.run(request)
    elapsed = time.perf_counter() - t0
    print(f"pipeline done in {elapsed:.1f}s")

    # ── summarise ──
    summary = _aggregate_summary(package_id, args.model if not args.dry_run else "(rule-only)", result, elapsed)
    review_rows = _pick_review_rows(result, n=30)

    _print_summary(summary)
    _print_review_rows(review_rows)

    # ── dump ──
    payload = {
        "summary": asdict(summary),
        "review_rows": review_rows,
        "request_options": options,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfull report written to: {out_path}")
    print("Open the JSON, fill `human_verdict` and `human_root_cause` on each "
          "review row, and we can iterate from there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
