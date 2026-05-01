"""
PR-C calibration: end-to-end run on the «Череухо» package with a real
Ollama LLM judge to measure the BUG-3 (grounding) and BUG-9 (evidence
floor) effects on real data.

Approach:
  1) Parse all three .docx files via Prepare-service's PreparePipeline.
  2) Convert prepare-side output (Section / Candidate) into the
     cPreparedArtifact shape that docback would normally build.
  3) Run CoverageAnalysisPipeline ONCE in NEW mode (current code) with
     enable_llm_judge=True against local Ollama. Capture:
       - per-status counts
       - low_confidence count (BUG-9 + BUG-3)
       - grounding-demoted count (BUG-3 alone — "[ungrounded]" in explanation)
       - DUPLICATE_PAIRS warning hits (BUG-14)
  4) For "legacy" comparison we monkey-patch _parse_response inside the
     calibration script so it skips the grounding gate AND recover what
     the LLM originally said via raw labels stashed in PairJudgment.metadata-
     like helper. We do NOT re-run the LLM — same judge calls feed both views.

Run from C-quality venv:
  cd C:/Users/Marilka/PycharmProjects/C-quality-service
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/calibrate_pr_c.py

Requires Ollama running on http://localhost:11434 with model qwen2.5:3b.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a standalone script. Note: we deliberately do NOT add
# Prepare-service to sys.path here — its `app/` namespace clashes with
# C-quality's `app/`. Parsing of .docx is delegated to a subprocess that
# imports the prepare-side `app/` in isolation.
ROOT = Path(__file__).resolve().parents[1]              # C-quality-service
PREPARE = ROOT.parent / "Prepare-service"
sys.path.insert(0, str(ROOT))

DOCS_DIR = Path("C:/Users/Marilka/Pictures")
DOC_FILES: List[Tuple[str, str]] = [
    ("tz",  "Индивидуальное_ТЗ_Череухо_ВКР.docx"),
    ("pz",  "Проект_ВКР_Череухо.docx"),
    ("pmi", "Череухо_ПМИ.docx"),
]


# ── Conversion: prepare → C-quality cPreparedArtifact shape ─────────────

# Mirror of docback's c_job.go::candidateTypeToFragmentKind
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

# TZ requirement candidate types — same split docback's prepared_builder uses.
_TZ_REQ_TYPES = {
    "requirement_like", "documentation_requirement", "environment_requirement",
}


def _parse_doc_via_subprocess(path: Path) -> Dict[str, Any]:
    """Run _prep_doc_to_json.py in a subprocess and return its JSON output.
    Avoids the prepare/C-quality `app/` namespace clash."""
    helper = ROOT / "scripts" / "_prep_doc_to_json.py"
    proc = subprocess.run(
        [sys.executable, str(helper), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"prepare subprocess failed for {path.name}: rc={proc.returncode}\n"
            f"stderr: {proc.stderr}"
        )
    return json.loads(proc.stdout)


def _to_prepared_artifact(role: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Build the cPreparedArtifact-shape dict that CoverageAnalysisPipeline
    expects. Mirrors docback's c_job.go::buildCoverageRequest."""
    sections = [
        {
            "section_id": s["id"],
            "title": s["section_name"],
            "level": s["level"],
            "number": s.get("number"),
        }
        for s in parsed["sections"]
    ]
    fragments = []
    requirement_candidates = []
    for c in parsed["bq_candidates"]:
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


# ── Calibration probe: capture raw-label and grounding decisions ────────

# We intercept _parse_response so each call records (req, unit, raw_label,
# grounded, low_confidence) in a global list. This lets us reconstruct
# legacy behaviour without re-running the LLM.

_PROBE: List[Dict[str, Any]] = []


def _patched_parse_response(orig):
    from app.domain.c_quality_enums import LLMLabel

    def wrapper(raw, req_id, unit_id, doc_id, evidence_text=""):
        # Decode raw_label as the LLM saw it BEFORE any post-validation
        # (constraint-empty CONFLICT downgrade, grounding gate).
        raw_label_str = str(raw.get("label", "IRRELEVANT")).upper()
        valid = {l.value for l in LLMLabel}
        original_label = raw_label_str if raw_label_str in valid else "IRRELEVANT"

        judgment = orig(raw, req_id, unit_id, doc_id, evidence_text=evidence_text)

        _PROBE.append({
            "req_id": req_id,
            "unit_id": unit_id,
            "target_document_id": doc_id,
            "original_llm_label": original_label,
            "final_label": judgment.llm_label.value,
            "low_confidence": judgment.low_confidence,
            "cited_phrases_count": len(judgment.cited_phrases),
            "ungrounded_demote": judgment.low_confidence and "[ungrounded]" in (judgment.explanation or ""),
        })
        return judgment
    return wrapper


def _apply_probe():
    from app.infrastructure.llm import ollama_coverage_judge as ocj
    orig = ocj._parse_response
    ocj._parse_response = _patched_parse_response(orig)


# ── Build & run ─────────────────────────────────────────────────────────


def _build_request(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "job_id": "calibrate-pr-c",
        "package_id": "cherevuyhho",
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
        "options": {
            "enable_llm_judge": True,
            "llm_backend": "ollama",
            "llm_model_name": "qwen2.5:3b",
            "min_retrieval_score": 0.0,
            "evidence_floor": 0.5,
        },
    }


def _summarize(result, probe_rows: List[dict]) -> Dict[str, Any]:
    by_status_new: Dict[str, int] = {}
    low_conf_count = 0
    for r in result.requirement_results:
        by_status_new[r.status.value] = by_status_new.get(r.status.value, 0) + 1
        if r.low_confidence:
            low_conf_count += 1

    # Reconstruct legacy view: pretend grounding gate didn't exist. The
    # original_llm_label is what the LLM said before any demotion. Build
    # legacy per-(req, target) status from those.
    from app.application.use_cases.aggregate_coverage import _STATUS_RANK, _label_to_status
    from app.domain.c_quality_enums import LLMLabel

    legacy_best_by_pair: Dict[tuple, str] = {}
    for row in probe_rows:
        try:
            lbl = LLMLabel(row["original_llm_label"])
        except ValueError:
            lbl = LLMLabel.IRRELEVANT
        status = _label_to_status(lbl).value
        key = (row["req_id"], row["target_document_id"])
        prev = legacy_best_by_pair.get(key)
        if prev is None or _STATUS_RANK[__import__(
            "app.domain.c_quality_enums", fromlist=["CoverageStatus"],
        ).CoverageStatus(status)] > _STATUS_RANK[__import__(
            "app.domain.c_quality_enums", fromlist=["CoverageStatus"],
        ).CoverageStatus(prev)]:
            legacy_best_by_pair[key] = status

    by_status_legacy: Dict[str, int] = {}
    # Legacy count includes only pairs that produced ANY judgment. Pairs
    # with empty shortlist landed in MISSING in both views — add them now.
    pairs_with_judgments = set(legacy_best_by_pair.keys())
    for r in result.requirement_results:
        key = (r.req_id, r.target_document_id)
        if key in pairs_with_judgments:
            by_status_legacy[legacy_best_by_pair[key]] = by_status_legacy.get(
                legacy_best_by_pair[key], 0) + 1
        else:
            by_status_legacy["MISSING"] = by_status_legacy.get("MISSING", 0) + 1

    ungrounded_demotes = sum(1 for r in probe_rows if r["ungrounded_demote"])
    duplicate_warnings = [w for w in result.warnings if w.startswith("DUPLICATE_PAIRS:")]

    return {
        "total_pairs":            len(result.requirement_results),
        "judgments_made":         len(probe_rows),
        "by_status_legacy":       by_status_legacy,
        "by_status_new":          by_status_new,
        "low_confidence_results": low_conf_count,
        "ungrounded_demotes":     ungrounded_demotes,
        "dedup_warnings":         duplicate_warnings,
        "all_warnings_count":     len(result.warnings),
    }


def _md_table(legacy: Dict[str, int], new: Dict[str, int]) -> List[str]:
    statuses = ["COVERED", "PARTIAL", "MISSING", "CONFLICT"]
    lines = ["| Status | Legacy | New |", "|---|---|---|"]
    for s in statuses:
        lines.append(f"| {s} | {legacy.get(s, 0)} | {new.get(s, 0)} |")
    return lines


def main() -> int:
    print("[1/4] Parsing 3 .docx via PreparePipeline (subprocess) …", flush=True)
    parsed: Dict[str, Dict[str, Any]] = {}
    for role, fname in DOC_FILES:
        path = DOCS_DIR / fname
        d = _parse_doc_via_subprocess(path)
        parsed[role] = d
        print(f"    {role}: {len(d['sections'])} sections, {len(d['bq_candidates'])} candidates",
              flush=True)

    print("[2/4] Applying probe (capture raw LLM labels) …", flush=True)
    _apply_probe()

    print("[3/4] Running CoverageAnalysisPipeline with Ollama qwen2.5:3b …", flush=True)
    from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
    pipeline = CoverageAnalysisPipeline()
    request = _build_request(parsed)
    result = pipeline.run(request)
    print(f"    requirements={len(result.requirement_results)} pair_judgments_made={len(_PROBE)}",
          flush=True)

    print("[4/4] Summarising …", flush=True)
    summary = _summarize(result, _PROBE)

    out_md = ROOT / "scripts" / "calibrate_pr_c_result.md"
    md = ["# PR-C calibration on Cherevuyhho (live Ollama qwen2.5:3b)", ""]
    md.append(f"Total (req × target) pairs: **{summary['total_pairs']}**  ")
    md.append(f"LLM judge calls: **{summary['judgments_made']}**  ")
    md.append(f"Low-confidence results (BUG-3 grounding + BUG-9 floor combined): **{summary['low_confidence_results']}**  ")
    md.append(f"Ungrounded LLM demotions (BUG-3 alone): **{summary['ungrounded_demotes']}**  ")
    md.append(f"Duplicate-pair dedup warnings (BUG-14): **{len(summary['dedup_warnings'])}**")
    md.append("")
    md.append("## Status distribution: legacy → new")
    md.append("Legacy = LLM raw label, no grounding gate, no evidence floor.  ")
    md.append("New = current code: grounding-demoted ungrounded verdicts to IRRELEVANT, low_confidence flagged on results below evidence_floor=0.5.")
    md.append("")
    md.extend(_md_table(summary["by_status_legacy"], summary["by_status_new"]))
    md.append("")
    md.append("## Detail dump")
    md.append("```json")
    md.append(json.dumps(summary, ensure_ascii=False, indent=2))
    md.append("```")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWritten: {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
