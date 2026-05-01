"""
PR-I diagnostic: requirement-extraction coverage report for a TZ document.

Parses one .docx via Prepare-service (subprocess, to avoid the prepare /
C-quality `app/` namespace clash), runs RequirementBuilder, and prints
per-section diagnostics:

  * sections seen + flags (requirement-bearing / category)
  * requirements extracted per section (count + 3-5 examples)
  * total extracted requirements
  * distribution by RequirementType
  * distribution by source_section_id
  * marker_count vs extracted_count + LOW_REQUIREMENT_EXTRACTION_COVERAGE
    flag if applicable

Usage from C-quality venv:
  cd C:/Users/Marilka/PycharmProjects/C-quality-service
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/diag_requirement_extraction.py <docx_path>

Run without args to default to the «Череухо» TZ.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TZ = Path("C:/Users/Marilka/Pictures/Индивидуальное_ТЗ_Череухо_ВКР.docx")
PREP_HELPER = ROOT / "scripts" / "_prep_doc_to_json.py"


def _parse_doc(path: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(PREP_HELPER), str(path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"prepare subprocess failed: {proc.stderr}")
    return json.loads(proc.stdout)


def _build_request_artifact(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Convert prepare output → cPreparedArtifact-shape for RequirementBuilder."""
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
        meta = {
            "sectionId": c["source_section_id"],
            "sectionTitle": c["source_section_name"],
            "fragmentId": c["candidate_id"],
            "candidateType": c["candidate_type"],
            "sectionCategory": c["section_category"],
        }
        fragments.append({
            "fragment_id": c["candidate_id"],
            "text": c["text"],
            "section_id": c["source_section_id"],
            "metadata": meta,
        })
        if c["candidate_type"] in {
            "requirement_like",
            "documentation_requirement",
            "environment_requirement",
        }:
            requirement_candidates.append({
                "req_id": c["candidate_id"],
                "text": c["text"],
                "section_id": c["source_section_id"],
                "fragment_id": c["candidate_id"],
                "metadata": meta,
            })
    return {
        "document_id": parsed["document_id"],
        "doc_role": "tz",
        "sections": sections,
        "fragments": fragments,
        "requirement_candidates": requirement_candidates,
    }


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TZ
    if not src.exists():
        print(f"file not found: {src}", file=sys.stderr)
        return 1

    print(f"[1/3] parsing {src.name} via prepare-service …", flush=True)
    parsed = _parse_doc(src)
    artifact = _build_request_artifact(parsed)
    print(
        f"      sections={len(parsed['sections'])} candidates={len(parsed['bq_candidates'])} "
        f"requirement_candidates_for_C={len(artifact['requirement_candidates'])}",
        flush=True,
    )
    if parsed.get("warnings"):
        print(f"      warnings:")
        for w in parsed["warnings"]:
            print(f"        - {w}")

    print(f"[2/3] running RequirementBuilder …", flush=True)
    from app.application.use_cases.build_requirements import RequirementBuilder
    from app.application.use_cases.run_coverage_analysis import (
        _build_extraction_diagnostics,
    )
    requirements = RequirementBuilder().build(artifact)

    diag = _build_extraction_diagnostics(artifact, requirements)
    print(f"[3/3] diagnostics:", flush=True)
    print(f"      extracted_count={diag['extracted_count']}")
    print(f"      marker_count={diag['marker_count']}")
    print(f"      sections_seen={diag['sections_seen']}")
    print(f"      requirement_sections_seen={diag['requirement_sections_seen']}")
    print(f"      low_extraction_coverage={diag['low_extraction_coverage']}")
    if diag["suspected_reason"]:
        print(f"      suspected_reason: {diag['suspected_reason']}")

    print(f"\n--- per-section section info + extracted-count ---")
    by_section = diag["sections_per_extracted_req"]
    for s in parsed["sections"]:
        sid = s["id"]
        print(
            f"  id={sid!r:14s} L{s['level']} cat={s['category']!r:25s} "
            f"title={s['section_name'][:40]!r:42s} extracted={by_section.get(sid, 0)}"
        )

    print(f"\n--- distribution by RequirementType ---")
    by_type: Dict[str, int] = {}
    for r in requirements:
        t = r.requirement_type.value if hasattr(r.requirement_type, "value") else str(r.requirement_type)
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:32s} {n}")

    print(f"\n--- 5 example requirements per section ---")
    by_sec_examples: Dict[str, List[str]] = {}
    for r in requirements:
        sid = r.source_section_id or "(none)"
        by_sec_examples.setdefault(sid, []).append(r.text[:130])
    for sid, examples in by_sec_examples.items():
        print(f"  [{sid}] {len(examples)} requirement(s); first 5:")
        for ex in examples[:5]:
            print(f"    - {ex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
