"""
Helper: parse one .docx via Prepare-service's PreparePipeline and dump
sections + bq_candidates as JSON to stdout. Called from calibrate_pr_c.py
in a subprocess so the parent can avoid importing Prepare-service's
`app/` namespace (which clashes with C-quality's `app/`).

Usage: PYTHONIOENCODING=utf-8 python _prep_doc_to_json.py <docx_path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PREPARE = Path(__file__).resolve().parents[2] / "Prepare-service"
sys.path.insert(0, str(PREPARE))

from app.services.pipeline.prepare import PreparePipeline   # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _prep_doc_to_json.py <docx_path>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    res = PreparePipeline().run(path.read_bytes(), path.name)
    out = {
        "document_id": res.document_id,
        "filename": res.filename,
        "sections": [
            {"id": s.id, "section_name": s.section_name,
             "level": s.level, "text": s.text, "number": s.number,
             "category": (s.category.value if s.category else None)}
            for s in res.sections
        ],
        "bq_candidates": [
            {"candidate_id": c.candidate_id,
             "source_section_id": c.source_section_id,
             "source_section_name": c.source_section_name,
             "section_category": c.section_category,
             "candidate_type": c.candidate_type,
             "text": c.text}
            for c in res.bq_candidates
        ],
        "warnings": list(res.warnings),
    }
    json.dump(out, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
