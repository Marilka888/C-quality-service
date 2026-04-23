"""
Run the coverage pipeline on one real package and compare extraction modes.

Loads a data/packages/<id>.json (chunks format), wraps it as a pair of
prepared_artifacts, runs the pipeline twice:

  (A) requirement_extraction = "fragments"   — regex-trigger path
  (B) requirement_extraction = "model"       — fine-tuned classifier

Both runs share lemmatisation / unit-normalisation / entity-stoplist
improvements. Prints a side-by-side summary: extracted requirements,
coverage status counts, a few example requirement pairs.

Usage:
    python scripts/live_compare.py --pkg pkg_0005
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from collections import Counter

# Windows defaults to cp1251 for redirected stdout, which dies on non-ASCII
# characters like arrows and Russian text. Force UTF-8 unconditionally.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
from app.core.config import CoverageConfig


def _chunks_to_prepared(pkg: dict) -> dict:
    """Convert data/packages/<id>.json (chunks format) into the request
    shape expected by CoverageAnalysisPipeline.run().

    Chunks become fragments; no section structure is reconstructed (the
    pipeline's retrieval / classifier handle flat text too — "model" mode
    will fall back to per-fragment sentence extraction when sections[]
    is empty).
    """
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


def _summarise(result, label: str) -> None:
    d = result.model_dump() if hasattr(result, "model_dump") else result
    reqs = d.get("requirement_results") or []
    statuses = Counter(r.get("status") for r in reqs)
    print(f"\n=== {label} ===")
    print(f"  requirements:            {len(reqs)}")
    print(f"  status distribution:     {dict(statuses)}")
    print(f"  supporting pair_judgments: {len(d.get('pair_judgments') or [])}")
    print(f"  warnings:                {d.get('warnings') or []}")

    # A few examples
    covered = [r for r in reqs if r.get("status") == "COVERED"]
    partial = [r for r in reqs if r.get("status") == "PARTIAL"]
    missing = [r for r in reqs if r.get("status") == "MISSING"]
    for bucket, sample in (("COVERED", covered), ("PARTIAL", partial), ("MISSING", missing)):
        if sample:
            print(f"  sample {bucket}:")
            for r in sample[:2]:
                req_text = (r.get("requirement_text") or "")[:110]
                print(f"    - {req_text}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="pkg_0005")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Classifier threshold for 'model' mode.")
    ap.add_argument("--packages-dir", default="./data/packages")
    ap.add_argument("--with-c", action="store_true",
                    help="Include Run C (E5 + reranker). Heavy on CPU; skipped by default.")
    args = ap.parse_args()

    pkg_path = Path(args.packages_dir) / f"{args.pkg}.json"
    if not pkg_path.exists():
        raise SystemExit(f"Not found: {pkg_path}")
    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
    request = _chunks_to_prepared(pkg)

    print(f"Loaded {args.pkg}: {len(request['documents'])} documents")

    # --- Run A: fragments (regex) ---
    config_a = CoverageConfig()
    config_a.requirement_extraction = "fragments"
    # BoW embedding — avoids the 1 GB sentence-transformer CPU forward pass
    # per (req, unit) pair; retrieval is lexical + constraints + section,
    # which is what the rest of the live comparison is meant to exercise.
    config_a.embedding.backend = "bow"
    pipeline_a = CoverageAnalysisPipeline(config_a)
    res_a = pipeline_a.run(request)
    _summarise(res_a, f"A) requirement_extraction='fragments' (regex)")

    # --- Run B: model + BoW (previous baseline) ---
    config_b = CoverageConfig()
    config_b.requirement_extraction = "model"
    config_b.requirement_model.threshold = args.threshold
    config_b.embedding.backend = "bow"
    pipeline_b = CoverageAnalysisPipeline(config_b)
    res_b = pipeline_b.run(request)
    _summarise(res_b, f"B) model + BoW (classifier, thr={args.threshold})")

    # --- Run C: model + E5 embedder + cross-encoder reranker ---
    # (Heavy on CPU; skip by default — set --with-c to include)
    if args.with_c:
        config_c = CoverageConfig()
        config_c.requirement_extraction = "model"
        config_c.requirement_model.threshold = args.threshold
        config_c.embedding.backend = "e5"
        config_c.reranker.enabled = True
        pipeline_c = CoverageAnalysisPipeline(config_c)
        res_c = pipeline_c.run(request)
        _summarise(res_c, f"C) model + E5 + BGE reranker")
    else:
        res_c = None

    # --- Run D: model + BoW + cross-encoder judge (BGE-as-judge) ---
    # Key question this run answers: does BGE zero-shot judging outperform
    # the rule-based DisabledCoverageJudge on status decisions?
    config_d = CoverageConfig()
    config_d.requirement_extraction = "model"
    config_d.requirement_model.threshold = args.threshold
    config_d.embedding.backend = "bow"  # keep retrieval cheap
    config_d.reranker.enabled = False    # judge reuses BGE directly
    config_d.llm.enabled = True
    config_d.llm.backend = "cross_encoder"
    pipeline_d = CoverageAnalysisPipeline(config_d)
    res_d = pipeline_d.run(request)
    _summarise(res_d, f"D) model + BoW + cross-encoder judge")

    # --- Diff ---
    def _counts(r):
        if r is None:
            return None, None
        d = r.model_dump() if hasattr(r, "model_dump") else r
        results = d.get("requirement_results") or []
        return len(results), Counter(x.get("status") for x in results)

    n_a, sa = _counts(res_a)
    n_b, sb = _counts(res_b)
    n_c, sc = _counts(res_c)
    n_d, sd = _counts(res_d)

    def _short(k):
        return str(k).rsplit(".", 1)[-1].rstrip("'>")

    print("\n=== DIFF across runs ===")
    cols = [("A (regex)", n_a, sa),
            ("B (model+BoW)", n_b, sb)]
    if sc is not None:
        cols.append(("C (+E5+rerank)", n_c, sc))
    cols.append(("D (+CE judge)", n_d, sd))

    print("  " + f"{'metric':18s}" + "".join(f"{name:>18s}" for name, _, _ in cols))
    print("  " + f"{'requirements':18s}" + "".join(f"{n:>18d}" for _, n, _ in cols))
    all_statuses = set()
    for _, _, s in cols:
        if s is not None:
            all_statuses.update(s.keys())
    for status_enum in sorted(all_statuses, key=_short):
        label = _short(status_enum)
        row = "  " + f"{label:18s}"
        for _, _, s in cols:
            row += f"{(s.get(status_enum, 0) if s else 0):>18d}"
        print(row)


if __name__ == "__main__":
    main()
