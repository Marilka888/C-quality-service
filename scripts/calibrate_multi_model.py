"""
Multi-model coverage calibration: run the same package through several
LLM judges and produce a cross-model comparison report.

What it gives you (per package, single run):
  * Per-pair label table   (req × model → status)
  * Per-model status counts (COVERED / PARTIAL / MISSING / CONFLICT)
  * Per-model unavailability + ungrounded-demote counts
  * Pairwise agreement matrix (Cohen's kappa, simple % agreement)
  * Top-N disagreement examples (pairs where models diverge most)
  * Wall-clock latency per model

Why this exists:
  Lets a researcher answer "does coverage quality scale with model
  size / provider?" without re-engineering the pipeline. The judge
  abstraction is already in place — we just swap `model_name` per
  run and re-aggregate.

Free-tier-friendly: defaults are 5 cost-zero models. Add paid models
to the MODELS list when an API key is available.

Usage from C-quality venv:
  cd C:/Users/Marilka/PycharmProjects/C-quality-service
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/calibrate_multi_model.py [docx_path_TZ docx_path_PMI docx_path_PZ]

Defaults to the «Череухо» package.

API keys (set in env, only the providers you actually use):
  GROQ_API_KEY     — Groq Cloud (free tier, ~30 req/min)
  GEMINI_API_KEY   — Google AI Studio (free tier, 1500 req/day)
  CEREBRAS_API_KEY — Cerebras Cloud (free tier, 30 req/min)
  OPENAI_API_KEY   — OpenAI (paid)
  ANTHROPIC_API_KEY — Anthropic (paid)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOCS_DIR = Path("C:/Users/Marilka/Pictures")
DEFAULT_DOCS: List[Tuple[str, str]] = [
    ("tz",  "Индивидуальное_ТЗ_Череухо_ВКР.docx"),
    ("pmi", "Череухо_ПМИ.docx"),
    ("pz",  "Проект_ВКР_Череухо.docx"),
]
PREP_HELPER = ROOT / "scripts" / "_prep_doc_to_json.py"


# Default model lineup — all free-tier:
#   * 2 small open-source local (Ollama)
#   * 1 large open-source via Groq (free)
#   * 1 Google free-tier
#   * 1 Cerebras free-tier
#
# Each entry: (label_for_report, backend, model_name, env_var_required).
# Skipped at runtime if the env_var isn't set (so missing keys don't
# break the run; they just narrow the comparison).
MODELS: List[Dict[str, Any]] = [
    {"label": "ollama-qwen2.5-3b",   "backend": "ollama",  "model": "qwen2.5:3b",  "env": None},
    {"label": "ollama-qwen2.5-7b",   "backend": "ollama",  "model": "qwen2.5:7b",  "env": None},
    {"label": "groq-llama-3.3-70b",  "backend": "litellm", "model": "groq/llama-3.3-70b-versatile", "env": "GROQ_API_KEY"},
    {"label": "gemini-2.0-flash",    "backend": "litellm", "model": "gemini/gemini-2.0-flash",       "env": "GEMINI_API_KEY"},
    {"label": "cerebras-llama-3.1-8b",  "backend": "litellm", "model": "cerebras/llama3.1-8b",       "env": "CEREBRAS_API_KEY"},
    {"label": "cerebras-qwen-3-235b",   "backend": "litellm", "model": "cerebras/qwen-3-235b-a22b-instruct-2507", "env": "CEREBRAS_API_KEY"},
    # Add paid models when keys are available, e.g.:
    # {"label": "gpt-4o-mini", "backend": "litellm", "model": "openai/gpt-4o-mini", "env": "OPENAI_API_KEY"},
    # {"label": "claude-3.5-haiku", "backend": "litellm", "model": "anthropic/claude-3-5-haiku-latest", "env": "ANTHROPIC_API_KEY"},
]

# How many disagreement examples to surface in the report.
DISAGREEMENT_TOP_N = 8


# ── prepare-side parsing (subprocess to avoid namespace clash) ──────────


def _parse_doc(path: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(PREP_HELPER), str(path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"prepare subprocess failed for {path.name}: {proc.stderr}")
    return json.loads(proc.stdout)


_FRAGMENT_KIND = {
    "test_step": "test_step", "test_method": "test_step",
    "acceptance_criterion": "test_step", "test_conclusion": "test_step",
    "requirement_like": "paragraph", "documentation_requirement": "paragraph",
    "environment_requirement": "paragraph", "input_output_spec": "paragraph",
    "descriptive_context": "paragraph", "metadata": "paragraph", "noise": "paragraph",
}
_TZ_REQ_TYPES = {"requirement_like", "documentation_requirement", "environment_requirement"}


def _to_prepared_artifact(role: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    sections = [
        {"section_id": s["id"], "title": s["section_name"],
         "level": s["level"], "number": s.get("number")}
        for s in parsed["sections"]
    ]
    fragments, requirement_candidates = [], []
    for c in parsed["bq_candidates"]:
        kind = _FRAGMENT_KIND.get(c["candidate_type"], "paragraph")
        meta = {
            "sectionId": c["source_section_id"],
            "sectionTitle": c["source_section_name"],
            "fragmentId": c["candidate_id"],
            "candidateType": c["candidate_type"],
            "sectionCategory": c["section_category"],
        }
        fragments.append({
            "fragment_id": c["candidate_id"], "text": c["text"],
            "kind": kind, "section_id": c["source_section_id"], "metadata": meta,
        })
        if c["candidate_type"] in _TZ_REQ_TYPES:
            requirement_candidates.append({
                "req_id": c["candidate_id"], "text": c["text"],
                "section_id": c["source_section_id"],
                "fragment_id": c["candidate_id"], "metadata": meta,
            })
    return {
        "document_id": parsed["document_id"], "doc_role": role,
        "sections": sections, "fragments": fragments,
        "requirement_candidates": requirement_candidates,
    }


def _build_request(parsed: Dict[str, Dict[str, Any]], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Optional env knobs to scope a single calibration run:
    #   MULTI_MODEL_TOP_K           — top_k for retrieval shortlist
    #                                 (default 1 — one LLM call per pair).
    #                                 docback prod also uses 1; setting to
    #                                 5 multiplies call count by 5 and rarely
    #                                 changes verdicts. Raise only when you
    #                                 specifically want to study how rank-2
    #                                 evidence differs from rank-1.
    #   MULTI_MODEL_TARGET_ROLES    — comma-separated subset of target roles.
    #                                 Default "pmi,pz". "pmi" alone halves
    #                                 the call count and keeps the most
    #                                 interesting axis (test-coverage).
    top_k = int(os.environ.get("MULTI_MODEL_TOP_K", "1"))
    roles_env = os.environ.get("MULTI_MODEL_TARGET_ROLES", "pmi,pz")
    target_roles = [r.strip().lower() for r in roles_env.split(",") if r.strip()]

    documents = []
    for r in ("tz", *target_roles):
        if r not in parsed:
            continue
        documents.append({
            "document_id": parsed[r]["document_id"],
            "doc_role": r,
            "prepared_artifact": _to_prepared_artifact(r, parsed[r]),
        })

    return {
        "job_id": f"calibrate-{model_cfg['label']}",
        "package_id": "cherevuyhho",
        "source_doc_role": "tz",
        "target_doc_roles": target_roles,
        "documents": documents,
        "options": {
            "enable_llm_judge": True,
            "judge_backend": model_cfg["backend"],
            "llm_model_name": model_cfg["model"],
            "min_retrieval_score": 0.0,
            "evidence_floor": 0.5,
            "top_k": top_k,
        },
    }


# ── pairwise agreement metrics ──────────────────────────────────────────


def _cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Cohen's kappa for two raters over the same N items."""
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    cats = sorted(set(labels_a) | set(labels_b))
    if len(cats) <= 1:
        return 1.0
    cat_idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)

    confusion = [[0] * K for _ in range(K)]
    for a, b in zip(labels_a, labels_b):
        confusion[cat_idx[a]][cat_idx[b]] += 1
    po = sum(confusion[i][i] for i in range(K)) / n
    row_marg = [sum(row) / n for row in confusion]
    col_marg = [sum(confusion[r][c] for r in range(K)) / n for c in range(K)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(K))
    if abs(1 - pe) < 1e-9:
        return 1.0
    return (po - pe) / (1 - pe)


def _simple_agreement(labels_a: List[str], labels_b: List[str]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    return sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a)


# ── main ────────────────────────────────────────────────────────────────


def main() -> int:
    if len(sys.argv) >= 4:
        docs = [
            ("tz",  Path(sys.argv[1])),
            ("pmi", Path(sys.argv[2])),
            ("pz",  Path(sys.argv[3])),
        ]
    else:
        docs = [(role, DOCS_DIR / fname) for role, fname in DEFAULT_DOCS]

    # ── parse all documents once ────────────────────────────────────────
    parsed: Dict[str, Dict[str, Any]] = {}
    print(f"[1/3] parsing {len(docs)} documents …", flush=True)
    for role, path in docs:
        if not path.exists():
            print(f"  ! missing {role}: {path}", file=sys.stderr)
            continue
        parsed[role] = _parse_doc(path)
        n_cands = len(parsed[role]["bq_candidates"])
        print(f"  {role}: sections={len(parsed[role]['sections'])} candidates={n_cands}", flush=True)

    if "tz" not in parsed:
        print("FATAL: no TZ document parsed; cannot run multi-model calibration.", file=sys.stderr)
        return 1

    # ── filter MODELS by env-key availability + backend filter ──────────
    # Optional env filter so a researcher can run a subset without
    # editing the script: set MULTI_MODEL_BACKENDS_INCLUDE to a
    # comma-separated list of backend names ("ollama", "litellm") and
    # only those will run. Empty/unset = include all.
    backend_filter_raw = os.environ.get("MULTI_MODEL_BACKENDS_INCLUDE", "").strip()
    backend_filter = (
        {b.strip().lower() for b in backend_filter_raw.split(",") if b.strip()}
        if backend_filter_raw else None
    )
    if backend_filter:
        print(f"  filter: only backends {sorted(backend_filter)}", flush=True)

    available_models: List[Dict[str, Any]] = []
    for m in MODELS:
        if backend_filter and m["backend"].lower() not in backend_filter:
            print(f"  SKIP {m['label']} (backend {m['backend']!r} excluded by MULTI_MODEL_BACKENDS_INCLUDE)", flush=True)
            continue
        if m["env"] and not os.environ.get(m["env"]):
            print(f"  SKIP {m['label']} ({m['env']} not set)", flush=True)
            continue
        available_models.append(m)
    if not available_models:
        print("FATAL: no model has its API key set in env.", file=sys.stderr)
        return 1
    print(f"[2/3] running pipeline for {len(available_models)} models:")
    for m in available_models:
        print(f"  - {m['label']:32s} backend={m['backend']:10s} model={m['model']}", flush=True)

    # ── per-model run ───────────────────────────────────────────────────
    from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
    pipeline = CoverageAnalysisPipeline()

    # PR-J checkpoint persistence: each model's result is dumped to
    # `scripts/_partial_<label>.json` immediately after the model
    # finishes. Subsequent runs of the same script reuse those partials
    # (skip re-running models we already have data for) so a kill in
    # the middle of model N doesn't lose model 1..N-1.
    per_model_results: Dict[str, Dict[str, Any]] = {}
    partials_dir = ROOT / "scripts"
    skip_partials = os.environ.get("MULTI_MODEL_IGNORE_PARTIALS", "").strip().lower() in {"1", "true", "yes"}

    for m in available_models:
        label = m["label"]
        partial_path = partials_dir / f"_partial_{label}.json"

        if not skip_partials and partial_path.exists():
            try:
                with partial_path.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                # Restore pair_labels keys (JSON converts tuples to lists).
                cached["pair_labels"] = {
                    tuple(k.split("\x01", 1)): v for k, v in cached.get("pair_labels", {}).items()
                }
                per_model_results[label] = cached
                print(f"\n[skip] {label} — using cached partial "
                      f"({partial_path.name}, pairs={cached.get('total_pairs', 0)}). "
                      f"Set MULTI_MODEL_IGNORE_PARTIALS=1 to re-run.",
                      flush=True)
                continue
            except Exception as exc:
                print(f"  ! could not load cached partial for {label}: {exc}; re-running", file=sys.stderr)

        request = _build_request(parsed, m)
        print(f"\n[run] {label} …", flush=True)
        t0 = time.time()
        try:
            result = pipeline.run(request)
        except Exception as exc:
            print(f"  ! pipeline failed for {label}: {exc}", file=sys.stderr)
            per_model_results[label] = {"error": str(exc)}
            continue
        elapsed = time.time() - t0
        # Status counts.
        statuses = Counter(r.status.value for r in result.requirement_results)
        # Per-pair label list, indexed by stable pair_key (req_id, target_doc_id).
        pair_labels: Dict[Tuple[str, str], str] = {
            (r.req_id, r.target_document_id): r.status.value
            for r in result.requirement_results
        }
        warns = list(result.warnings or [])
        ungrounded = sum(
            1 for j in (result.pair_judgments or [])
            if getattr(j, "low_confidence", False)
            and "[ungrounded]" in (j.explanation or "")
        )
        entry = {
            "status_counts": dict(statuses),
            "pair_labels": pair_labels,
            "warnings": warns,
            "ungrounded_demotes": ungrounded,
            "elapsed_seconds": round(elapsed, 1),
            "total_pairs": len(result.requirement_results),
        }
        per_model_results[label] = entry
        print(f"  done in {elapsed:.0f}s; pairs={len(pair_labels)} statuses={dict(statuses)}", flush=True)

        # Persist the partial RIGHT NOW so a subsequent kill never
        # loses this model. Tuples → "req_id\x01target_doc_id" string
        # so JSON is happy; reload step parses them back.
        try:
            json_safe = dict(entry)
            json_safe["pair_labels"] = {f"{k[0]}\x01{k[1]}": v for k, v in pair_labels.items()}
            with partial_path.open("w", encoding="utf-8") as f:
                json.dump(json_safe, f, ensure_ascii=False, indent=2)
            print(f"  cached partial → {partial_path.name}", flush=True)
        except Exception as exc:
            print(f"  ! could not write partial for {label}: {exc}", file=sys.stderr)

    # ── cross-model agreement matrix ────────────────────────────────────
    # Use the union of pairs that all models judged on. If a model
    # missed a pair (e.g. errored), it's excluded from comparisons
    # involving that pair.
    all_pair_keys = set()
    for r in per_model_results.values():
        if "pair_labels" in r:
            all_pair_keys.update(r["pair_labels"].keys())
    all_pair_keys_sorted = sorted(all_pair_keys)

    labels_have = [m for m in per_model_results if "pair_labels" in per_model_results[m]]
    kappa_matrix: Dict[str, Dict[str, float]] = {}
    agree_matrix: Dict[str, Dict[str, float]] = {}
    for a in labels_have:
        kappa_matrix[a] = {}
        agree_matrix[a] = {}
        for b in labels_have:
            common = [
                k for k in all_pair_keys_sorted
                if k in per_model_results[a]["pair_labels"]
                and k in per_model_results[b]["pair_labels"]
            ]
            la = [per_model_results[a]["pair_labels"][k] for k in common]
            lb = [per_model_results[b]["pair_labels"][k] for k in common]
            kappa_matrix[a][b] = round(_cohen_kappa(la, lb), 3)
            agree_matrix[a][b] = round(_simple_agreement(la, lb), 3)

    # ── disagreement examples: pairs with highest label diversity ───────
    disagreements: List[Tuple[Tuple[str, str], Counter]] = []
    for k in all_pair_keys_sorted:
        labels_for_pair = Counter()
        for m in labels_have:
            if k in per_model_results[m]["pair_labels"]:
                labels_for_pair[per_model_results[m]["pair_labels"][k]] += 1
        # diversity = number of distinct labels assigned across models
        if len(labels_for_pair) >= 2:
            disagreements.append((k, labels_for_pair))
    # Sort by diversity desc, then by deviation from mode.
    disagreements.sort(key=lambda x: (-len(x[1]), -sum(x[1].values())))

    # ── write Markdown report ───────────────────────────────────────────
    print(f"\n[3/3] writing report …", flush=True)
    md = ["# Multi-model coverage calibration", ""]
    md.append(f"Package: parsed sections per role: " +
              ", ".join(f"{r}={len(parsed[r]['sections'])}" for r in parsed))
    md.append(f"Models compared: **{len(labels_have)}**")
    md.append("")
    md.append("## Per-model status distribution")
    md.append("")
    md.append("| Model | total pairs | COVERED | PARTIAL | MISSING | CONFLICT | ungrounded | warnings | wall-time s |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for label in labels_have:
        r = per_model_results[label]
        sc = r.get("status_counts", {})
        md.append(
            f"| {label} | {r['total_pairs']} | "
            f"{sc.get('COVERED', 0)} | {sc.get('PARTIAL', 0)} | "
            f"{sc.get('MISSING', 0)} | {sc.get('CONFLICT', 0)} | "
            f"{r.get('ungrounded_demotes', 0)} | {len(r.get('warnings', []))} | "
            f"{r.get('elapsed_seconds', 0)} |"
        )

    md.append("")
    md.append("## Cohen's kappa (pairwise agreement)")
    md.append("Higher = more agreement. <0.4 = poor; 0.4-0.6 = moderate; >0.6 = substantial.")
    md.append("")
    md.append("| | " + " | ".join(labels_have) + " |")
    md.append("|" + "|".join(["---"] * (len(labels_have) + 1)) + "|")
    for a in labels_have:
        row = [a] + [f"{kappa_matrix[a][b]:.3f}" for b in labels_have]
        md.append("| " + " | ".join(row) + " |")

    md.append("")
    md.append("## Simple agreement % (same status on same pair)")
    md.append("")
    md.append("| | " + " | ".join(labels_have) + " |")
    md.append("|" + "|".join(["---"] * (len(labels_have) + 1)) + "|")
    for a in labels_have:
        row = [a] + [f"{int(agree_matrix[a][b] * 100)}%" for b in labels_have]
        md.append("| " + " | ".join(row) + " |")

    md.append("")
    md.append(f"## Top-{DISAGREEMENT_TOP_N} disagreements")
    md.append("Pairs where models give the most varied verdicts.")
    md.append("")
    md.append("| Pair (req_id::target) | Per-model verdicts |")
    md.append("|---|---|")
    for (req_id, tgt), counts in disagreements[:DISAGREEMENT_TOP_N]:
        cells = []
        for m in labels_have:
            v = per_model_results[m]["pair_labels"].get((req_id, tgt), "—")
            cells.append(f"{m}={v}")
        md.append(f"| `{req_id[:48]}::{tgt[:8]}` | {' / '.join(cells)} |")

    md.append("")
    md.append("## Detail dump")
    md.append("```json")
    serializable = {
        label: {
            **{k: v for k, v in r.items() if k != "pair_labels"},
            "n_pair_labels": len(r.get("pair_labels", {})),
        }
        for label, r in per_model_results.items()
    }
    md.append(json.dumps(serializable, ensure_ascii=False, indent=2))
    md.append("```")

    out = ROOT / "scripts" / "calibrate_multi_model_result.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
