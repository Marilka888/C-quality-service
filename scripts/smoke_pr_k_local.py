"""
PR-K smoke / e2e quality run against a LOCAL Ollama judge.

Goal: exercise every PR-K branch (applicability skip, adaptive selector,
conditional reranker, evidence-based aggregator, evidence_trace) on a
synthetic-but-realistic TZ↔PMI/PZ package and compare PR-K verdicts to
the DisabledCoverageJudge baseline.

Design notes:
  * No production config is mutated. The script reads its tuning from
    env vars; falls back to PR-K defaults otherwise.
  * No specific real package is hardcoded — the fixture is a small in-
    memory TZ + PMI + PZ that covers all type-classes (FUNCTIONAL,
    PERFORMANCE, SECURITY, DELIVERY, ARCHITECTURE, INTERFACE,
    DOCUMENTATION) so each PR-K branch is touched at least once.
  * If Ollama is unreachable, the script downgrades gracefully to the
    DisabledCoverageJudge baseline ONLY and prints LLM_UNAVAILABLE.
  * Output is two summary tables (LLM run + disabled-judge baseline)
    plus one JSON dump of evidence_trace per row to stdout. Nothing is
    written to disk by default.

Env vars (all optional):
  CQUALITY_LLM_MODELS        — comma-separated Ollama model names
                               (default qwen2.5:3b). Each model runs as a
                               separate sweep section.
  CQUALITY_LLM_MODEL         — single-model alias (overridden by _MODELS)
  CQUALITY_LLM_TIMEOUT       — judge timeout in seconds (default 60)
  OLLAMA_URL                 — full URL to /api/generate (default localhost)
  CQUALITY_DEBUG_ENABLED     — bool (default true)
  CQUALITY_DEBUG_MAX_CANDIDATES (default 5)
  CQUALITY_RERANKER_ENABLED  — bool (default false)  [reranker needs BGE]
  CQUALITY_RERANKER_MODE     — always|conditional (default conditional)
  CQUALITY_INITIAL_TOP_N     — int (default 10)
  CQUALITY_MIN_RETRIEVAL     — float (default 0.05)

Run from C-quality venv:

  cd C:/Users/Marilka/PycharmProjects/C-quality-service
  .venv/Scripts/python.exe scripts/smoke_pr_k_local.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# Allow running as standalone script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.application.use_cases.run_coverage_analysis import CoverageAnalysisPipeline
from app.core.config import CoverageConfig


# ── Fixture ─────────────────────────────────────────────────────────────


def _build_fixture_request(options: Dict[str, Any]) -> Dict[str, Any]:
    """A 7-row synthetic package that probes every PR-K branch."""
    tz_fragments = [
        # 1. FUNCTIONAL — should be COVERED
        {
            "fragment_id": "tz-f1", "section_id": "4.1",
            "text": "Система должна обеспечивать аутентификацию пользователей "
                    "через единую учётную запись.",
            "kind": "paragraph",
        },
        # 2. PERFORMANCE with numeric constraint — critical, broad sweep
        {
            "fragment_id": "tz-f2", "section_id": "4.2",
            "text": "Время отклика системы при типовой нагрузке не должно "
                    "превышать 2 секунд.",
            "kind": "paragraph",
        },
        # 3. STORAGE with numeric — CONFLICT (TZ 90 vs PMI 30 days)
        {
            "fragment_id": "tz-f3", "section_id": "4.3",
            "text": "Журнал событий безопасности должен храниться не менее "
                    "90 дней с момента записи.",
            "kind": "paragraph",
        },
        # 4. SECURITY — critical, expect COVERED via PZ
        {
            "fragment_id": "tz-f4", "section_id": "4.4",
            "text": "Система должна обеспечивать защиту от SQL-инъекций "
                    "при обработке пользовательского ввода.",
            "kind": "paragraph",
        },
        # 5. DELIVERY_REQUIREMENT — OUT_OF_SCOPE everywhere
        {
            "fragment_id": "tz-f5", "section_id": "5.1",
            "text": "Документация по проекту должна быть представлена в LMS "
                    "и проверена через систему «Антиплагиат».",
            "kind": "paragraph",
        },
        # 6. ARCHITECTURE_IMPLEMENTATION — NOT_APPLICABLE in PMI, COVERED in PZ
        {
            "fragment_id": "tz-f6", "section_id": "4.5",
            "text": "Backend-часть должна быть реализована на Python с "
                    "использованием FastAPI.",
            "kind": "paragraph",
        },
        # 7. INTERFACE — OPTIONAL on PZ, expect MISSING (no PZ coverage),
        #    OPTIONAL_NOT_FOUND
        {
            "fragment_id": "tz-f7", "section_id": "4.6",
            "text": "Пользовательский интерфейс должен быть выполнен в "
                    "соответствии с макетами Figma.",
            "kind": "paragraph",
        },
        # 8. PR-K grounding-gate probe — a FUNCTIONAL requirement whose
        #    only "match" is a distractor PMI fragment with similar tokens
        #    but no actual coverage. A small LLM might hallucinate COVERED;
        #    grounding gate must demote to MISSING_LOW_GROUNDING when the
        #    cited phrase isn't in the evidence.
        {
            "fragment_id": "tz-f8", "section_id": "4.7",
            "text": "Система должна поддерживать экспорт отчётов в "
                    "формате PDF с электронной подписью.",
            "kind": "paragraph",
        },
    ]
    pmi_fragments = [
        # Covers TZ-1
        {
            "fragment_id": "pmi-f1", "section_id": "3.1",
            "text": "Проверить, что пользователь может войти в систему "
                    "через единую учётную запись и получить токен доступа.",
            "kind": "test_step",
        },
        # Covers TZ-2 (performance, broad sweep should find it)
        {
            "fragment_id": "pmi-f2", "section_id": "3.2",
            "text": "Замерить время отклика системы при 100 одновременных "
                    "запросах; результат не должен превышать 2 секунд.",
            "kind": "test_step",
        },
        # CONFLICT vs TZ-3 — different value (30 vs 90)
        {
            "fragment_id": "pmi-f3", "section_id": "3.3",
            "text": "Проверить, что журнал событий хранится за последние "
                    "30 суток.",
            "kind": "test_step",
        },
        # Distractor — looks like coverage of TZ-2 but is unrelated
        {
            "fragment_id": "pmi-f4", "section_id": "3.4",
            "text": "Проверить корректное завершение работы при превышении "
                    "лимита подключений.",
            "kind": "test_step",
        },
        # Lure for TZ-8 (export PDF) — superficially matches "формат" /
        # "отчёт" tokens but doesn't actually describe the export feature.
        # If a small LLM marks COVERED here without a grounded cite, the
        # PR-K grounding gate must demote it.
        {
            "fragment_id": "pmi-f5", "section_id": "3.5",
            "text": "Проверить, что отчёты сохраняются в каталоге "
                    "/var/log в текстовом виде.",
            "kind": "test_step",
        },
    ]
    pz_fragments = [
        # Covers TZ-4 (security)
        {
            "fragment_id": "pz-f1", "section_id": "2.1",
            "text": "Защита от SQL-инъекций реализована через параметризованные "
                    "запросы и валидацию пользовательского ввода.",
            "kind": "paragraph",
        },
        # Covers TZ-6 (architecture)
        {
            "fragment_id": "pz-f2", "section_id": "2.2",
            "text": "Серверная часть приложения построена на языке Python "
                    "с применением фреймворка FastAPI.",
            "kind": "paragraph",
        },
        # Distractor
        {
            "fragment_id": "pz-f3", "section_id": "2.3",
            "text": "Развёртывание системы выполняется средствами Docker "
                    "Compose.",
            "kind": "paragraph",
        },
    ]

    return {
        "job_id": "smoke-pr-k",
        "package_id": "smoke-fixture",
        "source_doc_role": "tz",
        "target_doc_roles": ["pmi", "pz"],
        "documents": [
            {
                "document_id": "doc-tz", "doc_role": "tz",
                "prepared_artifact": {
                    "document_id": "doc-tz", "doc_role": "tz",
                    "fragments": tz_fragments,
                    "sections": [
                        {"section_id": "4.1", "title": "Функциональные требования",
                         "category": "requirements"},
                        {"section_id": "4.2", "title": "Производительность",
                         "category": "requirements"},
                        {"section_id": "4.3", "title": "Хранение данных",
                         "category": "requirements"},
                        {"section_id": "4.4", "title": "Безопасность",
                         "category": "requirements"},
                        {"section_id": "4.5", "title": "Архитектура",
                         "category": "requirements"},
                        {"section_id": "4.6", "title": "Интерфейс",
                         "category": "requirements"},
                        {"section_id": "5.1", "title": "Сдача документации",
                         "category": "process"},
                    ],
                    "requirement_candidates": [
                        {"text": f["text"], "section_id": f["section_id"],
                         "fragment_id": f["fragment_id"], "type": "requirement_like"}
                        for f in tz_fragments
                    ],
                },
            },
            {
                "document_id": "doc-pmi", "doc_role": "pmi",
                "prepared_artifact": {
                    "document_id": "doc-pmi", "doc_role": "pmi",
                    "fragments": pmi_fragments,
                    "sections": [
                        {"section_id": "3.1", "title": "Тесты функциональности"},
                        {"section_id": "3.2", "title": "Тесты производительности"},
                        {"section_id": "3.3", "title": "Тесты хранения"},
                        {"section_id": "3.4", "title": "Прочее"},
                    ],
                },
            },
            {
                "document_id": "doc-pz", "doc_role": "pz",
                "prepared_artifact": {
                    "document_id": "doc-pz", "doc_role": "pz",
                    "fragments": pz_fragments,
                    "sections": [
                        {"section_id": "2.1", "title": "Безопасность"},
                        {"section_id": "2.2", "title": "Архитектура"},
                        {"section_id": "2.3", "title": "Развёртывание"},
                    ],
                },
            },
        ],
        "options": dict(options),
    }


# ── Helpers ─────────────────────────────────────────────────────────────


def _ollama_reachable(model: str) -> Tuple[bool, str]:
    """Probe Ollama and confirm the model is loaded. Returns (ok, reason)."""
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate").rstrip(
        "/api/generate"
    )
    if not url:
        url = "http://localhost:11434"
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        names = {m.get("name") for m in (resp.json().get("models") or [])}
        if model not in names:
            return False, f"model {model!r} not in Ollama (have: {sorted(names)})"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _options_from_env(use_llm: bool, model: str) -> Dict[str, Any]:
    def _bool(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return v.lower() in {"1", "true", "yes", "on"}

    def _int(name: str, default: int) -> int:
        v = os.environ.get(name)
        return int(v) if v is not None else default

    def _float(name: str, default: float) -> float:
        v = os.environ.get(name)
        return float(v) if v is not None else default

    opts: Dict[str, Any] = {
        "min_retrieval_score": _float("CQUALITY_MIN_RETRIEVAL", 0.05),
        "initial_top_n": _int("CQUALITY_INITIAL_TOP_N", 10),
        "debug": _bool("CQUALITY_DEBUG_ENABLED", True),
        "debug_max_candidates": _int("CQUALITY_DEBUG_MAX_CANDIDATES", 5),
        "debug_include_discarded": _bool("CQUALITY_DEBUG_INCLUDE_DISCARDED", False),
        "enable_rule_verification": True,
        "enable_reranker": _bool("CQUALITY_RERANKER_ENABLED", False),
        "reranker_mode": os.environ.get("CQUALITY_RERANKER_MODE", "conditional"),
    }
    if use_llm:
        opts.update({
            "enable_llm_judge": True,
            "judge_backend": "ollama",
            "llm_model_name": model,
        })
    else:
        opts.update({
            "enable_llm_judge": False,
            "judge_backend": "disabled",
        })
    return opts


def _summarize(label: str, result, elapsed: float) -> Dict[str, Any]:
    rows = result.requirement_results
    by_status: Dict[str, int] = {}
    by_subcode: Dict[str, int] = {}
    selector_skips = 0
    selector_k_dist: Dict[int, int] = {}
    verifier_action_counts: Dict[str, int] = {}
    grounded = 0
    ungrounded = 0
    low_conf_rows = 0
    for r in rows:
        by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        sc = r.status_subcode or "(none)"
        by_subcode[sc] = by_subcode.get(sc, 0) + 1
        if r.low_confidence:
            low_conf_rows += 1
        if r.evidence_trace:
            sel = r.evidence_trace.get("selection") or {}
            if sel.get("skip_llm"):
                selector_skips += 1
            else:
                k = int(sel.get("selected_k") or 0)
                selector_k_dist[k] = selector_k_dist.get(k, 0) + 1
            for cand in r.evidence_trace.get("candidates", []):
                for a in cand.get("verifier_actions", []) or []:
                    verifier_action_counts[a] = verifier_action_counts.get(a, 0) + 1
                grp = cand.get("grounding_passed")
                if grp is True:
                    grounded += 1
                elif grp is False:
                    ungrounded += 1
    return {
        "label": label,
        "rows": len(rows),
        "elapsed_sec": round(elapsed, 2),
        "by_status": by_status,
        "by_subcode": by_subcode,
        "low_confidence_rows": low_conf_rows,
        "selector_skips": selector_skips,
        "selector_k_distribution": selector_k_dist,
        "verifier_actions": verifier_action_counts,
        "grounded_judgments": grounded,
        "ungrounded_judgments": ungrounded,
        "warnings": list(result.warnings or []),
    }


def _print_table(sections: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(" PR-K SMOKE — summary".upper())
    print("=" * 80)
    for s in sections:
        print(f"\n[{s['label']}]")
        print(f"  rows={s['rows']}  elapsed={s['elapsed_sec']}s")
        print(f"  status:    {s['by_status']}")
        print(f"  subcode:   {s['by_subcode']}")
        print(f"  selector:  skips={s['selector_skips']}  k-dist={s['selector_k_distribution']}")
        print(f"  verifier:  {s['verifier_actions']}")
        print(f"  grounding: passed={s['grounded_judgments']} failed={s['ungrounded_judgments']}")
        print(f"  low-conf rows: {s['low_confidence_rows']}")
        if s["warnings"]:
            print("  warnings:")
            for w in s["warnings"]:
                print(f"    • {w}")


def _print_per_row(label: str, result) -> None:
    print(f"\n--- per-row diagnostics [{label}] ---")
    for r in result.requirement_results:
        head = (r.req_text or "")[:90].replace("\n", " ")
        sel = (r.evidence_trace or {}).get("selection") or {}
        win = r.winning_candidate_id or "-"
        sc = r.status_subcode or "-"
        print(
            f"  · {r.target_doc_role} | type={r.requirement_type.value:<22} | "
            f"appl={r.applicability.value:<14} | level={r.coverage_requirement_level.value:<14} | "
            f"status={r.status.value:<8} subcode={sc:<24} "
            f"conf={r.final_confidence:.2f} win={win[:6]:<6} k={sel.get('selected_k', '-')}"
        )
        print(f"      req: {head}")
        if r.aggregation_reason:
            print(f"      decision: {r.aggregation_reason[:160]}")


def _maybe_dump_trace_one(result, target_role: str, status_filter: str | None = None) -> None:
    """Dump a single evidence_trace as JSON for visual inspection.
    Picks the first row matching `status_filter` (or any row if None) on
    the requested target role."""
    for r in result.requirement_results:
        if r.target_doc_role != target_role:
            continue
        if status_filter and r.status.value != status_filter:
            continue
        if not r.evidence_trace:
            continue
        print(f"\n--- evidence_trace sample [{target_role}/{r.status.value}] ---")
        print(json.dumps(r.evidence_trace, ensure_ascii=False, indent=2)[:2000])
        return


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    # Multi-model sweep support: comma-separated list, falls back to single
    # model if only CQUALITY_LLM_MODEL is set.
    models_env = os.environ.get("CQUALITY_LLM_MODELS")
    if models_env:
        models = [m.strip() for m in models_env.split(",") if m.strip()]
    else:
        single = os.environ.get("CQUALITY_LLM_MODEL", "qwen2.5:3b")
        models = [single]
    timeout = int(os.environ.get("CQUALITY_LLM_TIMEOUT", "60"))

    print(f"PR-K smoke: models={models}  timeout={timeout}s")

    sections: List[Dict[str, Any]] = []
    last_results: Dict[str, Any] = {}

    # ── Run 0: DisabledCoverageJudge baseline ──
    print("\n[baseline] DisabledCoverageJudge …")
    opts_base = _options_from_env(use_llm=False, model="(none)")
    cfg_base = CoverageConfig.from_options(opts_base)
    pipeline_base = CoverageAnalysisPipeline(config=cfg_base)
    t0 = time.perf_counter()
    res_base = pipeline_base.run(_build_fixture_request(opts_base))
    sections.append(_summarize("baseline / disabled judge", res_base, time.perf_counter() - t0))
    last_results["baseline"] = res_base

    # ── Per-model runs ──
    for i, model in enumerate(models, start=1):
        ok, reason = _ollama_reachable(model)
        print(f"\n[model {i}/{len(models)}] {model} — Ollama probe: ok={ok}  reason={reason}")
        if not ok:
            print(f"  LLM_UNAVAILABLE for {model}; skipping ({reason})")
            continue
        opts_llm = _options_from_env(use_llm=True, model=model)
        cfg_llm = CoverageConfig.from_options(opts_llm)
        cfg_llm.llm.timeout = timeout
        pipeline_llm = CoverageAnalysisPipeline(config=cfg_llm)
        t0 = time.perf_counter()
        res_llm = pipeline_llm.run(_build_fixture_request(opts_llm))
        sections.append(_summarize(f"Ollama / {model}", res_llm, time.perf_counter() - t0))
        last_results[model] = res_llm

    # ── Output ──
    _print_table(sections)
    _print_per_row("baseline", res_base)
    for model in models:
        if model in last_results:
            _print_per_row(f"ollama/{model}", last_results[model])

    # Dump one CONFLICT and one COVERED evidence_trace from the LAST LLM run
    # for visual inspection.
    last_llm = next(
        (last_results[m] for m in models if m in last_results), None
    )
    if last_llm is not None:
        _maybe_dump_trace_one(last_llm, "pmi", "CONFLICT")
        _maybe_dump_trace_one(last_llm, "pz", "COVERED")
    else:
        _maybe_dump_trace_one(res_base, "pmi", None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
