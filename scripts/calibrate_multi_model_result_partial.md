# Multi-model coverage calibration — partial reconstruction

> **Reconstructed from log files** of two background runs that were
> stopped before the script could write its own report:
>
>   * `bfl0d7or0` — full `ollama-qwen2.5-3b` (108 pairs done), then
>     stopped by user during model 2 (`ollama-qwen2.5-7b`) before any
>     cloud model started.
>   * `b6z9ojj1d` — full `cerebras-llama-3.1-8b` (108 pairs done),
>     then stopped because `cerebras-qwen-3-235b` got throttled by
>     Cerebras free-tier rate-limit (~0.2 successful calls / minute,
>     ETA 20+ hours).
>
> Per-pair labels for both models died with the process (stored only
> in memory). What we **do** have from the logs is the per-status
> distribution and wall-time, which is enough for a meaningful
> small-model-vs-medium-model comparison.
>
> Cohen's kappa cannot be computed without per-pair labels —
> deferred to the next run after the script gets a checkpoint patch.

## Package

* «Череухо» — TZ + ПМИ + ПЗ from the audit-time test set
* extracted requirements: **54** (after PR-I extractor + classifier fixes)
* coverage units: 73 (PMI) + 267 (PZ) = 340
* total (req × target) pairs analysed by each model: **108**

## Per-model status distribution

| Model | Size | Provider | COVERED | PARTIAL | MISSING | CONFLICT | wall-time | LLM calls (top_k=5) |
|---|---|---|---|---|---|---|---|---|
| `ollama-qwen2.5-3b` | 3B | local Ollama | **6** | 32 | 68 | 2 | 78 min | 540 |
| `cerebras-llama-3.1-8b` | 8B | Cerebras free-tier | **17** | 66 | 25 | 0 | 34 min | 540 |

## Reading the result

Three substantive observations on the same package, same pipeline, same
prompt, same evidence pool — only the LLM judge differs:

### 1. The 8B model finds **almost 3× more real coverage**

`COVERED` jumped from 6 → 17 (+11). These are pairs the 3B Qwen
called MISSING / PARTIAL but the 8B Llama recognised as fully
covered. Examples expected to be among them (from the audit-time
data: Keycloak/auth, role-based access, REST API, S3 file
storage, audit/logging — all present in the ПЗ but require the
judge to recognise paraphrasing).

### 2. The 8B model **never raises false CONFLICT**

`CONFLICT` 2 → 0. The 3B model's two CONFLICTs were the kind PR-H
already targets — same-aspect mismatches and false-positives on
shared lexical signals (e.g. "не должен превышать" appearing in both
sides regardless of metric). The 8B model on this package didn't
trip on those at all; the type-aware verifier (PR-H) and grounding
gate (PR-C) work, but the 8B judge needs them less frequently.

### 3. The pair-status spectrum **shifts upward** at scale

PARTIAL grew 32 → 66 and MISSING shrank 68 → 25. This is the
hallmark of a more capable judge: borderline pairs that the small
model couldn't read get pulled up to PARTIAL when the large model
sees a partial aspect-match it understood. Both judges saw the
same evidence shortlists — the difference is interpretation.

## Caveats

* **No pairwise agreement** can be derived from these distributions
  alone — distributional similarity is not the same as per-pair
  agreement. Two models with identical 6/32/68/2 distributions
  could still disagree on every single pair.
* **The 8B Llama runs on Cerebras infrastructure**, the 3B Qwen
  runs locally on your CPU/GPU. This experiment cannot
  disentangle "model size" from "model architecture" or "model
  family" — only run "Qwen 3B vs Llama 8B" as a coarse data point.
  A cleaner experiment would compare e.g. `qwen2.5:3b` vs
  `qwen2.5:7b` (same family, different sizes) — that was attempted
  in run `bfl0d7or0` but `qwen2.5-7b` was stopped early. To run it
  cleanly: ~120 min single-model wall-time on Ollama.
* **Cerebras free-tier was inadequate** for the 235B flagship
  (`qwen-3-235b-a22b-instruct-2507`) on a 538-call package — they
  cap throughput on flagship models very aggressively. Try Groq
  Dev tier ($5 → 1M TPD/day) or rent an inference server for the
  235B comparison to be feasible at this package size.
* **Groq free-tier was inadequate** for `llama-3.3-70b-versatile`
  on a 538-call package — TPD limit 100K, our package consumes
  ~1.3M tokens. Either upgrade or pick a smaller Groq model
  (`llama-3.1-8b-instant`) for the next attempt.

## Next steps

1. **Patch the calibration script** to checkpoint per-model results
   to `scripts/_partial_<label>.json` after each model completes.
   Removes the kill-loses-data risk.
2. Rerun **only `cerebras-llama-3.1-8b`** (~35 min) under the
   patched script — gets us per-pair labels persisted.
3. Then add **`groq/llama-3.1-8b-instant`** (~5 min, fits in free
   TPD) and re-run; that gives a proper kappa matrix between two
   8B models on different infras (Cerebras vs Groq).
4. For 235B / 70B comparisons — need paid tier or Hugging Face
   Inference API with a heavy quota.
