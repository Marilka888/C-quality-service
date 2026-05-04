# Multi-model coverage calibration

Run the same package through several LLM judges (Ollama / Groq / Gemini /
Cerebras / OpenAI / Anthropic / …) and produce a cross-model comparison
report — Cohen's kappa, per-status counts, top disagreements. Useful for
research that asks: "does coverage quality scale with model size /
provider?".

## What you get

- **Per-model status distribution** (COVERED / PARTIAL / MISSING / CONFLICT)
- **Cohen's kappa pairwise matrix** (agreement between any two models)
- **Simple agreement %** matrix
- **Top-N disagreements** — pairs where models give the most varied verdicts
- **Per-model wall time + LLM unavailability** (rate-limits, parse errors)

Output: a single Markdown file in
`scripts/calibrate_multi_model_result.md`.

## Free-tier-only setup (no money required)

Default model lineup (5 models, all free, $0):

1. **`ollama-qwen2.5-3b`** — local 3B baseline (already on your machine).
2. **`ollama-qwen2.5-7b`** — local 7B baseline (already on your machine).
3. **`groq-llama-3.3-70b`** — production-grade 70B in the cloud, fast.
4. **`gemini-2.0-flash`** — Google free tier.
5. **`cerebras-llama-3.1-70b`** — fastest inference (~2000 tok/sec) free.

To enable a cloud model you only need to set its API key in env. Models
without a key are skipped — the script never breaks just because you
didn't sign up for one.

### Get the keys (free, ~5 minutes total)

| Provider | Sign up | Env var |
|---|---|---|
| Groq | https://console.groq.com → "API Keys" | `GROQ_API_KEY` |
| Google AI Studio | https://aistudio.google.com → "Get API key" | `GEMINI_API_KEY` |
| Cerebras | https://inference.cerebras.ai → "API keys" | `CEREBRAS_API_KEY` |

None of them require a credit card. Free tiers:
- **Groq**: ~30 req/min, 14400/day.
- **Gemini Flash**: 15 req/min, 1500/day.
- **Cerebras**: 30 req/min, 1M tokens/day.

For a 538-call package each cloud model takes ~18-36 minutes due to
rate limits — set up + go grab tea.

### Set the keys

PowerShell:
```powershell
$env:GROQ_API_KEY = "gsk_..."
$env:GEMINI_API_KEY = "AIza..."
$env:CEREBRAS_API_KEY = "csk-..."
```

Git Bash / Linux:
```bash
export GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=AIza...
export CEREBRAS_API_KEY=csk-...
```

## Run

From C-quality venv:

```powershell
cd C:\Users\Marilka\PycharmProjects\C-quality-service
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe scripts/calibrate_multi_model.py
```

This defaults to the «Череухо» package
(`Индивидуальное_ТЗ_Череухо_ВКР.docx` + ПМИ + ПЗ from
`C:/Users/Marilka/Pictures/`). To run on a custom package:

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_multi_model.py path\to\TZ.docx path\to\PMI.docx path\to\PZ.docx
```

The script prints progress per model, then writes
`scripts/calibrate_multi_model_result.md`.

## Adding paid models (when you have a key)

In `scripts/calibrate_multi_model.py` find the `MODELS` list and
uncomment / append entries:

```python
{"label": "gpt-4o-mini", "backend": "litellm",
 "model": "openai/gpt-4o-mini", "env": "OPENAI_API_KEY"},
{"label": "claude-3.5-haiku", "backend": "litellm",
 "model": "anthropic/claude-3-5-haiku-latest", "env": "ANTHROPIC_API_KEY"},
{"label": "gpt-4o", "backend": "litellm",
 "model": "openai/gpt-4o", "env": "OPENAI_API_KEY"},
```

Cost per package (~538 LLM calls):
- `gpt-4o-mini` — under $1
- `gpt-4o` — $5-7
- `claude-3-5-haiku` — $2-3
- `claude-3-5-sonnet` — $7-10

> **NB:** ChatGPT Plus / Pro subscriptions do **not** include API quota.
> Top up balance on https://platform.openai.com/billing separately.

## Adding any other LiteLLM-supported provider

[LiteLLM supports 100+ providers](https://docs.litellm.ai/docs/providers).
The pattern is always the same — model name in the form
`<provider>/<model>` plus the matching env var.

Examples:
- Together AI: `"together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"` + `TOGETHER_API_KEY`
- Mistral: `"mistral/mistral-small-latest"` + `MISTRAL_API_KEY`
- OpenRouter: `"openrouter/meta-llama/llama-3.2-3b-instruct:free"` + `OPENROUTER_API_KEY`

## Interpreting the report

**Cohen's kappa** — how much two models agree, corrected for chance:
- `< 0.4` — poor agreement
- `0.4 – 0.6` — moderate
- `> 0.6` — substantial

If a small model has kappa `< 0.3` against larger models on the same
package, that's evidence the small model is getting coverage wrong on
many pairs.

**Top disagreements** — pairs that look right to one model and wrong
to another. The fastest path to manual review of edge cases.

## Resuming partial runs

Each finished model gets dumped to `scripts/_partial_<label>.json`
immediately after its pipeline returns. If the calibration is killed
(or a single model gets stuck on rate limits like Cerebras flagship
qwen-3-235b on free tier — ~1 successful call per 5 min), simply
re-launch the script: it will **skip** any model that already has a
cached partial and only run the missing ones.

Force a re-run of all models, ignoring caches:

```powershell
$env:MULTI_MODEL_IGNORE_PARTIALS = "1"
```

Want to drop a single cached model and re-run only that one? Delete
its partial JSON:

```powershell
Remove-Item scripts\_partial_cerebras-llama-3.1-8b.json
```

## Reducing API calls

The pipeline issues `len(requirements) × len(target_roles) × top_k` LLM
calls. Free-tier providers (Groq 100K TPD, Cerebras throttled flagship
models) often can't absorb the full 540-call package. Three knobs to
trim, ordered by impact:

* `MULTI_MODEL_TOP_K=1` (default) — one LLM call per pair. Setting
  this to `5` quintuples calls; rarely changes verdicts because rank-1
  retrieval is usually the same fragment as the eventual winner.
* `MULTI_MODEL_TARGET_ROLES=pmi` — only TZ→PMI, skip TZ→PZ. Halves
  calls and keeps the most interesting axis for thesis purposes
  (PMI carries the test-coverage signal). Default `pmi,pz`.
* For very tight budgets — manually trim the source TZ to fewer
  requirements before calling prepare-service.

Combined `top_k=1 + pmi-only` brings the package down from **540 →
54 calls** per model. That fits comfortably in any free tier and lets
you run all five models in a single afternoon.

## Subsetting the model list

Two env knobs let you scope the run without editing the script:

* `MULTI_MODEL_BACKENDS_INCLUDE=litellm,ollama` — comma-separated
  list of backend types to include.
* The per-model `env` field — set its env var to be present, omit
  it to be skipped automatically.

So for cloud-only:
```powershell
$env:MULTI_MODEL_BACKENDS_INCLUDE = "litellm"
```

For local-only:
```powershell
$env:MULTI_MODEL_BACKENDS_INCLUDE = "ollama"
```

## Notes on reproducibility

- Temperature is fixed at `0.1` in `LiteLLMCoverageJudge` so runs are
  near-deterministic but not identical (LLMs sample differently
  across calls even at low T).
- The package is parsed once via prepare-service before the multi-model
  loop so all models see the same evidence pool — the only varying
  variable is the LLM verdict.
- Grounding gate (BUG-3 fix) and evidence floor (BUG-9 fix) apply to
  every judge backend equally — they're shared post-parse logic, not
  judge-specific.
