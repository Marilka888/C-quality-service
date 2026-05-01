# PR-C calibration on Cherevuyhho (live Ollama qwen2.5:3b)

Total (req × target) pairs: **24**  
LLM judge calls: **119**  
Low-confidence results (BUG-3 grounding + BUG-9 floor combined): **12**  
Ungrounded LLM demotions (BUG-3 alone): **0**  
Duplicate-pair dedup warnings (BUG-14): **0**

## Status distribution: legacy → new
Legacy = LLM raw label, no grounding gate, no evidence floor.  
New = current code: grounding-demoted ungrounded verdicts to IRRELEVANT, low_confidence flagged on results below evidence_floor=0.5.

| Status | Legacy | New |
|---|---|---|
| COVERED | 0 | 0 |
| PARTIAL | 2 | 2 |
| MISSING | 21 | 21 |
| CONFLICT | 1 | 1 |

## Detail dump
```json
{
  "total_pairs": 24,
  "judgments_made": 119,
  "by_status_legacy": {
    "MISSING": 21,
    "CONFLICT": 1,
    "PARTIAL": 2
  },
  "by_status_new": {
    "MISSING": 21,
    "CONFLICT": 1,
    "PARTIAL": 2
  },
  "low_confidence_results": 12,
  "ungrounded_demotes": 0,
  "dedup_warnings": [],
  "all_warnings_count": 1
}
```