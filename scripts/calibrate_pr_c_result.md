# PR-C calibration on Cherevuyhho (live Ollama qwen2.5:3b)

Total (req × target) pairs: **108**  
LLM judge calls: **538**  
Low-confidence results (BUG-3 grounding + BUG-9 floor combined): **83**  
Ungrounded LLM demotions (BUG-3 alone): **12**  
Duplicate-pair dedup warnings (BUG-14): **0**

## Status distribution: legacy → new
Legacy = LLM raw label, no grounding gate, no evidence floor.  
New = current code: grounding-demoted ungrounded verdicts to IRRELEVANT, low_confidence flagged on results below evidence_floor=0.5.

| Status | Legacy | New |
|---|---|---|
| COVERED | 6 | 5 |
| PARTIAL | 33 | 33 |
| MISSING | 65 | 68 |
| CONFLICT | 4 | 2 |

## Detail dump
```json
{
  "total_pairs": 108,
  "judgments_made": 538,
  "by_status_legacy": {
    "MISSING": 65,
    "PARTIAL": 33,
    "CONFLICT": 4,
    "COVERED": 6
  },
  "by_status_new": {
    "MISSING": 68,
    "PARTIAL": 33,
    "CONFLICT": 2,
    "COVERED": 5
  },
  "low_confidence_results": 83,
  "ungrounded_demotes": 12,
  "dedup_warnings": [],
  "all_warnings_count": 1
}
```