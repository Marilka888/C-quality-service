"""Generate labeling pairs CSV from packages/*.json.

Usage:
python scripts/generate_pairs.py --packages ./data/packages --out ./data/pairs_for_labeling.csv --top_k 5
"""

import argparse, json, re
from pathlib import Path
import pandas as pd

REQ_TRIGGERS = re.compile(r"(должен|должна|должны|необходимо|требуется|обеспечивать|запрещено|не\s+допускается|не\s+должен)",
                          flags=re.IGNORECASE | re.UNICODE)

def tokenize(s: str):
    return set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", s.lower()))

def score(req: str, cand: str) -> float:
    a, b = tokenize(req), tokenize(cand)
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a | b) or 1)

def extract_tz_reqs(tz_chunks):
    reqs, idx = [], 1
    for ch in tz_chunks:
        t = (ch.get("text") or "").strip()
        if t and REQ_TRIGGERS.search(t):
            reqs.append((f"TZ-RQ-{idx:05d}", t))
            idx += 1
    return reqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for fp in sorted(Path(args.packages).glob("*.json")):
        pkg = json.loads(fp.read_text(encoding="utf-8"))
        package_id = pkg["package_id"]
        docs = {d["doc_type"]: d for d in pkg.get("documents", [])}
        tz = docs.get("TZ")
        if not tz:
            continue
        tz_reqs = extract_tz_reqs(tz.get("chunks", []))

        for target_dt in ["PZ", "PMI"]:
            target = docs.get(target_dt)
            if not target:
                continue
            candidates = target.get("chunks", [])

            for tz_key, tz_text in tz_reqs:
                scored = [(score(tz_text, c.get("text", "")), c) for c in candidates]
                scored.sort(key=lambda x: x[0], reverse=True)
                for sc, c in scored[: args.top_k]:
                    rows.append({
                        "package_id": package_id,
                        "tz_req_key": tz_key,
                        "tz_text": tz_text,
                        "target_doc_type": target_dt,
                        "cand_chunk_id": c.get("chunk_id"),
                        "cand_text": c.get("text", ""),
                        "score_final": sc,
                        "label": ""  # fill: entails|neutral|contradicts
                    })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved: {args.out} (rows={len(df)})")

if __name__ == "__main__":
    main()
