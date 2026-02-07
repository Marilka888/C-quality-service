from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re

from .dto import Document, Match, Defect, Stats, Candidate

REQ_TRIGGERS = [
    r"\bдолжен\b", r"\bдолжна\b", r"\bдолжны\b",
    r"\bнеобходимо\b", r"\bтребуется\b", r"\bобеспечивать\b",
    r"\bзапрещено\b", r"\bне\s+допускается\b", r"\bне\s+должен\b",
]
_req_re = re.compile("|".join(REQ_TRIGGERS), flags=re.IGNORECASE | re.UNICODE)

@dataclass
class ExtractedReq:
    req_key: str
    text: str

def extract_requirements_from_tz(tz_doc: Document) -> List[ExtractedReq]:
    reqs: List[ExtractedReq] = []
    idx = 1
    for ch in tz_doc.chunks:
        t = ch.text.strip()
        if t and _req_re.search(t):
            reqs.append(ExtractedReq(req_key=f"TZ-RQ-{idx:05d}", text=t))
            idx += 1
    return reqs

def _tokenize(s: str):
    return set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", s.lower()))

def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / (len(ta | tb) or 1)

def baseline_match(requirement: str, target_doc: Document, cfg) -> Tuple[str, float, List[Candidate]]:
    best_score = 0.0
    candidates: List[Candidate] = []

    for ch in target_doc.chunks:
        sc = _jaccard(requirement, ch.text)
        candidates.append(Candidate(chunk_id=ch.chunk_id, doc_type=target_doc.doc_type, score_final=sc, text=ch.text[:300]))
        best_score = max(best_score, sc)

    candidates.sort(key=lambda c: c.score_final, reverse=True)
    top = candidates[: cfg.top_k]

    if best_score >= cfg.emb_threshold_ok:
        return "OK", best_score, top
    if best_score >= cfg.emb_threshold_partial:
        return "PARTIAL", best_score, top
    return "MISSING", best_score, top

def check_c(documents: List[Document], cfg) -> Tuple[Stats, List[Match], List[Defect]]:
    tz = next((d for d in documents if d.doc_type == "TZ"), None)
    if tz is None:
        raise ValueError("TZ document is required")

    pz = next((d for d in documents if d.doc_type == "PZ"), None)
    pmi = next((d for d in documents if d.doc_type == "PMI"), None)

    reqs = extract_requirements_from_tz(tz)
    stats = Stats(tz_requirements=len(reqs))
    matches: List[Match] = []
    defects: List[Defect] = []

    for r in reqs:
        if pz is not None:
            status, score, top = baseline_match(r.text, pz, cfg)
            matches.append(Match(
                tz_req_key=r.req_key, tz_text=r.text,
                target_doc_type="PZ", status=status,
                matched_chunk_id=(top[0].chunk_id if top and status != "MISSING" else None),
                score_final=score, top_candidates=top
            ))
            if status == "MISSING":
                stats.missing_in_pz += 1
                defects.append(Defect(
                    defect_type="C1_MISSING", severity="MEDIUM",
                    tz_req_key=r.req_key, tz_text=r.text, target_doc_type="PZ",
                    message="Требование ТЗ не найдено в ПЗ (baseline).",
                    evidence={"top_candidates": [c.model_dump(exclude_none=True) for c in top]}
                ))

        if pmi is not None:
            status, score, top = baseline_match(r.text, pmi, cfg)
            matches.append(Match(
                tz_req_key=r.req_key, tz_text=r.text,
                target_doc_type="PMI", status=status,
                matched_chunk_id=(top[0].chunk_id if top and status != "MISSING" else None),
                score_final=score, top_candidates=top
            ))
            if status == "MISSING":
                stats.missing_in_pmi += 1
                defects.append(Defect(
                    defect_type="C1_MISSING", severity="MEDIUM",
                    tz_req_key=r.req_key, tz_text=r.text, target_doc_type="PMI",
                    message="Требование ТЗ не найдено в ПМИ (baseline).",
                    evidence={"top_candidates": [c.model_dump(exclude_none=True) for c in top]}
                ))

    return stats, matches, defects
