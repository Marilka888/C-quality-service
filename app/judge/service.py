from typing import Dict, List

from app.schemas import JudgeDecision, JudgeRequest, JudgeResponse
from app.judge.llm_judge import call_llm
from app.retrieval.retriever import CRetriever


# Model folder is expected to be inside the service root: ./model
retriever = CRetriever(model_path="./model")


def _retrieve_candidates(
    tz_text: str,
    pmi_pool: List[Dict],
    top_k: int,
    min_score: float,
) -> List[Dict]:
    """Return top-k PMI candidates as dicts {id, text, score}."""
    if not pmi_pool:
        return []

    texts = [c.get("text") or "" for c in pmi_pool]
    ranked = retriever.retrieve(tz_text, texts, k=top_k)  # [(idx, score)]

    out: List[Dict] = []
    for idx, score in ranked:
        if score < min_score:
            continue
        c = pmi_pool[idx]
        out.append({"id": c.get("id"), "text": c.get("text"), "score": float(score)})
    return out


def judge_pairs(req: JudgeRequest) -> JudgeResponse:
    decisions: List[JudgeDecision] = []

    # -----------------------------
    # Mode 1: explicit pairs
    # -----------------------------
    if req.pairs is not None:
        for pair in req.pairs:
            pmi_pool: List[Dict] = []
            if pair.pmi is not None:
                pmi_pool = [{"id": pair.pmi.id, "text": pair.pmi.text}]

            candidates = _retrieve_candidates(pair.tz.text, pmi_pool, req.top_k, req.min_score)

            if not candidates:
                llm_result = {
                    "verdict": "MISSING",
                    "explanation": "В ПМИ не найден кандидат-тест для проверки требования",
                }
                chosen_pmi_id = pair.pmi.id if pair.pmi else None
            else:
                best = candidates[0]
                llm_result = call_llm(pair.tz.text, best["text"])
                chosen_pmi_id = best.get("id")

            decisions.append(
                JudgeDecision(
                    tzId=pair.tz.id,
                    pmiId=chosen_pmi_id,
                    verdict=llm_result.get("verdict", "MISSING"),
                    explanation=llm_result.get("explanation", ""),
                )
            )

        return JudgeResponse(packageId=req.packageId, decisions=decisions)

    # -----------------------------
    # Mode 2: retrieval + judge
    # -----------------------------
    if req.tzRequirements is None or req.pmiTests is None:
        return JudgeResponse(packageId=req.packageId, decisions=[])

    pmi_pool = [t.dict() for t in req.pmiTests]

    for tz in req.tzRequirements:
        candidates = _retrieve_candidates(tz.text, pmi_pool, req.top_k, req.min_score)

        if not candidates:
            llm_result = {"verdict": "MISSING", "explanation": "В ПМИ не найден кандидат-тест"}
            chosen_pmi_id = None
        else:
            best = candidates[0]
            llm_result = call_llm(tz.text, best["text"])
            chosen_pmi_id = best.get("id")

        decisions.append(
            JudgeDecision(
                tzId=tz.id,
                pmiId=chosen_pmi_id,
                verdict=llm_result.get("verdict", "MISSING"),
                explanation=llm_result.get("explanation", ""),
            )
        )

    return JudgeResponse(packageId=req.packageId, decisions=decisions)
