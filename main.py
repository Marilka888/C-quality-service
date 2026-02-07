
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import subprocess, json

app = FastAPI(title="C-quality LLM Judge")

class Candidate(BaseModel):
    id: str
    text: str

class JudgeRequest(BaseModel):
    tz: Candidate
    pmi: Optional[Candidate]

class JudgeResponse(BaseModel):
    verdict: str
    confidence: float
    explanation: str

OLLAMA_MODEL = "llama3"

def call_llm(prompt: str) -> str:
    p = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120
    )
    return p.stdout.decode("utf-8")

@app.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest):
    prompt = f"""
Ты — эксперт по проверке согласованности ТЗ и ПМИ.

ТЗ:
{req.tz.text}

ПМИ:
{req.pmi.text if req.pmi else "ОТСУТСТВУЕТ"}

Верни JSON строго:
{{"verdict":"COVERED|MISSING|CONFLICT|EXTRA","confidence":0-1,"explanation":"..."}}
"""

    raw = call_llm(prompt)

    try:
        data = json.loads(raw.strip().split("```")[-1])
        return JudgeResponse(**data)
    except Exception:
        return JudgeResponse(
            verdict="MISSING",
            confidence=0.5,
            explanation="Invalid LLM output"
        )
