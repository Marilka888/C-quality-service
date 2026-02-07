from fastapi import APIRouter
from app.schemas import JudgeRequest, JudgeResponse
from app.judge.service import judge_pairs

router = APIRouter()

@router.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest):
    return judge_pairs(req)
