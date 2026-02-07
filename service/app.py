from __future__ import annotations
from fastapi import FastAPI, HTTPException
from .dto import CheckCRequest, CheckCResponse
from .pipeline import check_c

app = FastAPI(title="Quality-C Service", version="0.1.0")

@app.post("/check/c", response_model=CheckCResponse)
def check_c_endpoint(req: CheckCRequest):
    try:
        stats, matches, defects = check_c(req.documents, req.config)
        return CheckCResponse(
            package_id=req.package_id,
            stats=stats,
            matches=matches,
            defects=defects,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
