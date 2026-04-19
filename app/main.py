from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="C Quality Service", version="2.0.0")
app.include_router(router)
