"""FastAPI backend.   uvicorn api:app --reload   ->  http://127.0.0.1:8000/docs"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

from recsys.service import RecService

state = {}


@asynccontextmanager
async def lifespan(app):
    state["svc"] = RecService()      # models are fitted once at startup
    yield


app = FastAPI(title="Movie Recommender API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=50),
              model: str = Query("Hybrid")):
    svc = state["svc"]
    if not 0 <= user_id < svc.nu:
        raise HTTPException(404, f"user_id must be in [0, {svc.nu - 1}]")
    if model not in svc.models:
        raise HTTPException(400, f"model must be one of {list(svc.models)}")
    return {"user_id": user_id, "model": model, "recommendations": svc.recommend(user_id, k, model)}


@app.get("/similar/{item_id}")
def similar(item_id: int, k: int = Query(10, ge=1, le=50)):
    svc = state["svc"]
    if not 0 <= item_id < svc.ni:
        raise HTTPException(404, "unknown item_id")
    return {"item": svc.title(item_id), "similar": svc.similar(item_id, k)}


@app.get("/cold-start")
def cold_start(genres: Optional[List[str]] = Query(None), k: int = Query(10, ge=1, le=50)):
    """Recommendations for a brand-new user who only told us their favourite genres."""
    svc = state["svc"]
    bad = [g for g in (genres or []) if g not in svc.genres]
    if bad:
        raise HTTPException(400, f"unknown genres {bad}; valid: {svc.genres}")
    return {"genres": genres or [], "recommendations": svc.cold_start(genres or [], k)}
