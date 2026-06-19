from __future__ import annotations
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.engine import GolgiEngine

app = FastAPI(title="Golgi")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/", StaticFiles(directory="static", html=True), name="static")

engine = GolgiEngine(exa_key=os.getenv("EXA_API_KEY"))

@app.get("/api/health")
def health(): return {"ok": True}

@app.get("/api/search")
def api_search(q: str = Query(...), max: int = 8, since: str | None = None):
    return engine.search(q, num=max, since=since)