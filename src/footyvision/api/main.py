from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from footyvision import __version__
from footyvision.api.routers import (
    assistant,
    coverage,
    health,
    metrics,
    players,
    reports,
    search,
    similarity,
    talent,
)
from footyvision.config import get_settings

app = FastAPI(
    title="FootyVision API",
    version=__version__,
    description="AI football scouting platform — similarity, talent scoring and LLM reports.",
)

# Browsers may only call this API from the configured frontends. This is hygiene, not a
# security boundary: it constrains other *sites*, not scripts. The rate limiter in
# api/limits.py is what actually protects the LLM budget.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(health.router)
app.include_router(coverage.router)
app.include_router(players.router)
app.include_router(similarity.router)
app.include_router(reports.router)
app.include_router(search.router)
app.include_router(talent.router)
app.include_router(metrics.router)
app.include_router(assistant.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"name": "FootyVision API", "version": __version__, "docs": "/docs"}
