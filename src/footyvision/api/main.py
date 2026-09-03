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

app = FastAPI(
    title="FootyVision API",
    version=__version__,
    description="AI football scouting platform — similarity, talent scoring and LLM reports.",
)

# Open CORS for local frontend development (tighten before any real deployment).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
