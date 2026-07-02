from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import (
    ModelInfoResponse,
    RankingsResponse,
    ScoreResponse,
)
from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.ml.features import load_feature_frame
from footyvision.ml.scoring import performance_score, rank_players
from footyvision.ml.talent import get_cached_model, style_profile

router = APIRouter(tags=["talent"])


def _frame(session: Session, min_minutes: float | None):
    mm = get_settings().min_minutes if min_minutes is None else min_minutes
    return load_feature_frame(session, mm)


@router.get("/players/{player_id}/score", response_model=ScoreResponse)
def player_score(
    player_id: int,
    min_minutes: float | None = Query(None),
    session: Session = Depends(get_session),
) -> ScoreResponse:
    """Position-aware Performance Score (0-100) + the model's style profile."""
    frame = _frame(session, min_minutes)
    score = performance_score(frame, player_id)
    if score is None:
        raise HTTPException(status_code=404, detail="Player not found in the feature pool.")
    profile = style_profile(get_cached_model(frame), frame, player_id) or {}
    return ScoreResponse(**score, style_profile=profile)


@router.get("/rankings", response_model=RankingsResponse)
def rankings(
    position_group: str | None = Query(None, description="GK / DEF / MID / FWD"),
    top_n: int = Query(20, ge=1, le=100),
    min_minutes: float | None = Query(None),
    session: Session = Depends(get_session),
) -> RankingsResponse:
    """Leaderboard by Performance Score, optionally within one position group."""
    frame = _frame(session, min_minutes)
    results = rank_players(frame, position_group, top_n)
    return RankingsResponse(count=len(results), results=results)


@router.get("/talent/model-info", response_model=ModelInfoResponse)
def model_info(
    min_minutes: float | None = Query(None), session: Session = Depends(get_session)
) -> ModelInfoResponse:
    """Evaluation of the XGBoost position classifier (honest held-out accuracy)."""
    tm = get_cached_model(_frame(session, min_minutes))
    return ModelInfoResponse(
        task="position-group classification",
        classes=tm.classes,
        test_accuracy=round(tm.test_accuracy, 3),
        n_train=tm.n_train,
        n_test=tm.n_test,
    )
