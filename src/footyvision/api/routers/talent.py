from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import (
    ModelInfoResponse,
    RankingsResponse,
    RoleModelInfo,
    ScoreResponse,
)
from footyvision.config import get_settings
from footyvision.db.base import get_session
from footyvision.ml.features import load_feature_frame
from footyvision.ml.scoring import performance_score, rank_players
from footyvision.ml.talent import (
    get_cached_exact_model,
    get_cached_importance,
    get_cached_model,
    get_cached_role_model,
    style_profile,
)

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
    roles = style_profile(get_cached_role_model(frame), frame, player_id) or {}
    best_role, confidence = (None, None)
    if roles:
        best_role, confidence = max(roles.items(), key=lambda kv: kv[1])
    exact = style_profile(get_cached_exact_model(frame), frame, player_id) or {}
    shortlist = [name for name, _ in sorted(exact.items(), key=lambda kv: -kv[1])[:3]]
    return ScoreResponse(
        **score,
        style_profile=profile,
        predicted_role=best_role,
        role_confidence=confidence,
        role_profile=roles,
        predicted_position=shortlist[0] if shortlist else None,
        position_shortlist=shortlist,
    )


@router.get("/rankings", response_model=RankingsResponse)
def rankings(
    position_group: str | None = Query(None, description="GK / DEF / MID / FWD"),
    top_n: int = Query(20, ge=1, le=100),
    gender: str | None = Query(None, description="all / female / male"),
    min_minutes: float | None = Query(None),
    session: Session = Depends(get_session),
) -> RankingsResponse:
    """Leaderboard by Performance Score, optionally within one position group and/or gender."""
    frame = _frame(session, min_minutes)
    results = rank_players(frame, position_group, top_n, gender=gender)
    return RankingsResponse(count=len(results), results=results)


@router.get("/talent/model-info", response_model=ModelInfoResponse)
def model_info(
    min_minutes: float | None = Query(None), session: Session = Depends(get_session)
) -> ModelInfoResponse:
    """Evaluation of the position classifiers (honest held-out accuracy, both grains)."""
    frame = _frame(session, min_minutes)
    tm = get_cached_model(frame)
    rm = get_cached_role_model(frame)
    em = get_cached_exact_model(frame)
    return ModelInfoResponse(
        task="position-group classification",
        classes=tm.classes,
        test_accuracy=round(tm.test_accuracy, 3),
        n_train=tm.n_train,
        n_test=tm.n_test,
        features=tm.features,
        top_features=get_cached_importance(tm, frame),
        role_model=RoleModelInfo(
            classes=rm.classes,
            test_accuracy=round(rm.test_accuracy, 3),
            n_train=rm.n_train,
            n_test=rm.n_test,
        ),
        exact_model=RoleModelInfo(
            classes=em.classes,
            test_accuracy=round(em.test_accuracy, 3),
            n_train=em.n_train,
            n_test=em.n_test,
            top3_accuracy=(
                round(em.top3_accuracy, 3) if em.top3_accuracy is not None else None
            ),
        ),
    )
