from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from footyvision.api.schemas import PlayerOut, SeasonStatsOut
from footyvision.db.base import get_session
from footyvision.db.models import Player, PlayerSeasonStats

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=list[PlayerOut])
def list_players(
    search: str | None = Query(None, description="Case-insensitive name or nickname filter"),
    with_stats: bool = Query(
        False,
        description=(
            "Only players that have a season aggregate. The players table also holds "
            "everyone who merely appeared in a match, and those have no radar or score."
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> list[Player]:
    stmt = select(Player)
    if search:
        # Names here are the full legal form ("Lionel Andrés Messi Cuccittini"), so a
        # search for the name anyone would actually type has to reach the nickname too.
        pattern = f"%{search}%"
        stmt = stmt.where(or_(Player.name.ilike(pattern), Player.nickname.ilike(pattern)))
    if with_stats:
        # A scalar subquery rather than a join: a player can hold several season rows,
        # and joining would return him once per season.
        top_minutes = (
            select(func.max(PlayerSeasonStats.minutes))
            .where(PlayerSeasonStats.player_id == Player.id)
            .scalar_subquery()
        )
        stmt = stmt.where(
            select(PlayerSeasonStats.id).where(PlayerSeasonStats.player_id == Player.id).exists()
        )
        # Most-played first: `limit` truncates the list, so the cut has to fall on the
        # least relevant players. Alphabetical order would just show everyone up to "C".
        stmt = stmt.order_by(top_minutes.desc(), Player.name)
    else:
        stmt = stmt.order_by(Player.name)
    return list(session.scalars(stmt.limit(limit)))


@router.get("/{player_id}", response_model=PlayerOut)
def get_player(player_id: int, session: Session = Depends(get_session)) -> Player:
    player = session.get(Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return player


@router.get("/{player_id}/seasons", response_model=list[SeasonStatsOut])
def player_seasons(
    player_id: int, session: Session = Depends(get_session)
) -> list[PlayerSeasonStats]:
    stmt = select(PlayerSeasonStats).where(PlayerSeasonStats.player_id == player_id)
    return list(session.scalars(stmt))
