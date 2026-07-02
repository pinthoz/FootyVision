"""Feature matrix and position grouping for the similarity engine.

The comparable features are the per-90 rates (not raw totals), because players differ
in minutes played. Standardisation and similarity are always done *within a position
group* — comparing a goalkeeper's progressive passes to a winger's is meaningless.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from footyvision.db.models import METRIC_COLUMNS, Competition, Player, PlayerSeasonStats

# The per-90 columns that describe playing style — the similarity feature space.
PER90_FEATURES: tuple[str, ...] = tuple(f"{m}_per90" for m in METRIC_COLUMNS)


def position_group(position: str | None) -> str:
    """Map a granular StatsBomb position to a broad group for fair comparison.

    Order matters: 'Wing Back' contains both 'back' and 'wing' and must resolve to DEF.
    """
    if not position:
        return "Unknown"
    p = position.lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p:
        return "DEF"
    if "midfield" in p:
        return "MID"
    if "wing" in p or "forward" in p or "striker" in p:
        return "FWD"
    return "MID"


def load_feature_frame(
    session: Session,
    min_minutes: float | None = None,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> pd.DataFrame:
    """Load player-season rows (with names) into a DataFrame, one row per player-season.

    A `position_group` column is added. Filters are optional so the caller can scope
    the comparison pool (e.g. to one competition/season).
    """
    stmt = (
        select(
            PlayerSeasonStats.player_id,
            Player.name.label("name"),
            PlayerSeasonStats.competition_id,
            Competition.name.label("competition"),
            PlayerSeasonStats.sb_season_id,
            PlayerSeasonStats.primary_position,
            PlayerSeasonStats.matches_played,
            PlayerSeasonStats.minutes,
            *[getattr(PlayerSeasonStats, f) for f in PER90_FEATURES],
        )
        .join(Player, Player.id == PlayerSeasonStats.player_id)
        .join(Competition, Competition.id == PlayerSeasonStats.competition_id)
    )

    if min_minutes is not None:
        stmt = stmt.where(PlayerSeasonStats.minutes >= min_minutes)
    if competition_id is not None:
        stmt = stmt.where(PlayerSeasonStats.competition_id == competition_id)
    if season_id is not None:
        stmt = stmt.where(PlayerSeasonStats.sb_season_id == season_id)

    engine: Engine = session.get_bind()
    frame = pd.read_sql(stmt, engine)
    frame["position_group"] = frame["primary_position"].map(position_group)
    return frame
