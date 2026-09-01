"""Database schema for the FootyVision scouting platform.

IDs that originate from StatsBomb (competition_id, season_id, team_id, player_id,
match_id) are reused as primary keys so the ETL is idempotent (upserts by natural key).
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from footyvision.db.base import Base

# Counting metrics aggregated from StatsBomb events. Kept as a tuple so the ETL and
# similarity layers can iterate the canonical metric list without hardcoding it.
METRIC_COLUMNS: tuple[str, ...] = (
    "goals",
    "assists",
    "shots",
    "xg",
    "passes",
    "passes_completed",
    "progressive_passes",
    "dribbles",
    "dribbles_completed",
    "carries",
    "progressive_carries",
    "tackles",
    "interceptions",
    "blocks",
    "clearances",
    "ball_recoveries",
    "pressures",
)


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb competition_id
    name: Mapped[str] = mapped_column(String(120))
    country: Mapped[str | None] = mapped_column(String(120))

    seasons: Mapped[list[Season]] = relationship(back_populates="competition")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("competition_id", "sb_season_id", name="uq_comp_season"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    sb_season_id: Mapped[int] = mapped_column(Integer)  # StatsBomb season_id
    name: Mapped[str] = mapped_column(String(60))  # e.g. "2015/2016"

    competition: Mapped[Competition] = relationship(back_populates="seasons")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb team_id
    name: Mapped[str] = mapped_column(String(120))


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb player_id
    name: Mapped[str] = mapped_column(String(120), index=True)
    country: Mapped[str | None] = mapped_column(String(120))


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)  # StatsBomb match_id
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    sb_season_id: Mapped[int] = mapped_column(Integer, index=True)
    match_date: Mapped[date | None] = mapped_column(Date)
    home_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))


class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"
    __table_args__ = (UniqueConstraint("match_id", "player_id", name="uq_match_player"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    position: Mapped[str | None] = mapped_column(String(40))
    minutes: Mapped[float] = mapped_column(Float, default=0.0)

    goals: Mapped[float] = mapped_column(Float, default=0.0)
    assists: Mapped[float] = mapped_column(Float, default=0.0)
    shots: Mapped[float] = mapped_column(Float, default=0.0)
    xg: Mapped[float] = mapped_column(Float, default=0.0)
    passes: Mapped[float] = mapped_column(Float, default=0.0)
    passes_completed: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_passes: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_completed: Mapped[float] = mapped_column(Float, default=0.0)
    carries: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_carries: Mapped[float] = mapped_column(Float, default=0.0)
    tackles: Mapped[float] = mapped_column(Float, default=0.0)
    interceptions: Mapped[float] = mapped_column(Float, default=0.0)
    blocks: Mapped[float] = mapped_column(Float, default=0.0)
    clearances: Mapped[float] = mapped_column(Float, default=0.0)
    ball_recoveries: Mapped[float] = mapped_column(Float, default=0.0)
    pressures: Mapped[float] = mapped_column(Float, default=0.0)


class PlayerSeasonStats(Base):
    """Per player+season totals plus per-90 rates — the table the ML/similarity layer reads."""

    __tablename__ = "player_season_stats"
    __table_args__ = (
        UniqueConstraint("player_id", "competition_id", "sb_season_id", name="uq_player_season"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    sb_season_id: Mapped[int] = mapped_column(Integer, index=True)
    primary_position: Mapped[str | None] = mapped_column(String(40))
    matches_played: Mapped[int] = mapped_column(Integer, default=0)
    minutes: Mapped[float] = mapped_column(Float, default=0.0)

    # Totals
    goals: Mapped[float] = mapped_column(Float, default=0.0)
    assists: Mapped[float] = mapped_column(Float, default=0.0)
    shots: Mapped[float] = mapped_column(Float, default=0.0)
    xg: Mapped[float] = mapped_column(Float, default=0.0)
    passes: Mapped[float] = mapped_column(Float, default=0.0)
    passes_completed: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_passes: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_completed: Mapped[float] = mapped_column(Float, default=0.0)
    carries: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_carries: Mapped[float] = mapped_column(Float, default=0.0)
    tackles: Mapped[float] = mapped_column(Float, default=0.0)
    interceptions: Mapped[float] = mapped_column(Float, default=0.0)
    blocks: Mapped[float] = mapped_column(Float, default=0.0)
    clearances: Mapped[float] = mapped_column(Float, default=0.0)
    ball_recoveries: Mapped[float] = mapped_column(Float, default=0.0)
    pressures: Mapped[float] = mapped_column(Float, default=0.0)

    # Per-90 rates (the comparable features)
    goals_per90: Mapped[float] = mapped_column(Float, default=0.0)
    assists_per90: Mapped[float] = mapped_column(Float, default=0.0)
    shots_per90: Mapped[float] = mapped_column(Float, default=0.0)
    xg_per90: Mapped[float] = mapped_column(Float, default=0.0)
    passes_per90: Mapped[float] = mapped_column(Float, default=0.0)
    passes_completed_per90: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_passes_per90: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_per90: Mapped[float] = mapped_column(Float, default=0.0)
    dribbles_completed_per90: Mapped[float] = mapped_column(Float, default=0.0)
    carries_per90: Mapped[float] = mapped_column(Float, default=0.0)
    progressive_carries_per90: Mapped[float] = mapped_column(Float, default=0.0)
    tackles_per90: Mapped[float] = mapped_column(Float, default=0.0)
    interceptions_per90: Mapped[float] = mapped_column(Float, default=0.0)
    blocks_per90: Mapped[float] = mapped_column(Float, default=0.0)
    clearances_per90: Mapped[float] = mapped_column(Float, default=0.0)
    ball_recoveries_per90: Mapped[float] = mapped_column(Float, default=0.0)
    pressures_per90: Mapped[float] = mapped_column(Float, default=0.0)
