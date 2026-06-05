"""Pydantic response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    country: str | None = None


class SeasonStatsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player_id: int
    competition_id: int
    sb_season_id: int
    primary_position: str | None
    matches_played: int
    minutes: float

    goals_per90: float
    assists_per90: float
    xg_per90: float
    progressive_passes_per90: float
    tackles_per90: float
    interceptions_per90: float
