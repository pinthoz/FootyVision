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


class TargetOut(BaseModel):
    player_id: int
    name: str
    primary_position: str | None
    position_group: str
    minutes: float


class SimilarPlayerOut(BaseModel):
    player_id: int
    name: str
    primary_position: str | None
    position_group: str
    competition_id: int
    sb_season_id: int
    minutes: float
    similarity: float
    # a little context so results are readable without a second call
    xg_per90: float
    progressive_passes_per90: float
    tackles_per90: float
    dribbles_per90: float


class SimilarResponse(BaseModel):
    target: TargetOut
    count: int
    results: list[SimilarPlayerOut]


class RadarMetric(BaseModel):
    value: float
    percentile: float


class RadarResponse(BaseModel):
    player_id: int
    name: str
    position_group: str
    minutes: float
    metrics: dict[str, RadarMetric]


class ReportResponse(BaseModel):
    player_id: int
    name: str
    report: str
    context: dict


class ReportContextResponse(BaseModel):
    player_id: int
    context: dict


class NLSearchRequest(BaseModel):
    query: str


class SearchResultRow(BaseModel):
    player_id: int
    name: str
    competition: str | None
    primary_position: str | None
    position_group: str
    stats: dict[str, float]


class SearchResponse(BaseModel):
    interpreted: dict
    count: int
    results: list[SearchResultRow]


class ScoreResponse(BaseModel):
    player_id: int
    name: str
    position_group: str
    performance_score: float
    breakdown: list[dict]
    style_profile: dict[str, float]


class RankingRow(BaseModel):
    player_id: int
    name: str
    competition: str | None
    position_group: str
    primary_position: str | None
    performance_score: float


class RankingsResponse(BaseModel):
    count: int
    results: list[RankingRow]


class ModelInfoResponse(BaseModel):
    task: str
    classes: list[str]
    test_accuracy: float
    n_train: int
    n_test: int


class AssistantRequest(BaseModel):
    question: str
    k: int = 6


class AssistantSource(BaseModel):
    player_id: int
    name: str
    score: float


class AssistantResponse(BaseModel):
    answer: str
    sources: list[AssistantSource]
