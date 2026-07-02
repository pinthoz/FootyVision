"""Safe structured player search.

The LLM (or a client) produces a `PlayerQuery`; Pydantic validation is the security
boundary — only whitelisted fields and operators are accepted, so no arbitrary SQL or
attribute access is ever possible. Execution runs over the in-memory feature frame.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from footyvision.ml.features import PER90_FEATURES, load_feature_frame

# Fields a query is allowed to filter or sort on. Note: the dataset has no age/DOB,
# so age-based queries (e.g. "under 23") cannot be answered — the LLM is told this.
SEARCHABLE_FIELDS: frozenset[str] = frozenset(PER90_FEATURES) | {"minutes", "matches_played"}

Operator = Literal["gt", "gte", "lt", "lte", "eq"]
_OPS = {
    "gt": lambda s, v: s > v,
    "gte": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "lte": lambda s, v: s <= v,
    "eq": lambda s, v: s == v,
}


class Condition(BaseModel):
    field: str
    op: Operator
    value: float

    @field_validator("field")
    @classmethod
    def _known_field(cls, v: str) -> str:
        if v not in SEARCHABLE_FIELDS:
            raise ValueError(f"Unknown field '{v}'. Allowed: {sorted(SEARCHABLE_FIELDS)}")
        return v


class PlayerQuery(BaseModel):
    position_group: Literal["GK", "DEF", "MID", "FWD"] | None = None
    competition: str | None = Field(None, description="Substring matched against league name.")
    min_minutes: float | None = None
    conditions: list[Condition] = Field(default_factory=list)
    order_by: str | None = None
    order_desc: bool = True
    limit: int = Field(20, ge=1, le=100)

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        # LLMs often emit explicit nulls (e.g. "limit": null); drop them so field
        # defaults apply instead of failing validation on a non-optional field.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data

    @field_validator("order_by")
    @classmethod
    def _known_order(cls, v: str | None) -> str | None:
        if v is not None and v not in SEARCHABLE_FIELDS:
            raise ValueError(f"Unknown order_by '{v}'. Allowed: {sorted(SEARCHABLE_FIELDS)}")
        return v


def execute_query(session: Session, query: PlayerQuery) -> list[dict[str, Any]]:
    """Run a validated PlayerQuery against the feature frame; return result rows."""
    frame = load_feature_frame(session, query.min_minutes)
    if frame.empty:
        return []

    if query.competition:
        frame = frame[frame["competition"].str.contains(query.competition, case=False, na=False)]
    if query.position_group:
        frame = frame[frame["position_group"] == query.position_group]
    for cond in query.conditions:
        frame = frame[_OPS[cond.op](frame[cond.field], cond.value)]

    if query.order_by:
        frame = frame.sort_values(query.order_by, ascending=not query.order_desc)
    frame = frame.head(query.limit)

    # Return identity columns plus every field referenced by the query, so the caller
    # can see exactly why each row matched.
    referenced = {c.field for c in query.conditions}
    if query.order_by:
        referenced.add(query.order_by)
    referenced.update({"minutes", "matches_played"})

    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        rows.append(
            {
                "player_id": int(r["player_id"]),
                "name": r["name"],
                "competition": r["competition"],
                "primary_position": r["primary_position"],
                "position_group": r["position_group"],
                "stats": {f: round(float(r[f]), 3) for f in sorted(referenced)},
            }
        )
    return rows
