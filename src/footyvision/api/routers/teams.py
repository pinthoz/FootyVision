"""Team attack and defence ratings, the one thing this project had no notion of.

Every other endpoint describes a player, and a player's per-90 numbers are shaped by the
side around him: a striker in a weak attack sees fewer chances than one in a strong one,
and nothing here could previously say which he was. These are Poisson coefficients on a
log scale, fitted across a whole season — 0 is an average side, a positive attack scores
more than average and a *negative* defence concedes less.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import TeamStrengthOut, TeamStrengthResponse
from footyvision.db.base import get_session
from footyvision.db.models import Competition, Team
from footyvision.ml.matches import load_matches, team_strengths

router = APIRouter(tags=["teams"])

# Fitting the Poisson model walks every match in the database, which is fast but not free,
# and the answer only moves when a season is loaded. Memoised per process, like the talent
# models, which is also why the API has to be restarted after an import.
_CACHE: dict[str, TeamStrengthResponse] = {}


@router.get("/teams/strength", response_model=TeamStrengthResponse)
def strength(
    session: Session = Depends(get_session),
    competition_id: int | None = Query(None, description="Restrict to one competition."),
) -> TeamStrengthResponse:
    """Attack and defence ratings per team, strongest attack first."""
    key = str(competition_id)
    if key in _CACHE:
        return _CACHE[key]

    matches = load_matches(session)
    if competition_id is not None:
        matches = matches[matches["competition_id"] == competition_id]

    info: dict[str, float] = {}
    if matches.empty:
        return TeamStrengthResponse(home_advantage=0.0, matches=0, teams=[])

    ratings = team_strengths(matches, info)
    names = {t.id: t.name for t in session.query(Team).all()}
    competitions = {c.id: c.name for c in session.query(Competition).all()}
    # A team belongs to whichever competition it actually played in, read off the fixtures
    # rather than stored: the teams table has no competition of its own.
    played_in = {
        int(row.home_team_id): int(row.competition_id) for row in matches.itertuples(index=False)
    } | {int(row.away_team_id): int(row.competition_id) for row in matches.itertuples(index=False)}

    response = TeamStrengthResponse(
        home_advantage=round(info.get("home_advantage", 0.0), 4),
        matches=int(len(matches)),
        teams=[
            TeamStrengthOut(
                team_id=rating.team_id,
                name=names.get(rating.team_id, str(rating.team_id)),
                competition=competitions.get(played_in.get(rating.team_id, -1)),
                attack=round(rating.attack, 4),
                defence=round(rating.defence, 4),
                matches=rating.matches,
            )
            for rating in sorted(ratings, key=lambda r: -r.attack)
        ],
    )
    _CACHE[key] = response
    return response
