"""How complete each loaded competition-season actually is.

The StatsBomb open dataset mixes whole league seasons with fragments carrying the same
competition and season names — La Liga 2018/19 is only Barcelona's matches, and the
Bundesliga 2015/16 in this database is 34 games of a 306-game season. A match count alone
does not tell them apart, so each season is measured against the number of games a full
double round-robin of its teams would have.

This lives here rather than in the coverage router because two very different callers need
it: the router reports it, and `load_feature_frame` uses it to keep fragments out of the
pool that percentiles, similarity and every model are computed over.
"""

from __future__ import annotations

from sqlalchemy import func, select, union
from sqlalchemy.orm import Session

from footyvision.db.models import Match

# Below this share of a full double round-robin a season is a fragment, not a season.
COMPLETE_THRESHOLD = 0.9


def season_coverage(session: Session) -> dict[tuple[int, int], float]:
    """Matches held over a full double round-robin, per (competition_id, season_id).

    Capped at 1.0: play-off and final-stage formats exceed a plain round-robin, and a
    season being *longer* than the baseline says nothing about it being incomplete.
    """
    match_counts = {
        (r.competition_id, r.sb_season_id): r.n
        for r in session.execute(
            select(
                Match.competition_id,
                Match.sb_season_id,
                func.count().label("n"),
            ).group_by(Match.competition_id, Match.sb_season_id)
        )
    }

    # UNION (not UNION ALL) dedupes, so counting its rows counts distinct teams.
    sides = union(
        select(
            Match.competition_id.label("cid"),
            Match.sb_season_id.label("sid"),
            Match.home_team_id.label("tid"),
        ),
        select(Match.competition_id, Match.sb_season_id, Match.away_team_id),
    ).subquery()
    team_counts = {
        (r.cid, r.sid): r.n
        for r in session.execute(
            select(sides.c.cid, sides.c.sid, func.count(sides.c.tid).label("n")).group_by(
                sides.c.cid, sides.c.sid
            )
        )
    }

    coverage: dict[tuple[int, int], float] = {}
    for key, matches in match_counts.items():
        teams = team_counts.get(key, 0)
        expected = teams * (teams - 1) if teams > 1 else 0
        coverage[key] = min(matches / expected, 1.0) if expected else 0.0
    return coverage


def fragment_seasons(
    session: Session, threshold: float = COMPLETE_THRESHOLD
) -> set[tuple[int, int]]:
    """Competition-seasons too incomplete to compare players across.

    A season with no matches recorded at all is *not* listed. Coverage cannot be judged
    without fixtures, and excluding on an unknown would empty the pool of any instance
    holding aggregates without match rows — including every test fixture.
    """
    return {key: None for key, ratio in season_coverage(session).items() if ratio < threshold}.keys()
