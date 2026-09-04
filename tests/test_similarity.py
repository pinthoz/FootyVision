"""Unit tests for the similarity engine — synthetic frame, no DB required."""

from __future__ import annotations

import pandas as pd
import pytest

from footyvision.ml.features import PER90_FEATURES, position_group
from footyvision.ml.similarity import (
    find_similar,
    radar_percentiles,
    standardize_within_groups,
)


def test_position_group_mapping():
    assert position_group("Goalkeeper") == "GK"
    assert position_group("Right Center Back") == "DEF"
    assert position_group("Left Wing Back") == "DEF"  # 'back' wins over 'wing'
    assert position_group("Center Defensive Midfield") == "MID"
    assert position_group("Right Wing") == "FWD"
    assert position_group("Center Forward") == "FWD"
    assert position_group(None) == "Unknown"


def _row(pid: int, name: str, position: str, minutes: float, **feats) -> dict:
    d = {
        "player_id": pid,
        "name": name,
        "competition_id": 1,
        "sb_season_id": 1,
        "primary_position": position,
        "matches_played": 12,
        "minutes": minutes,
        "position_group": position_group(position),
    }
    d.update({f: 0.0 for f in PER90_FEATURES})
    d.update(feats)
    return d


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                1,
                "Striker A",
                "Center Forward",
                1000,
                goals_per90=0.8,
                xg_per90=0.7,
                shots_per90=3.0,
            ),
            _row(
                2,
                "Striker B",
                "Center Forward",
                1000,
                goals_per90=0.75,
                xg_per90=0.68,
                shots_per90=2.9,
            ),
            _row(
                3,
                "Striker C",
                "Center Forward",
                1000,
                goals_per90=0.05,
                xg_per90=0.05,
                tackles_per90=3.0,
            ),
            _row(4, "Defender D", "Center Back", 1000, tackles_per90=3.0, clearances_per90=5.0),
        ]
    )


def test_standardize_is_zero_mean_within_group():
    frame = _frame()
    z = standardize_within_groups(frame, list(PER90_FEATURES))
    fwd_idx = frame.index[frame["position_group"] == "FWD"]
    # Each feature should average ~0 across the forwards after z-scoring.
    assert z.loc[fwd_idx, "goals_per90"].mean() == pytest.approx(0.0, abs=1e-9)


def test_find_similar_stays_in_group_and_ranks_by_profile():
    frame = _frame()
    target, results = find_similar(frame, player_id=1, top_n=10)
    assert target["name"] == "Striker A"
    # The centre-back (different group) must never appear.
    assert 4 not in set(results["player_id"])
    # Striker B (near-identical profile) ranks above Striker C.
    order = list(results["player_id"])
    assert order.index(2) < order.index(3)
    assert results.iloc[0]["player_id"] == 2


def test_radar_percentiles_reflect_standing_in_group():
    frame = _frame()
    _, group, metrics = radar_percentiles(frame, player_id=1)
    assert group == "FWD"
    # Striker A is the top scorer among forwards -> top percentile.
    assert metrics["goals_per90"]["percentile"] == 100.0
    assert metrics["goals_per90"]["value"] == 0.8


def test_find_similar_returns_none_for_unknown_player():
    assert find_similar(_frame(), player_id=999) is None


# --- fragment seasons ------------------------------------------------------------------


def _seed_two_seasons(db_session):
    """A complete four-team season and a two-match fragment of another."""
    from footyvision.db.models import (
        METRIC_COLUMNS,
        Competition,
        Match,
        Player,
        PlayerSeasonStats,
        Team,
    )

    db_session.add(Competition(id=90, name="Whole League", country="Nowhere"))
    db_session.add(Competition(id=91, name="Fragment League", country="Nowhere"))
    for tid in (901, 902, 903, 904):
        db_session.add(Team(id=tid, name=f"Team {tid}"))
    db_session.flush()

    match_id = 5000
    for home in (901, 902, 903, 904):
        for away in (901, 902, 903, 904):
            if home != away:
                db_session.add(
                    Match(
                        id=match_id,
                        competition_id=90,
                        sb_season_id=1,
                        home_team_id=home,
                        away_team_id=away,
                    )
                )
                match_id += 1
    for home, away in ((901, 902), (903, 904)):
        db_session.add(
            Match(
                id=match_id, competition_id=91, sb_season_id=1, home_team_id=home, away_team_id=away
            )
        )
        match_id += 1

    pid = 500
    for competition_id in (90, 91):
        for _ in range(3):
            pid += 1
            db_session.add(Player(id=pid, name=f"P{pid}", country="Nowhere"))
            db_session.add(
                PlayerSeasonStats(
                    player_id=pid,
                    competition_id=competition_id,
                    sb_season_id=1,
                    primary_position="Center Forward",
                    matches_played=20,
                    minutes=1800.0,
                    **{f"{m}_per90": 1.0 for m in METRIC_COLUMNS},
                )
            )
    db_session.commit()


def test_fragment_seasons_are_kept_out_of_the_pool(db_session, monkeypatch):
    """A per-90 rate from two games of a twelve-game season is noise, and ranking it
    alongside real seasons moves everybody else's percentile.

    The threshold is set explicitly rather than relied on: it defaults to 0 (keep
    everything) and is an operator's choice, so the test must state the setting it is
    about instead of inheriting whatever the environment happens to carry.
    """
    from footyvision.config import Settings
    from footyvision.ml import features as features_module
    from footyvision.ml.features import load_feature_frame

    monkeypatch.setattr(features_module, "get_settings", lambda: Settings(min_season_coverage=0.9))
    _seed_two_seasons(db_session)

    pool = load_feature_frame(db_session, 600)

    assert "Whole League" in set(pool["competition"])
    assert "Fragment League" not in set(pool["competition"])


def test_fragments_can_be_asked_for_explicitly(db_session):
    """The data is filtered, never deleted — the coverage panel still reports it."""
    from footyvision.ml.features import load_feature_frame

    _seed_two_seasons(db_session)

    pool = load_feature_frame(db_session, 600, include_fragments=True)

    assert "Fragment League" in set(pool["competition"])


def test_a_season_with_no_matches_recorded_is_not_treated_as_a_fragment(db_session):
    """Coverage cannot be judged without fixtures, and excluding on an unknown would empty
    the pool of any instance holding aggregates without match rows."""
    from footyvision.db.quality import fragment_seasons
    from footyvision.ml.features import load_feature_frame

    # The shared fixture seeds season stats and no matches at all.
    assert fragment_seasons(db_session) == set()
    assert not load_feature_frame(db_session, 600).empty
