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
