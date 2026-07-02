"""Unit tests for the Performance Score and the XGBoost position classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footyvision.ml.features import PER90_FEATURES, position_group
from footyvision.ml.scoring import performance_score, rank_players, score_frame
from footyvision.ml.talent import style_profile, train_position_classifier

_SPECS = {
    "Center Forward": {"xg_per90": 0.6, "goals_per90": 0.5, "shots_per90": 3.0},
    "Center Back": {"tackles_per90": 3.0, "clearances_per90": 5.0, "interceptions_per90": 2.5},
    "Center Midfield": {"progressive_passes_per90": 9.0, "passes_completed_per90": 45.0},
    "Goalkeeper": {"passes_completed_per90": 22.0, "clearances_per90": 2.0},
}


def _synthetic(n_per_class: int = 14, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pid = 1
    for position, base in _SPECS.items():
        for _ in range(n_per_class):
            feats = {f: 0.0 for f in PER90_FEATURES}
            for k, v in base.items():
                feats[k] = max(0.0, v + rng.normal(0, 0.12 * v))
            rows.append({
                "player_id": pid, "name": f"P{pid}", "competition": "Test",
                "competition_id": 1, "sb_season_id": 1, "primary_position": position,
                "matches_played": 20, "minutes": 1800,
                "position_group": position_group(position), **feats,
            })
            pid += 1
    return pd.DataFrame(rows)


def test_score_frame_bounds_and_column():
    scored = score_frame(_synthetic())
    assert "performance_score" in scored.columns
    assert scored["performance_score"].between(0, 100).all()


def test_performance_score_breakdown_sums_to_score():
    frame = _synthetic()
    fwd_id = int(frame[frame["position_group"] == "FWD"].iloc[0]["player_id"])
    res = performance_score(frame, fwd_id)
    assert res["position_group"] == "FWD"
    assert 0 <= res["performance_score"] <= 100
    total = sum(b["contribution"] for b in res["breakdown"])
    assert abs(total - res["performance_score"]) < 0.5  # rounding tolerance


def test_rank_players_within_group_sorted():
    ranked = rank_players(_synthetic(), position_group="FWD", top_n=5)
    scores = [r["performance_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(r["position_group"] == "FWD" for r in ranked)


def test_classifier_learns_and_profiles_style():
    frame = _synthetic(n_per_class=16)
    tm = train_position_classifier(frame, seed=42)
    # Well-separated synthetic classes should be classified well.
    assert tm.test_accuracy >= 0.7
    assert set(tm.classes) == {"GK", "DEF", "MID", "FWD"}

    fwd_id = int(frame[frame["position_group"] == "FWD"].iloc[0]["player_id"])
    profile = style_profile(tm, frame, fwd_id)
    assert abs(sum(profile.values()) - 1.0) < 1e-2  # probs are rounded to 3 decimals
    assert max(profile, key=profile.get) == "FWD"
