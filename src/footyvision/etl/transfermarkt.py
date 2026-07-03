"""Load Transfermarkt market values (Kaggle: davidcariboo/player-scores) for 2015/16.

Two CSVs live in ./data (downloaded once via the Kaggle API):
  - player_valuations.csv : date-stamped historical values (pick the 2015/16 one)
  - players.csv           : names + dates of birth (for the age feature)
This is the real training target for the value predictor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LA_LIGA_TM = "ES1"
# A wide window (values are only updated a few times a year, so a tight season window
# misses players); we then keep each player's valuation closest to the season reference.
SEASON_START = "2014-07-01"
SEASON_END = "2017-06-30"
SEASON_REF = pd.Timestamp("2016-01-01")  # mid-season reference for "age" and value pick


def read_laliga_values_2016(data_dir: str | Path = "data") -> pd.DataFrame:
    """Return [name, value_eur, age] for La Liga players in the 2015/16 window."""
    data_dir = Path(data_dir)
    vals = pd.read_csv(data_dir / "player_valuations.csv", parse_dates=["date"])
    players = pd.read_csv(data_dir / "players.csv")

    window = vals[
        (vals["date"] >= SEASON_START)
        & (vals["date"] <= SEASON_END)
        & (vals["player_club_domestic_competition_id"] == LA_LIGA_TM)
    ].copy()

    # For each player keep the valuation closest to mid-season.
    window["gap"] = (window["date"] - SEASON_REF).abs()
    picked = window.sort_values("gap").drop_duplicates("player_id")

    meta = players[["player_id", "name", "date_of_birth"]].copy()
    merged = picked.merge(meta, on="player_id", how="left")

    dob = pd.to_datetime(merged["date_of_birth"], errors="coerce")
    merged["age"] = ((SEASON_REF - dob).dt.days / 365.25).round(1)

    out = merged[["name", "market_value_in_eur", "age"]].rename(
        columns={"market_value_in_eur": "value_eur"}
    )
    out = out.dropna(subset=["name"])
    return out.sort_values("value_eur", ascending=False).drop_duplicates("name")
