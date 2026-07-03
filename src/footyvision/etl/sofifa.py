"""Fetch FIFA player values from SoFIFA (via soccerdata) as a market-value proxy.

Used as the training target for the value predictor. FIFA 16 (version 160001) aligns with
our StatsBomb La Liga 2015/16 features. SoFIFA scrapes per player, so the first run is slow
but soccerdata caches every page (re-runs are instant and resumable).
"""
from __future__ import annotations

import warnings

import pandas as pd

from footyvision.ml.value import parse_value_eur

FIFA16_VERSION = 160001
LA_LIGA = "ESP-La Liga"

warnings.filterwarnings("ignore", message="credentials were not supplied")


def _find_col(columns: list, *keywords: str) -> str | None:
    for col in columns:
        name = str(col).lower()
        if any(k in name for k in keywords):
            return col
    return None


def read_laliga_values(version: int = FIFA16_VERSION) -> pd.DataFrame:
    """Return a DataFrame [name, value_eur, overall, potential, age] for La Liga players."""
    import soccerdata as sd

    sofifa = sd.SoFIFA(leagues=LA_LIGA, versions=version)
    ratings = sofifa.read_player_ratings().reset_index()

    cols = list(ratings.columns)
    name_col = _find_col(cols, "player") or "player"
    value_col = _find_col(cols, "value")
    overall_col = _find_col(cols, "overall")
    potential_col = _find_col(cols, "potential")
    age_col = _find_col(cols, "age")

    out = pd.DataFrame({"name": ratings[name_col].astype(str)})
    out["value_eur"] = ratings[value_col].map(parse_value_eur) if value_col else 0.0
    out["overall"] = pd.to_numeric(ratings[overall_col], errors="coerce") if overall_col else None
    out["potential"] = (
        pd.to_numeric(ratings[potential_col], errors="coerce") if potential_col else None
    )
    out["age"] = pd.to_numeric(ratings[age_col], errors="coerce") if age_col else None
    # One row per player (dedupe if a player appears in multiple teams/updates).
    return out.sort_values("value_eur", ascending=False).drop_duplicates("name")
