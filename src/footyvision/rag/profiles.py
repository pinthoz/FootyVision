"""Build a short English text profile per player — the documents the RAG store embeds.

The text is natural-language *playing-style* prose (not a raw metric dump): the local
embedding model (nomic-embed-text) discriminates far better on descriptive language that
resembles how scouts phrase queries. English on purpose (nomic is English-first); the
assistant's final answer can still be Portuguese.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from footyvision.ml.features import PER90_FEATURES
from footyvision.ml.scoring import score_frame

# Map each per-90 metric to a scout-style phrase, so a high percentile reads naturally.
STYLE_PHRASES: dict[str, str] = {
    "goals_per90": "scoring goals",
    "xg_per90": "getting into dangerous scoring positions",
    "shots_per90": "shooting frequently",
    "assists_per90": "creating assists for teammates",
    "passes_per90": "high passing volume",
    "passes_completed_per90": "retaining possession with accurate passing",
    "progressive_passes_per90": "progressing the ball forward with passes",
    "dribbles_per90": "taking opponents on",
    "dribbles_completed_per90": "beating defenders with dribbles",
    "carries_per90": "carrying the ball",
    "progressive_carries_per90": "driving forward with the ball at his feet",
    "tackles_per90": "winning tackles",
    "interceptions_per90": "reading the game and intercepting passes",
    "blocks_per90": "blocking shots and passes",
    "clearances_per90": "clearing danger from defence",
    "ball_recoveries_per90": "recovering loose balls",
    "pressures_per90": "pressing opponents high",
}


def build_profiles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One natural-language profile document per player."""
    scored = score_frame(frame)
    docs: list[dict[str, Any]] = []

    for group, sub in scored.groupby("position_group"):
        percentiles = sub[list(PER90_FEATURES)].rank(pct=True) * 100.0
        for idx, r in sub.iterrows():
            ranked = percentiles.loc[idx].sort_values(ascending=False)
            strengths = ", ".join(_phrase_with_value(r, f, ranked[f]) for f in ranked.index[:4])
            weakest = ranked.index[-1]
            text = (
                f"{r['name']} is a {r['primary_position']} ({group}) in {r['competition']}. "
                f"Playing style: he excels at {strengths}. "
                f"He is weaker at {_phrase_with_value(r, weakest, ranked[weakest])}. "
                f"Performance score {r['performance_score']:.0f} out of 100 "
                f"over {int(r['minutes'])} minutes played."
            )
            docs.append({"player_id": int(r["player_id"]), "name": r["name"], "text": text})
    return docs


def _phrase_with_value(row: pd.Series, feature: str, percentile: float) -> str:
    """ "scoring goals (0.61 per 90, 97th percentile among his position group)".

    The numbers matter: without them the assistant cannot answer comparative questions
    ("who has the most xG?") because every profile reduces to the same stock phrases.
    """
    return (
        f"{STYLE_PHRASES[feature]} "
        f"({row[feature]:.2f} {_metric_label(feature)} per 90, "
        f"{_ordinal(round(percentile))} percentile for a {row['position_group']})"
    )


def _metric_label(feature: str) -> str:
    """ "ball_recoveries_per90" -> "ball recoveries"; "xg_per90" -> "xG"."""
    name = feature.removesuffix("_per90")
    return "xG" if name == "xg" else name.replace("_", " ")


def _ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 83 -> 83rd, 11 -> 11th."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
