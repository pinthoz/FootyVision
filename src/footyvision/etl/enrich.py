"""Fill in the player details the event stream does not carry.

`sb.events()` names a player and nothing else, so the players table ends up with an id
and a name. The lineup feed for the same match carries two more fields worth having:

  * `player_nickname` — the short form the player is actually known by ("Joselu" for
    "José Luis Sanmartín Mato"). Other providers use this form, so it is a far better
    key for cross-source matching than the full legal name.
  * `country` — nationality, a basic scouting filter that was sitting unused.

Date of birth, preferred foot and height are absent from the open data entirely and are
matched from the Transfermarkt export instead.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from statsbombpy import sb

from footyvision.db.models import Match, Player

warnings.filterwarnings("ignore", message="credentials were not supplied")


def enrich_from_lineups(
    session: Session,
    on_match: Callable[[int, int, int], None] | None = None,
) -> dict[str, int]:
    """Walk loaded matches filling nickname and country, stopping once nothing is missing.

    A player only appears in the lineups of matches they were in the squad for, so this
    has to walk matches rather than players. It does not have to walk *all* of them: the
    squads repeat heavily, so the set of players still missing detail empties long before
    the fixture list does, and the loop exits there.
    """
    players = {p.id: p for p in session.scalars(select(Player))}
    pending = {pid for pid, p in players.items() if p.nickname is None and p.country is None}
    match_ids = list(session.scalars(select(Match.id)))

    filled_nickname = 0
    filled_country = 0
    seen = 0

    for match_id in match_ids:
        if not pending:
            break
        seen += 1
        try:
            lineups = sb.lineups(match_id=match_id)
        except Exception:
            # A single unavailable lineup should not abandon the rest of the walk.
            continue

        for frame in lineups.values():
            for row in frame.itertuples(index=False):
                pid = int(row.player_id)
                player = players.get(pid)
                if player is None:
                    continue
                nickname = getattr(row, "player_nickname", None)
                if player.nickname is None and isinstance(nickname, str) and nickname.strip():
                    player.nickname = nickname.strip()[:120]
                    filled_nickname += 1
                country = getattr(row, "country", None)
                if player.country is None and isinstance(country, str) and country.strip():
                    player.country = country.strip()[:120]
                    filled_country += 1
                if player.nickname is not None or player.country is not None:
                    pending.discard(pid)

        if on_match:
            on_match(match_id, seen, len(pending))

    session.commit()
    return {
        "matches_read": seen,
        "nicknames": filled_nickname,
        "countries": filled_country,
        "still_missing": len(pending),
    }


def _normalize(name: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(ascii_.lower().split())


def _transfermarkt_lookup(data_dir: str, columns: list[str]) -> dict[str, dict]:
    """Name-keyed view of the Transfermarkt export, with ambiguous names left out.

    A name appearing more than once cannot be resolved by name alone, so those rows are
    dropped rather than guessed at — a wrong date of birth is worse than none.
    """
    frame = pd.read_csv(Path(data_dir) / "players.csv", usecols=["name", *columns])
    frame = frame.dropna(subset=["name"])
    frame["key"] = frame["name"].map(_normalize)
    frame = frame.drop_duplicates("key", keep=False)
    return {r["key"]: {c: r[c] for c in columns} for r in frame.to_dict("records")}


def _match(player: Player, lookup: dict[str, dict]) -> dict | None:
    """Nickname first, full name second.

    StatsBomb records full legal names ("José Luis Sanmartín Mato") while Transfermarkt
    uses the short form ("Joselu"), so the nickname is usually the one that lands.
    """
    for candidate in (player.nickname, player.name):
        if candidate:
            hit = lookup.get(_normalize(candidate))
            if hit is not None:
                return hit
    return None


def backfill_birthdates(session: Session, data_dir: str = "data") -> dict[str, int]:
    """Attach dates of birth from the Transfermarkt export, matched by name."""
    lookup = _transfermarkt_lookup(data_dir, ["date_of_birth"])
    matched = 0
    players = list(session.scalars(select(Player)))

    for player in players:
        if player.date_of_birth is not None:
            continue
        hit = _match(player, lookup)
        if hit is None:
            continue
        dob = pd.to_datetime(hit["date_of_birth"], errors="coerce")
        if pd.notna(dob):
            player.date_of_birth = dob.date()
            matched += 1

    session.commit()
    return {
        "players": len(players),
        "matched": matched,
        "unmatched": sum(1 for p in players if p.date_of_birth is None),
    }


_FEET = {"right", "left", "both"}


def backfill_physical(session: Session, data_dir: str = "data") -> dict[str, int]:
    """Attach preferred foot and height from the Transfermarkt export.

    Foot is the one that matters to the model: a direct laterality signal worth roughly
    ten accuracy points on exact-position prediction, which the per-90 counts cannot
    supply because they carry no side. Height is stored as a scouting attribute only —
    the same ablation showed it adds nothing to position prediction.
    """
    lookup = _transfermarkt_lookup(data_dir, ["foot", "height_in_cm"])
    feet = 0
    heights = 0
    players = list(session.scalars(select(Player)))

    for player in players:
        hit = _match(player, lookup)
        if hit is None:
            continue
        foot = hit.get("foot")
        if player.foot is None and isinstance(foot, str) and foot.lower() in _FEET:
            player.foot = foot.lower()
            feet += 1
        height = hit.get("height_in_cm")
        # The export uses 0 for "unknown"; a 0 cm player is worse than a null one.
        if player.height_cm is None and pd.notna(height) and float(height) > 100:
            player.height_cm = float(height)
            heights += 1

    session.commit()
    return {
        "players": len(players),
        "feet": feet,
        "heights": heights,
        "without_foot": sum(1 for p in players if p.foot is None),
    }
