"""Unit tests for structured/NL search — no DB or live LLM."""

from __future__ import annotations

import pandas as pd
import pytest

from footyvision.ml.features import PER90_FEATURES, position_group
from footyvision.search import nl, query
from footyvision.search.nl import NLParseError, parse_nl
from footyvision.search.query import Condition, PlayerQuery, execute_query


def test_condition_rejects_unknown_field():
    with pytest.raises(ValueError):
        Condition(field="age", op="lt", value=23)


def test_player_query_rejects_unknown_order_by():
    with pytest.raises(ValueError):
        PlayerQuery(order_by="salary")


def test_player_query_tolerates_llm_nulls():
    # LLMs emit explicit nulls; non-optional fields must fall back to defaults.
    q = PlayerQuery.model_validate({"limit": None, "order_desc": None, "position_group": None})
    assert q.limit == 20
    assert q.order_desc is True
    assert q.position_group is None


def _row(pid, name, comp, position, minutes, **feats) -> dict:
    d = {
        "player_id": pid,
        "name": name,
        "competition": comp,
        "competition_id": 1,
        "sb_season_id": 1,
        "primary_position": position,
        "matches_played": 20,
        "minutes": minutes,
        "position_group": position_group(position),
    }
    d.update({f: 0.0 for f in PER90_FEATURES})
    d.update(feats)
    return d


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(1, "Winger A", "La Liga", "Right Wing", 2500, xg_per90=0.30, dribbles_per90=5.0),
            _row(2, "Winger B", "La Liga", "Left Wing", 2500, xg_per90=0.20, dribbles_per90=6.0),
            _row(3, "Striker C", "Bundesliga", "Center Forward", 2500, xg_per90=0.55),
        ]
    )


def test_execute_query_filters_group_competition_and_condition(monkeypatch):
    monkeypatch.setattr(query, "load_feature_frame", lambda *a, **k: _frame())
    q = PlayerQuery(
        position_group="FWD",
        competition="La Liga",  # excludes the Bundesliga striker
        conditions=[Condition(field="xg_per90", op="gt", value=0.25)],
        order_by="xg_per90",
    )
    rows = execute_query(session=None, query=q)
    assert [r["name"] for r in rows] == ["Winger A"]
    assert rows[0]["stats"]["xg_per90"] == 0.30


def test_execute_query_orders_and_limits(monkeypatch):
    monkeypatch.setattr(query, "load_feature_frame", lambda *a, **k: _frame())
    q = PlayerQuery(order_by="dribbles_per90", order_desc=True, limit=2)
    rows = execute_query(session=None, query=q)
    assert [r["name"] for r in rows] == ["Winger B", "Winger A"]


class _FakeClient:
    def __init__(self, payload: str):
        self.payload = payload

    def chat(self, system: str, user: str, **_) -> str:
        return self.payload


def test_parse_nl_extracts_fenced_json():
    payload = (
        '```json\n{"position_group": "FWD", "conditions": '
        '[{"field": "xg_per90", "op": "gt", "value": 0.25}], "limit": 10}\n```'
    )
    q = parse_nl("strikers with xg over 0.25", client=_FakeClient(payload))
    assert q.position_group == "FWD"
    assert q.conditions[0].field == "xg_per90"
    assert q.limit == 10


def test_parse_nl_rejects_invalid_field():
    payload = '{"conditions": [{"field": "age", "op": "lt", "value": 23}]}'
    with pytest.raises(NLParseError):
        parse_nl("under 23", client=_FakeClient(payload))


def test_extract_json_raises_without_object():
    with pytest.raises(NLParseError):
        nl._extract_json("no json here")
