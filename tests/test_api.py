"""HTTP-level tests for the API — real routing and SQL, no Postgres and no live LLM.

These cover what the routers add on top of the ML/search layers: request validation,
response shape and the error contract (404 for unknown players, 422 for bad input,
503 when the local LLM is unreachable).
"""

from __future__ import annotations

import pytest

from footyvision.api.routers import assistant as assistant_router
from footyvision.api.routers import reports as reports_router
from footyvision.api.routers import search as search_router
from footyvision.llm.client import LLMError
from footyvision.search.nl import NLParseError
from footyvision.search.query import PlayerQuery

# --- meta -----------------------------------------------------------------------------


def test_root_reports_version(client):
    body = client.get("/").json()
    assert body["name"] == "FootyVision API"
    assert body["version"]


def test_health_reports_database_reachable(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] is True


def test_openapi_schema_is_generated(client):
    # Catches response_model/annotation mistakes that only surface at schema build time.
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/players/{player_id}/similar" in schema.json()["paths"]


# --- players --------------------------------------------------------------------------


def test_list_players_returns_all_seeded(client):
    body = client.get("/players").json()
    assert len(body) == 6
    assert body[0]["name"] == "Alpha Striker"  # ordered by name


def test_list_players_filters_by_name_case_insensitively(client):
    body = client.get("/players", params={"search": "anchor"}).json()
    assert {p["name"] for p in body} == {"Delta Anchor", "Echo Anchor"}


def test_list_players_can_exclude_players_without_a_season_aggregate(client, db_session):
    # A player who only ever appeared in a match has no radar and no score, so the
    # dashboard must be able to keep them out of the search results.
    from footyvision.db.models import Player

    db_session.add(Player(id=99, name="Zulu Benchwarmer"))
    db_session.commit()

    everyone = {p["name"] for p in client.get("/players").json()}
    selectable = {p["name"] for p in client.get("/players", params={"with_stats": True}).json()}
    assert "Zulu Benchwarmer" in everyone
    assert "Zulu Benchwarmer" not in selectable
    assert len(selectable) == 6


def test_list_players_rejects_out_of_range_limit(client):
    assert client.get("/players", params={"limit": 500}).status_code == 422


def test_get_player_returns_404_for_unknown_id(client):
    response = client.get("/players/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Player not found"


def test_player_seasons_returns_per90_rates(client):
    body = client.get("/players/1/seasons").json()
    assert len(body) == 1
    assert body[0]["xg_per90"] == pytest.approx(0.80)


# --- similarity and radar -------------------------------------------------------------


def test_similar_players_excludes_the_target_and_ranks_by_similarity(client):
    body = client.get("/players/1/similar", params={"min_minutes": 500}).json()
    assert body["target"]["name"] == "Alpha Striker"
    returned = [r["player_id"] for r in body["results"]]
    assert 1 not in returned
    scores = [r["similarity"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_similar_players_stays_within_the_position_group(client):
    body = client.get("/players/1/similar", params={"min_minutes": 500}).json()
    assert {r["position_group"] for r in body["results"]} == {"FWD"}


def test_similar_players_404s_when_below_the_minutes_floor(client):
    # Player 6 has 200 minutes, so a 500-minute floor removes them from the pool.
    response = client.get("/players/6/similar", params={"min_minutes": 500})
    assert response.status_code == 404


def test_radar_returns_percentiles_for_every_metric(client):
    body = client.get("/players/1/radar", params={"min_minutes": 500}).json()
    assert body["position_group"] == "FWD"
    # The best xG per 90 among forwards must sit at the top of its percentile range.
    assert body["metrics"]["xg_per90"]["percentile"] == pytest.approx(100.0)


# --- search ---------------------------------------------------------------------------


def test_structured_search_needs_no_llm(client):
    response = client.post(
        "/search/structured",
        json={
            "position_group": "FWD",
            "min_minutes": 500,
            "conditions": [{"field": "xg_per90", "op": "gt", "value": 0.5}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert {r["name"] for r in body["results"]} == {"Alpha Striker", "Bravo Striker"}


def test_structured_search_rejects_a_field_outside_the_whitelist(client):
    response = client.post(
        "/search/structured",
        json={"conditions": [{"field": "salary", "op": "gt", "value": 1}]},
    )
    assert response.status_code == 422


def test_nl_search_runs_the_query_the_llm_structured(client, monkeypatch):
    monkeypatch.setattr(
        search_router,
        "parse_nl",
        lambda _q: PlayerQuery(position_group="FWD", min_minutes=500, order_by="xg_per90"),
    )
    body = client.post("/search", json={"query": "best forwards"}).json()
    assert body["interpreted"]["position_group"] == "FWD"
    assert body["results"][0]["name"] == "Alpha Striker"


def test_nl_search_returns_503_when_the_llm_is_unreachable(client, monkeypatch):
    def _unreachable(_q):
        raise LLMError("connection refused")

    monkeypatch.setattr(search_router, "parse_nl", _unreachable)
    response = client.post("/search", json={"query": "best forwards"})
    assert response.status_code == 503


def test_nl_search_returns_422_when_the_llm_output_is_unusable(client, monkeypatch):
    def _unparsable(_q):
        raise NLParseError("model did not return valid JSON")

    monkeypatch.setattr(search_router, "parse_nl", _unparsable)
    response = client.post("/search", json={"query": "???"})
    assert response.status_code == 422


# --- LLM-backed endpoints -------------------------------------------------------------


def test_report_context_is_computed_without_an_llm(client):
    body = client.get("/players/1/report/context", params={"min_minutes": 500}).json()
    assert body["player_id"] == 1
    assert body["context"]


def test_report_context_404s_for_a_player_outside_the_pool(client):
    response = client.get("/players/999/report/context", params={"min_minutes": 500})
    assert response.status_code == 404


def test_report_returns_503_when_the_llm_is_unreachable(client, monkeypatch):
    def _unreachable(*_args, **_kwargs):
        raise LLMError("LM Studio is not running")

    monkeypatch.setattr(reports_router, "generate_report", _unreachable)
    response = client.post("/players/1/report")
    assert response.status_code == 503
    assert "LM Studio" in response.json()["detail"]


def test_assistant_returns_503_when_the_llm_is_unreachable(client, monkeypatch):
    def _unreachable(_session):
        raise LLMError("embedding model not loaded")

    monkeypatch.setattr(assistant_router, "get_store", _unreachable)
    response = client.post("/assistant", json={"question": "who wins the ball back?"})
    assert response.status_code == 503


# --- metric distribution --------------------------------------------------------------


def test_metric_distribution_returns_every_player_in_the_pool(client):
    body = client.get("/metrics/xg_per90/distribution", params={"min_minutes": 500}).json()
    assert body["metric"] == "xg_per90"
    # Player 6 sits below the minutes floor, so five of the six seeded players remain.
    assert body["count"] == 5
    assert max(v["value"] for v in body["values"]) == pytest.approx(0.80)


def test_metric_distribution_can_scope_to_a_position_group(client):
    body = client.get(
        "/metrics/xg_per90/distribution",
        params={"min_minutes": 500, "position_group": "FWD"},
    ).json()
    assert body["position_group"] == "FWD"
    assert {v["name"] for v in body["values"]} == {
        "Alpha Striker",
        "Bravo Striker",
        "Charlie Striker",
    }


def test_metric_distribution_rejects_a_metric_outside_the_feature_set(client):
    response = client.get("/metrics/salary/distribution")
    assert response.status_code == 422


# --- rate limiting and CORS -------------------------------------------------------------


def test_llm_endpoints_are_throttled_per_client(client, monkeypatch):
    """An unthrottled public deployment lets anyone drain the API key attached to it."""
    from footyvision.api import limits
    from footyvision.config import Settings, get_settings

    limits.reset()
    get_settings.cache_clear()
    monkeypatch.setattr(
        get_settings, "__wrapped__", lambda: Settings(rate_limit_per_minute=2), raising=False
    )
    monkeypatch.setattr(limits, "get_settings", lambda: Settings(rate_limit_per_minute=2))

    def _answer(self, question, k=6):
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(assistant_router.ScoutAssistant, "answer", _answer)
    monkeypatch.setattr(assistant_router, "get_store", lambda *_a, **_k: object())

    codes = [client.post("/assistant", json={"question": "q"}).status_code for _ in range(3)]

    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    limits.reset()


def test_a_throttled_response_says_when_to_retry(client, monkeypatch):
    from footyvision.api import limits
    from footyvision.config import Settings

    limits.reset()
    monkeypatch.setattr(limits, "get_settings", lambda: Settings(rate_limit_per_minute=1))

    def _answer(self, question, k=6):
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(assistant_router.ScoutAssistant, "answer", _answer)
    monkeypatch.setattr(assistant_router, "get_store", lambda *_a, **_k: object())

    client.post("/assistant", json={"question": "q"})
    blocked = client.post("/assistant", json={"question": "q"})

    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
    limits.reset()


def test_rate_limit_of_zero_disables_throttling(client, monkeypatch):
    """Local development and the rest of this suite run with the limiter off."""
    from footyvision.api import limits
    from footyvision.config import Settings

    limits.reset()
    monkeypatch.setattr(limits, "get_settings", lambda: Settings(rate_limit_per_minute=0))

    def _answer(self, question, k=6):
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(assistant_router.ScoutAssistant, "answer", _answer)
    monkeypatch.setattr(assistant_router, "get_store", lambda *_a, **_k: object())

    codes = [client.post("/assistant", json={"question": "q"}).status_code for _ in range(5)]

    assert codes == [200] * 5


def test_clients_are_told_apart_by_the_forwarded_header():
    """Render and Vercel terminate TLS in front of the app, so every caller would
    otherwise share one bucket behind the proxy's own address."""
    from starlette.datastructures import Headers

    from footyvision.api.limits import client_key

    class _Request:
        def __init__(self, forwarded):
            self.headers = Headers({"x-forwarded-for": forwarded} if forwarded else {})
            self.client = None

    assert client_key(_Request("203.0.113.7, 10.0.0.1")) == "203.0.113.7"
    assert client_key(_Request(None)) == "unknown"


def test_cors_defaults_to_local_development_when_unset():
    """ "*" would let any site on the internet spend this instance's LLM budget."""
    from footyvision.config import Settings

    assert Settings(cors_origins="").allowed_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert Settings(cors_origins="https://a.app, https://b.app").allowed_origins == [
        "https://a.app",
        "https://b.app",
    ]


def test_cors_allows_vercel_origins():
    from fastapi.testclient import TestClient

    from footyvision.api.main import app

    client = TestClient(app)
    response = client.get("/", headers={"Origin": "https://footy-vision-tau.vercel.app"})
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin") == "https://footy-vision-tau.vercel.app"
    )
