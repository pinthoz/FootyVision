from __future__ import annotations

from footyvision.config import Settings


def test_sqlalchemy_url_is_built_from_parts():
    s = Settings(
        postgres_user="u",
        postgres_password="p",
        postgres_host="h",
        postgres_port=1234,
        postgres_db="d",
        database_url=None,
    )
    assert s.sqlalchemy_url == "postgresql+psycopg2://u:p@h:1234/d"


def test_explicit_database_url_wins():
    s = Settings(database_url="postgresql+psycopg2://x/y")
    assert s.sqlalchemy_url == "postgresql+psycopg2://x/y"
