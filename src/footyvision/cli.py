"""FootyVision command line: initialise the schema and run the ETL.

Examples
--------
    footyvision init-db
    footyvision competitions
    footyvision load --competition 43 --season 3 --limit 5
    footyvision aggregate --competition 43 --season 3
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from footyvision.config import get_settings
from footyvision.db.base import Base, SessionLocal, engine
from footyvision.etl import load, statsbomb

app = typer.Typer(add_completion=False, help="FootyVision ETL & admin CLI.")
console = Console()


@app.command("init-db")
def init_db() -> None:
    """Create all tables (quick start; use Alembic migrations for production)."""
    import footyvision.db.models  # noqa: F401  (register mappers)

    Base.metadata.create_all(engine)
    console.print("[green]Schema created.[/green]")


@app.command("competitions")
def competitions() -> None:
    """List the competition/season pairs available in StatsBomb Open Data."""
    df = statsbomb.competitions()
    cols = ["competition_id", "season_id", "country_name", "competition_name", "season_name"]
    table = Table(title="StatsBomb Open Data — competitions")
    for col in cols:
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(*(str(r[c]) for c in cols))
    console.print(table)


@app.command("load")
def load_cmd(
    competition: int = typer.Option(..., "--competition", "-c"),
    season: int = typer.Option(..., "--season", "-s"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Only load first N matches."),
    aggregate: bool = typer.Option(True, help="Rebuild season aggregates after loading."),
) -> None:
    """Extract + aggregate + load a competition season into Postgres."""
    settings = get_settings()
    with SessionLocal() as session:
        def _progress(match_row, i, total):
            console.print(f"  [{i}/{total}] match {match_row['match_id']} loaded")

        console.print(f"[cyan]Loading competition={competition} season={season}...[/cyan]")
        n = load.load_competition_season(session, competition, season, limit, on_match=_progress)
        console.print(f"[green]{n} matches loaded.[/green]")

        if aggregate:
            written = load.rebuild_season_aggregates(
                session, competition, season, settings.min_minutes
            )
            console.print(
                f"[green]{written} player-season rows aggregated "
                f"(min {settings.min_minutes} minutes).[/green]"
            )


@app.command("aggregate")
def aggregate_cmd(
    competition: int = typer.Option(..., "--competition", "-c"),
    season: int = typer.Option(..., "--season", "-s"),
) -> None:
    """Recompute season aggregates from already-loaded match stats."""
    settings = get_settings()
    with SessionLocal() as session:
        written = load.rebuild_season_aggregates(session, competition, season, settings.min_minutes)
        console.print(f"[green]{written} player-season rows aggregated.[/green]")


if __name__ == "__main__":
    app()
