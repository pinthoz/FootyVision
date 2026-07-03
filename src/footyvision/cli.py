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


@app.command("talent-report")
def talent_report(
    min_minutes: int | None = typer.Option(None, "--min-minutes"),
) -> None:
    """Train the XGBoost position classifier and print its honest evaluation + SHAP."""
    # Imported lazily so the heavy ML stack only loads for this command.
    from footyvision.ml.features import load_feature_frame
    from footyvision.ml.talent import role_mismatches, shap_importance, train_position_classifier

    settings = get_settings()
    mm = settings.min_minutes if min_minutes is None else min_minutes
    with SessionLocal() as session:
        frame = load_feature_frame(session, mm)
        console.print(f"[cyan]Training position classifier on {len(frame)} players...[/cyan]")
        tm = train_position_classifier(frame)
        console.print(
            f"[green]Held-out accuracy: {tm.test_accuracy:.1%}[/green] "
            f"(train={tm.n_train}, test={tm.n_test}, classes={tm.classes})"
        )

        imp = Table(title="Top features by mean |SHAP| — which metrics define position")
        imp.add_column("feature")
        imp.add_column("mean |SHAP|", justify="right")
        for row in shap_importance(tm, frame, top_n=10):
            imp.add_row(row["feature"], f"{row['mean_abs_shap']:.3f}")
        console.print(imp)

        mism = Table(title="Role mismatches — stats resemble a different position")
        for col in ["player", "listed", "plays like", "confidence"]:
            mism.add_column(col)
        for r in role_mismatches(tm, frame, top_n=10):
            mism.add_row(r["name"], r["listed"], r["plays_like"], f"{r['confidence']:.2f}")
        console.print(mism)


@app.command("value-report")
def value_report(
    min_minutes: int | None = typer.Option(None, "--min-minutes"),
) -> None:
    """Match Transfermarkt 2015/16 values to our players, train LightGBM, print metrics."""
    from footyvision.etl.transfermarkt import read_laliga_values_2016
    from footyvision.ml.features import PER90_FEATURES, load_feature_frame
    from footyvision.ml.value import bargains, match_values, shap_importance, train_value_model

    settings = get_settings()
    mm = settings.min_minutes if min_minutes is None else min_minutes
    console.print("[cyan]Loading Transfermarkt La Liga 2015/16 values...[/cyan]")
    values = read_laliga_values_2016()
    console.print(f"  {len(values)} La Liga players with market values.")

    with SessionLocal() as session:
        features = load_feature_frame(session, mm)
    merged = match_values(features, values, keep_cols=("value_eur", "age"))
    console.print(f"[green]Matched {len(merged)}/{len(features)} players to a value.[/green]")

    vm = train_value_model(merged, feature_cols=[*PER90_FEATURES, "age"])
    console.print(
        f"[green]LightGBM value model — R²={vm.r2:.2f}, MAE=€{vm.mae_eur:,.0f}[/green] "
        f"(train={vm.n_train}, test={vm.n_test})"
    )

    imp = Table(title="Top features by mean |SHAP| — what drives value")
    imp.add_column("feature")
    imp.add_column("mean |SHAP|", justify="right")
    for row in shap_importance(vm, merged, top_n=10):
        imp.add_row(row["feature"], f"{row['mean_abs_shap']:.3f}")
    console.print(imp)

    barg = Table(title="Bargains — performance implies more value than FIFA price")
    for col in ["player", "pos", "FIFA value", "model value", "upside"]:
        barg.add_column(col)
    for r in bargains(vm, merged, top_n=10):
        barg.add_row(
            r["name"], r["position_group"], f"€{r['actual_value']:,.0f}",
            f"€{r['predicted_value']:,.0f}", f"€{r['upside']:,.0f}",
        )
    console.print(barg)


@app.command("index")
def index_cmd() -> None:
    """Embed all player profiles and persist the RAG vector store (needs the LLM server)."""
    from footyvision.rag.service import STORE_PATH, build_store

    with SessionLocal() as session:
        console.print("[cyan]Embedding player profiles via the local model...[/cyan]")
        store = build_store(session)
    console.print(f"[green]Indexed {len(store)} profiles -> {STORE_PATH}[/green]")


if __name__ == "__main__":
    app()
