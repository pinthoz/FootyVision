# FootyVision

**AI football scouting platform.** Helps sporting directors, scouts and analysts answer
questions like *"who are the best stylistic replacements for this player?"*, *"which young
players have the highest upside?"* and *"is this signing good value?"*.

The guiding principle: **traditional ML models make the predictions; the LLM explains and
interprets them** (scouting reports, comparisons, natural-language search).

---

## Status

**Phase 0 — Foundations (current).** Project scaffold, Postgres schema, and a working ETL
that turns [StatsBomb Open Data](https://github.com/statsbomb/open-data) event streams into
per-player season statistics. See [docs/ROADMAP.md](docs/ROADMAP.md) for what comes next.

## Architecture (target)

```
StatsBomb Open Data ──► ETL (extract ▸ aggregate ▸ load) ──► PostgreSQL
                                                                  │
                 ┌────────────────────────────────────────────────┼───────────────┐
                 ▼                         ▼                        ▼
         Similarity Engine          Talent Score (XGBoost)   Value Predictor
         (z-score + cosine)         + SHAP explainability    (later phase)
                 └───────────────┬──────────────────────────────────┘
                                 ▼
                     LLM layer (local, LM Studio/Ollama)
                  reports ▸ comparisons ▸ NL → safe SQL search
                                 ▼
                       Next.js scout dashboard
```

## Tech stack

- **Backend:** FastAPI · SQLAlchemy 2 · Alembic · PostgreSQL
- **Data/ML:** pandas · scikit-learn · XGBoost · SHAP · UMAP (visualisation only)
- **LLM:** local OpenAI-compatible endpoint (LM Studio / Ollama — Qwen/Gemma/Llama)
- **Frontend (later):** Next.js · Plotly/Recharts
- **Infra:** Docker Compose

## Quick start

Prerequisites: Docker (for Postgres) and Python 3.11+.

```bash
# 1. Configure environment
cp .env.example .env

# 2. Start Postgres
docker compose up -d db

# 3. Install the package (editable) in a virtualenv
python -m venv .venv
. .venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 4. Create the schema
footyvision init-db

# 5. See what data is available, then load a season
footyvision competitions
footyvision load --competition 43 --season 3 --limit 5   # FIFA World Cup 2018, first 5 matches

# 6. Run the API
uvicorn footyvision.api.main:app --reload
# -> http://localhost:8000/docs
```

Run the full stack (API + DB) with `docker compose up --build`.

## Project layout

```
src/footyvision/
  config.py            # env-driven settings
  db/         base.py  # engine/session/Base
              models.py
  etl/        statsbomb.py   # extract (statsbombpy)
              aggregate.py   # events -> per-player-match stats (+ minutes, progressive)
              load.py        # upsert to DB + season aggregates
  api/        main.py, routers/, schemas.py
  cli.py               # init-db / competitions / load / aggregate
tests/                 # unit tests (no DB/network needed)
```

## Tests

```bash
pytest
```

## Notes on data

Only StatsBomb Open Data is used (free, openly licensed, no scraping). Coverage is limited
to the competitions in that dataset. Market values (Transfermarkt) are intentionally out of
scope for now — see the roadmap.
