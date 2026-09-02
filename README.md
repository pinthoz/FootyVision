<div align="center">

# FootyVision ⚽

**An AI football scouting platform — the models do the maths, the LLM explains them.**

[![CI](https://github.com/pinthoz/FootyVision/actions/workflows/ci.yml/badge.svg)](https://github.com/pinthoz/FootyVision/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

[Quick start](#quick-start) · [What it does](#what-it-does) · [Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md) · [Contributing](CONTRIBUTING.md)

</div>

---

FootyVision helps sporting directors, scouts and analysts answer questions like *"who are the
best stylistic replacements for this player?"*, *"which players fit this role?"*, *"is this
signing good value?"* and *"find me a ball-winning midfielder"*.

**Guiding principle:** traditional ML models do the maths; the **LLM explains and interprets**
(scouting reports, comparisons, natural-language search, a RAG assistant) — it never invents
numbers.

> Built end-to-end: StatsBomb event ETL → Postgres → similarity / scoring / value ML →
> local LLM + embeddings → FastAPI + a web dashboard. Runs fully **locally**, with no paid APIs.

## What it does

| Capability | How it works | Try it |
|---|---|---|
| **Similarity engine** | Per-90 features, z-scored **within position group**, cosine similarity | `GET /players/{id}/similar` |
| **Scouting radars** | Percentile-vs-peers on each metric | `GET /players/{id}/radar` |
| **Performance Score** | Transparent position-weighted percentile composite (0–100) | `GET /players/{id}/score`, `GET /rankings` |
| **Role classifier** | XGBoost predicts position from style (~85%), **SHAP** explains, flags role-mismatches | `footyvision talent-report` |
| **LLM scouting reports** | Report grounded in computed stats; the model can't invent numbers | `POST /players/{id}/report` |
| **Natural-language search** | LLM → validated Pydantic `PlayerQuery` (never raw SQL) → safe query | `POST /search` |
| **Market value model** | LightGBM + age + SHAP on real Transfermarkt values | `footyvision value-report` |
| **RAG assistant** | Player profiles embedded locally → retrieve + grounded answer, cites names | `POST /assistant` |

**Example** — *"La Liga forwards with xG per 90 over 0.5"* → Ronaldo, Benzema, Suárez, Messi.
*"médio defensivo que ganhe bolas e intercete"* → retrieves Busquets/Camacho and recommends
Camacho, citing his real intercepting/tackling profile.

## Architecture

```
StatsBomb Open Data ──► ETL (extract ▸ aggregate ▸ load) ──► PostgreSQL
                                                                 │
        ┌──────────────┬──────────────────┬─────────────────────┼──────────────────┐
        ▼              ▼                  ▼                      ▼                  ▼
  Similarity     Performance Score   XGBoost role clf      Value predictor    RAG vector
  (z + cosine)   (weighted pctl)     + SHAP                (LightGBM + SHAP)   store (embeds)
        └──────────────┴────────┬─────────┴──────────────────────┴──────────────────┘
                                ▼
                 Local LLM  (LM Studio / Ollama — chat + embeddings)
              reports ▸ NL → safe query ▸ conversational RAG assistant
                                ▼
                    FastAPI  ──►  web dashboard (radar / search / assistant)
```

Layer-by-layer detail and the reasoning behind each design decision:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

## Tech stack

- **Backend:** FastAPI · SQLAlchemy 2 · Alembic · PostgreSQL
- **Data/ML:** pandas · scikit-learn · **XGBoost** · **LightGBM** · **SHAP** · rapidfuzz
- **LLM (local):** OpenAI-compatible endpoint — chat (Gemma/Qwen/Llama) + embeddings (`EmbeddingGemma-300M`)
- **Data sources:** StatsBomb Open Data (`statsbombpy`) · Transfermarkt values (Kaggle) · FBref/SoFIFA (`soccerdata`)
- **Frontend:** Next.js dashboard (`frontend/web`) · self-contained Plotly radar demo (`frontend/radar_demo.html`)
- **Infra:** Docker Compose · GitHub Actions

## Quick start

Prerequisites: Docker (for Postgres), Python 3.11+, and a local LLM server
(LM Studio or Ollama) for the LLM/RAG features.

```bash
# 1. Configuration
cp .env.example .env                  # the defaults work out of the box

# 2. Postgres
docker compose up -d db

# 3. Python env + package
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows;  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# 4. Schema + data (StatsBomb Open Data)
footyvision init-db
footyvision competitions              # list available competitions
footyvision load -c 11 -s 27          # La Liga 2015/16 (the one full season; ~380 matches)

# 5. API
uvicorn footyvision.api.main:app --reload    # -> http://localhost:8000/docs
```

**LLM features** (reports, NL search, assistant): start LM Studio, load a chat model + the
`embeddinggemma-300m` embedding model, Start Server (port 1234), set `LLM_MODEL` in `.env`, then:

```bash
footyvision index                     # embed player profiles into the RAG vector store
```

**Value model** (optional): needs Kaggle access to `davidcariboo/player-scores`; drop
`players.csv` + `player_valuations.csv` into `data/`, then `footyvision value-report`.

**Everything at once:**

- Docker: `docker compose up --build` (the API reaches the host LLM via `host.docker.internal`).
- Windows: `.\scripts\start.ps1` boots Postgres, LM Studio, the API and the dashboard.
- Dashboard only: `cd frontend/web && npm install && npm run dev`.

## API reference

`/health` · `/llm/health` · `/players` · `/players/{id}` · `/players/{id}/seasons` ·
`/players/{id}/similar` · `/players/{id}/radar` · `/players/{id}/score` · `/rankings` ·
`/talent/model-info` · `POST /players/{id}/report` · `/players/{id}/report/context` ·
`POST /search` · `POST /search/structured` · `POST /assistant`.
Interactive OpenAPI docs at `/docs`.

## Repository layout

```
src/footyvision/       # the Python package
  config.py            #   env-driven settings
  db/                  #   base.py, models.py            — SQLAlchemy schema
  etl/                 #   statsbomb, aggregate, load, transfermarkt, sofifa
  ml/                  #   features, similarity, scoring, talent, value
  llm/                 #   client.py (chat + embeddings), scouting.py
  search/              #   query.py (safe PlayerQuery), nl.py (NL → query)
  rag/                 #   profiles, store, assistant, service
  api/                 #   main.py, routers/, schemas.py
  cli.py               #   init-db · load · aggregate · talent-report · value-report · index
frontend/              # radar_demo.html · web/ (Next.js dashboard)
migrations/            # Alembic
scripts/               # start.ps1 · eval_embeddings.py (retrieval benchmark)
tests/                 # 52 tests (unit + API), DB/network/LLM-free
docs/                  # ARCHITECTURE.md, ROADMAP.md
```

## Data reality (and why it matters)

This project deliberately reports what public/free data **can't** do, not just what it can:

- **StatsBomb Open Data** has no Portuguese league and only **one** complete domestic season
  (La Liga 2015/16) — so the pool is that season plus a partial Bundesliga.
- **FBref** only exposes advanced stats (xG, progression) for the Big-5 leagues, so a rich
  Primeira Liga engine isn't feasible from free sources.
- The **embedding model was chosen by measurement, not by leaderboard**
  ([`scripts/eval_embeddings.py`](scripts/eval_embeddings.py) scores retrieval on these 411
  profiles). The original setup — nomic-embed-text with no task prefixes — retrieved the
  right position for only **30% of Portuguese queries** against 90% of English ones.
  EmbeddingGemma-300M with its proper prefixes reaches **73% / 97%**. A cross-lingual gap
  remains, and the script reports it rather than hiding it.
- The **value model** is trained on real Transfermarkt values but scores a low held-out
  R² (≈0.05): one season of public per-90 stats plus age barely predicts market value (SHAP
  correctly ranks age #1). Cross-source name matching (Spanish multi-surnames ↔ short TM
  names) adds label noise. A truthful evaluation beats an inflated one.

## Development

```bash
pytest                             # 52 tests, no Postgres/network/LLM required
ruff check src tests
ruff format --check src tests
```

CI runs the same checks on Python 3.11 and 3.12, plus a production build of the dashboard.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the ground rules (tests stay offline, the LLM
never invents numbers, no raw SQL from user input).

## License

[MIT](LICENSE) © Ana Pinto.

Football data from [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under
their terms; market values from the public Transfermarkt dataset on Kaggle. This project is
not affiliated with StatsBomb or Transfermarkt.
