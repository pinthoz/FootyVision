# FootyVision ⚽

**An AI football scouting platform.** It helps sporting directors, scouts and analysts answer
questions like *"who are the best stylistic replacements for this player?"*, *"which players
fit a role?"*, *"is this signing good value?"* and *"find me a ball-winning midfielder"*.

**Guiding principle:** traditional ML models do the maths; the **LLM explains and interprets**
(scouting reports, comparisons, natural-language search, a RAG assistant) — it never invents numbers.

> Built end-to-end: StatsBomb event ETL → Postgres → similarity / scoring / value ML →
> local LLM + embeddings → FastAPI + a web dashboard. Runs fully **locally** (no paid APIs).

---

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

## Tech stack

- **Backend:** FastAPI · SQLAlchemy 2 · Alembic · PostgreSQL
- **Data/ML:** pandas · scikit-learn · **XGBoost** · **LightGBM** · **SHAP** · rapidfuzz
- **LLM (local):** OpenAI-compatible endpoint — chat (Gemma/Qwen/Llama) + embeddings (`nomic-embed-text`)
- **Data sources:** StatsBomb Open Data (`statsbombpy`) · Transfermarkt values (Kaggle) · FBref/SoFIFA (`soccerdata`)
- **Frontend:** self-contained radar demo (`frontend/radar_demo.html`, Plotly) · Next.js dashboard (`frontend/web`)
- **Infra:** Docker Compose

## Quick start

Prerequisites: Docker (Postgres), Python 3.11+, and a local LLM server (LM Studio or Ollama)
for the LLM/RAG features.

```bash
# 1. Postgres
docker compose up -d db

# 2. Python env + package
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows;  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# 3. Schema + data (StatsBomb Open Data)
footyvision init-db
footyvision competitions              # list available competitions
footyvision load -c 11 -s 27          # La Liga 2015/16 (the one full season; ~380 matches)

# 4. API
uvicorn footyvision.api.main:app --reload    # -> http://localhost:8000/docs
```

**LLM features** (reports, NL search, assistant): start LM Studio, load a chat model + the
`nomic-embed-text` embedding model, Start Server (port 1234), set `LLM_MODEL` in `.env`, then:

```bash
footyvision index                     # embed player profiles into the RAG vector store
```

**Value model** (optional): needs Kaggle access to `davidcariboo/player-scores`; drop
`players.csv` + `player_valuations.csv` into `data/`, then `footyvision value-report`.

**Full stack in Docker:** `docker compose up --build` (the API reaches the host LLM via
`host.docker.internal`). The web dashboard lives in `frontend/web` (`npm install && npm run dev`).

## API reference

`/health` · `/llm/health` · `/players` · `/players/{id}` · `/players/{id}/seasons` ·
`/players/{id}/similar` · `/players/{id}/radar` · `/players/{id}/score` · `/rankings` ·
`/talent/model-info` · `POST /players/{id}/report` · `/players/{id}/report/context` ·
`POST /search` · `POST /search/structured` · `POST /assistant`. Full docs at `/docs`.

## Project layout

```
src/footyvision/
  config.py                 # env-driven settings
  db/       base.py, models.py
  etl/      statsbomb.py, aggregate.py, load.py, transfermarkt.py, sofifa.py
  ml/       features.py, similarity.py, scoring.py, talent.py, value.py
  llm/      client.py (chat + embeddings), scouting.py
  search/   query.py (safe PlayerQuery), nl.py (NL → query)
  rag/      profiles.py, store.py, assistant.py, service.py
  api/      main.py, routers/, schemas.py
  cli.py    # init-db · load · aggregate · talent-report · value-report · index
frontend/   radar_demo.html · web/ (Next.js)
tests/      # 31 unit tests, DB/network/LLM-free
```

## Engineering notes (honest data reality)

This project deliberately reports what public/free data **can't** do, not just what it can:

- **StatsBomb Open Data** has no Portuguese league and only **one** complete domestic season
  (La Liga 2015/16) — so the pool is that season + partial Bundesliga.
- **FBref** only exposes advanced stats (xG, progression) for the Big-5 leagues, so a rich
  Primeira Liga engine isn't feasible from free sources.
- The **value model** is trained on real Transfermarkt values but scores a low held-out
  R² (≈0.05): one season of public per-90 stats + age barely predicts market value (SHAP
  correctly ranks age #1). Cross-source name matching (Spanish multi-surnames ↔ short TM
  names) adds label noise. A truthful evaluation beats an inflated one.

## Tests

```bash
pytest          # 31 tests, no DB/network/LLM required
ruff check src tests
```
