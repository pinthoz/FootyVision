# FootyVision — Roadmap

The goal is a **vertical slice that looks like a professional scouting product**, not a
broad-but-shallow demo. Phases 1–3 alone already showcase data engineering, ML, LLMs and
full-stack work.

Guiding principle: **ML predicts, the LLM explains.**

---

## Phase 0 — Foundations ✅ (in progress)
- Repo scaffold, `pyproject`, Docker Compose, ruff/pytest.
- Postgres schema (competitions, seasons, teams, players, matches, per-match & per-season stats).
- ETL: StatsBomb Open Data → per-player-per-match aggregation (minutes, progressive passes/carries,
  defensive actions) → per-season totals and **per-90 rates**.
- FastAPI skeleton with health + player endpoints.

**Definition of done:** `footyvision load` populates the DB and `/players/{id}/seasons`
returns per-90 stats.

## Phase 1 — Similarity Engine
- Build the player feature matrix from `player_season_stats` (per-90 features).
- **Standardise per position group** (z-score) — never compare a striker to a centre-back.
- Cosine / Euclidean similarity → "most similar players to X".
- Minutes threshold already enforced in ETL (`MIN_MINUTES`) to avoid small-sample noise.
- UMAP projection for a 2D "player map" (visualisation only — *not* used for similarity math).
- API: `GET /players/{id}/similar`. Frontend: **radar charts** comparing two players.

## Phase 2 — LLM Scouting Reports
- Local OpenAI-compatible client (LM Studio / Ollama) behind a thin `LLMClient` interface.
- Feed structured stats + similar players → generate a scouting report (strengths, weaknesses,
  tactical fit, development potential, risk).
- The LLM only sees numbers we computed; it never invents stats.
- API: `POST /players/{id}/report`.

## Phase 3 — Natural-Language Search (safe text-to-SQL)
- LLM produces a **validated structured filter object** (Pydantic), not raw SQL.
- We translate that object into a parameterised, read-only query → no injection risk.
- Example: *"sub-23 wingers with > 0.25 xG per 90 in the Bundesliga"*.
- API: `POST /search`.

## Phase 4 — Talent Score (ML)
- Supervised model (XGBoost) over per-90 features (+ age when available) for an upside score.
- **SHAP** for explainability so the score is defensible, not a black box.
- Honest evaluation: train/test split by season, report metrics, avoid leakage.

## Phase 5 — Stretch goals
- **Market Value Predictor** (LightGBM) — requires a value dataset (e.g. a static Kaggle
  Transfermarkt export, not live scraping).
- **Career Simulator** — probability of reaching a top-5 league / CL / national team.
  Highest data risk (needs longitudinal labelled careers); treat as research, not a promise.
- **RAG scouting assistant** — vector DB over reports/news for conversational Q&A.

---

## Cross-cutting quality (do throughout)
- Alembic migrations once the schema stabilises (currently `init-db` uses `create_all`).
- Tests for each new computation; keep aggregation logic DB/network-free and unit-tested.
- Document every heuristic (e.g. what "progressive" means) so results are explainable.
