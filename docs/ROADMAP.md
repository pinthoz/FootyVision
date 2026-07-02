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

## Phase 1 — Similarity Engine ✅
- Feature matrix from `player_season_stats` (per-90 features) — [ml/features.py](../src/footyvision/ml/features.py).
- **Standardised per position group** (z-score) — a striker is never compared to a centre-back.
- Cosine similarity → "most similar players to X" — [ml/similarity.py](../src/footyvision/ml/similarity.py).
- Minutes threshold enforced (`MIN_MINUTES`) to avoid small-sample noise.
- API: `GET /players/{id}/similar` and `GET /players/{id}/radar` (percentiles within group).
- Frontend: interactive **radar** comparing two players — [frontend/radar_demo.html](../frontend/radar_demo.html).
- *Still open:* UMAP 2D "player map" (visualisation only), Euclidean option, position-group weighting.

## Phase 2 — LLM Scouting Reports ✅
- Local OpenAI-compatible client behind a thin `LLMClient` — [llm/client.py](../src/footyvision/llm/client.py).
- Grounded context (radar percentiles + similar players + strengths/weaknesses) →
  prompt that **forbids inventing stats** → report — [llm/scouting.py](../src/footyvision/llm/scouting.py).
- API: `POST /players/{id}/report`, `GET /players/{id}/report/context` (grounding, no LLM),
  `GET /llm/health`. Report button wired into the radar UI.
- Needs a local LLM running (LM Studio / Ollama); returns HTTP 503 with guidance if it is not.
- *Still open:* stream tokens, cache reports, let the user pick report language/length.

## Phase 3 — Natural-Language Search (safe text-to-SQL) ✅
- LLM produces a **validated `PlayerQuery`** (Pydantic whitelist) — never raw SQL — so no
  injection is possible: [search/query.py](../src/footyvision/search/query.py),
  [search/nl.py](../src/footyvision/search/nl.py).
- API: `POST /search` (natural language, via LLM) and `POST /search/structured` (direct
  query, no LLM — powers testing and power users).
- Validated live: *"La Liga forwards with xG/90 > 0.5"* → Ronaldo/Benzema/Suárez/Messi;
  *"midfielders with progressive passes/90 > 7"* → Kroos/Modrić/Banega.
- *Known gap:* no age/DOB in the dataset, so "under-23"-style filters can't be answered
  (the prompt tells the LLM to ignore them).

## Phase 4 — Performance Score + XGBoost/SHAP ✅
- No future/age/value labels exist, so instead of a fabricated "potential" target we ship:
- **Performance Score (0-100)** — transparent, position-weighted percentile composite,
  fully traceable to metric contributions: [ml/scoring.py](../src/footyvision/ml/scoring.py).
- **XGBoost position classifier** on a REAL label (position group from style), with honest
  held-out evaluation (~85% accuracy), **SHAP** feature importance, per-player *style
  profiles*, and *role-mismatch* detection (e.g. Dani Alves plays like a MID):
  [ml/talent.py](../src/footyvision/ml/talent.py).
- API: `GET /players/{id}/score`, `GET /rankings`, `GET /talent/model-info`.
  CLI: `footyvision talent-report` (accuracy + SHAP + mismatches).

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
