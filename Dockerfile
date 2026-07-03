FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build deps for psycopg2 + runtime libgomp1 for LightGBM/XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# --trusted-host survives TLS-intercepting corporate proxies (see docs/ROADMAP note).
RUN pip install --upgrade pip \
    && pip install -e . --trusted-host pypi.org --trusted-host files.pythonhosted.org

EXPOSE 8000

# Create tables (idempotent) then serve the API.
CMD ["sh", "-c", "footyvision init-db && uvicorn footyvision.api.main:app --host 0.0.0.0 --port 8000"]
