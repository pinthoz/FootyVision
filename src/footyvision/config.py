"""Central application settings, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    postgres_user: str = "footy"
    postgres_password: str = "footy"
    postgres_db: str = "footyvision"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    # --- ETL ---
    min_minutes: int = Field(default=600, description="Min season minutes to include a player.")

    # --- LLM (local, OpenAI-compatible endpoint) ---
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_embed_model: str = "text-embedding-embeddinggemma-300m-qat"
    # Embedding models are trained with task prefixes and lose accuracy without them.
    # These are EmbeddingGemma's; nomic-embed-text uses "search_document: "/"search_query: ".
    # See scripts/eval_embeddings.py for the measurements behind this default.
    llm_embed_document_prefix: str = "title: none | text: "
    llm_embed_query_prefix: str = "task: search result | query: "
    llm_api_key: str = "not-needed-for-local"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
