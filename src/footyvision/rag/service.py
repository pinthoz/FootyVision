"""Lazily build/load and cache the player vector store for the assistant."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from footyvision.config import get_settings
from footyvision.llm.client import LLMClient
from footyvision.ml.features import load_feature_frame
from footyvision.rag.profiles import build_profiles
from footyvision.rag.store import VectorStore

STORE_PATH = Path("data/player_vectors.npz")

_STORE: VectorStore | None = None


def build_store(session: Session, client: LLMClient | None = None) -> VectorStore:
    """Embed all player profiles and persist the vector store to disk."""
    frame = load_feature_frame(session, get_settings().min_minutes)
    docs = build_profiles(frame)
    store = VectorStore.build(docs, client or LLMClient())
    store.save(STORE_PATH)
    return store


def get_store(session: Session, rebuild: bool = False) -> VectorStore:
    """Return the cached store; load from disk or build (embedding) on first use."""
    global _STORE
    if _STORE is not None and not rebuild:
        return _STORE
    if not rebuild and STORE_PATH.exists():
        _STORE = VectorStore.load(STORE_PATH)
    else:
        _STORE = build_store(session)
    return _STORE
