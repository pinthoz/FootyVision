"""Unit tests for the RAG assistant — synthetic data, no DB or live LLM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footyvision.ml.features import PER90_FEATURES, position_group
from footyvision.rag.assistant import ScoutAssistant
from footyvision.rag.profiles import build_profiles
from footyvision.rag.store import VectorStore


def _frame() -> pd.DataFrame:
    rows = []
    specs = {"Center Forward": {"xg_per90": 0.6}, "Center Back": {"tackles_per90": 3.0}}
    pid = 1
    for position, base in specs.items():
        for _ in range(3):
            feats = {f: 0.0 for f in PER90_FEATURES}
            feats.update(base)
            rows.append({
                "player_id": pid, "name": f"Player {pid}", "competition": "La Liga",
                "primary_position": position, "position_group": position_group(position),
                "minutes": 1800, "matches_played": 20, **feats,
            })
            pid += 1
    return pd.DataFrame(rows)


def test_build_profiles_text():
    docs = build_profiles(_frame())
    assert len(docs) == 6
    d = docs[0]
    assert d["name"] in d["text"]
    assert "Performance score" in d["text"]


class _FakeClient:
    """embed maps text/query to a 2D vector via a keyword; chat echoes the context."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "forward" in t.lower() else [0.0, 1.0] for t in texts]

    def chat(self, system: str, user: str, **_) -> str:
        return "Resposta baseada nos jogadores recuperados."


def test_vector_store_build_and_search():
    docs = [
        {"player_id": 1, "name": "Striker", "text": "a lethal forward"},
        {"player_id": 2, "name": "Defender", "text": "a solid centre back"},
    ]
    store = VectorStore.build(docs, _FakeClient())
    assert len(store) == 2
    # Query embeds to the "forward" vector -> Striker ranks first.
    hits = store.search([1.0, 0.0], k=2)
    assert hits[0].name == "Striker"
    assert hits[0].score >= hits[1].score


def test_assistant_answer_grounds_and_cites():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore([1, 2], ["Striker", "Defender"],
                        ["a lethal forward", "a solid centre back"], vectors)
    result = ScoutAssistant(store, client=_FakeClient()).answer("who is a good forward?", k=1)
    assert "Resposta" in result["answer"]
    assert result["sources"][0]["name"] == "Striker"
