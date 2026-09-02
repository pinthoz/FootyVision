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
            rows.append(
                {
                    "player_id": pid,
                    "name": f"Player {pid}",
                    "competition": "La Liga",
                    "primary_position": position,
                    "position_group": position_group(position),
                    "minutes": 1800,
                    "matches_played": 20,
                    **feats,
                }
            )
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

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        self.last_kind = kind
        return [[1.0, 0.0] if "forward" in t.lower() else [0.0, 1.0] for t in texts]

    def chat(self, system: str, user: str, **_) -> str:
        return "Answer grounded in the retrieved players."


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
    store = VectorStore(
        [1, 2], ["Striker", "Defender"], ["a lethal forward", "a solid centre back"], vectors
    )
    result = ScoutAssistant(store, client=_FakeClient()).answer("who is a good forward?", k=1)
    assert "Answer" in result["answer"]
    assert result["sources"][0]["name"] == "Striker"


def test_profiles_carry_the_numbers_behind_each_phrase():
    # Without values the assistant cannot answer "who has the most xG?" — every profile
    # collapses to the same stock phrases.
    docs = build_profiles(_frame())
    forward = next(d for d in docs if "Center Forward" in d["text"])
    assert "0.60 xG per 90" in forward["text"]
    assert "percentile" in forward["text"]


def test_store_pins_players_named_in_the_question():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore(
        [1, 2],
        ["Gareth Frank Bale", "Cédric Bakambu"],
        ["a winger", "a striker"],
        vectors,
    )
    # Accent-insensitive, and short words in the question must not match anything.
    named = {h.name for h in store.mentioned("most xG like Bale and bakambu?")}
    assert named == {"Gareth Frank Bale", "Cédric Bakambu"}
    assert store.mentioned("who is the best midfielder?") == []


def test_assistant_retrieves_named_players_even_when_embeddings_disagree():
    # Both stored vectors are orthogonal to the query embedding, so pure similarity
    # search would surface whichever happens to rank first — not necessarily the two
    # players the question is about.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore(
        [1, 2, 3],
        ["Gareth Frank Bale", "Cédric Bakambu", "Someone Else"],
        ["a winger", "a striker", "a defender"],
        np.vstack([vectors, [[0.0, 1.0]]]).astype(np.float32),
    )
    result = ScoutAssistant(store, client=_FakeClient()).answer("xG like Bale and Bakambu?", k=1)
    names = {s["name"] for s in result["sources"]}
    assert {"Gareth Frank Bale", "Cédric Bakambu"} <= names


def test_named_players_pull_in_stylistic_neighbours_not_similar_names():
    # Two forwards share a style vector; the third is a defender whose *name* is close to
    # the ones in the question. Retrieval must follow style, not spelling.
    vectors = np.array(
        [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]],
        dtype=np.float32,
    )
    store = VectorStore(
        [1, 2, 3],
        ["Cédric Bakambu", "Another Forward", "Alhassane Bangoura"],
        ["a striker", "another striker", "a defender"],
        vectors,
    )
    result = ScoutAssistant(store, client=_FakeClient()).answer("most xG like Bakambu?", k=2)
    names = [s["name"] for s in result["sources"]]
    assert names[0] == "Cédric Bakambu"
    assert "Another Forward" in names
    assert "Alhassane Bangoura" not in names


def test_client_applies_the_query_prefix_to_questions():
    # Retrieval models are trained asymmetrically: a question must carry the query prefix,
    # indexed profiles the document one. Embedding both the same way measurably hurts —
    # see scripts/eval_embeddings.py.
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    store = VectorStore([1], ["Striker"], ["a lethal forward"], vectors)
    client = _FakeClient()
    ScoutAssistant(store, client=client).answer("who scores goals?", k=1)
    assert client.last_kind == "query"
