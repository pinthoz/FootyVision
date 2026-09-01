"""A tiny in-memory vector store (cosine retrieval) over player-profile embeddings.

For ~400 players a numpy matrix is more than enough — no external vector DB needed. Vectors
are persisted to a .npz so the API doesn't re-embed on every start. pgvector/FAISS would be
the swap-in at much larger scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from footyvision.llm.client import LLMClient


@dataclass
class Hit:
    player_id: int
    name: str
    text: str
    score: float


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self, ids, names, texts, vectors: np.ndarray) -> None:
        self.ids = np.asarray(ids)
        self.names = np.asarray(names, dtype=object)
        self.texts = np.asarray(texts, dtype=object)
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, docs: list[dict], client: LLMClient, batch_size: int = 32) -> VectorStore:
        texts = [d["text"] for d in docs]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(client.embed(texts[i : i + batch_size]))
        return cls(
            [d["player_id"] for d in docs],
            [d["name"] for d in docs],
            texts,
            np.asarray(vectors, dtype=np.float32),
        )

    def search(self, query_vector: list[float], k: int = 6) -> list[Hit]:
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q
        top = np.argsort(scores)[::-1][:k]
        return [
            Hit(int(self.ids[i]), str(self.names[i]), str(self.texts[i]), float(scores[i]))
            for i in top
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=self.ids, names=self.names, texts=self.texts, vectors=self.vectors)

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        data = np.load(path, allow_pickle=True)
        return cls(data["ids"], data["names"], data["texts"], data["vectors"])
