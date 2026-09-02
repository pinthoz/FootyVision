"""A tiny in-memory vector store (cosine retrieval) over player-profile embeddings.

For ~400 players a numpy matrix is more than enough — no external vector DB needed. Vectors
are persisted to a .npz so the API doesn't re-embed on every start. pgvector/FAISS would be
the swap-in at much larger scale.
"""

from __future__ import annotations

import re
import unicodedata
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


def _name_tokens(text: str) -> list[str]:
    """Lowercase, accent-stripped word tokens — so "Bakambu" matches "Cédric Bakambu"."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", folded)


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

    def mentioned(self, question: str) -> list[Hit]:
        """Players named in the question, matched on a distinctive part of their name.

        Pure embedding search does not reliably retrieve the players a question names —
        "like Bale and Bakambu" can miss both. Pinning them is the lexical half of a
        hybrid retrieval: cheap, exact, and it makes comparisons actually answerable.
        """
        asked = set(_name_tokens(question))
        hits: list[Hit] = []
        for i, name in enumerate(self.names):
            # Skip very short tokens ("de", "da") — only distinctive ones identify a player.
            if any(tok in asked for tok in _name_tokens(str(name)) if len(tok) >= 4):
                hits.append(Hit(int(self.ids[i]), str(name), str(self.texts[i]), 1.0))
        return hits

    def style_centroid(self, player_ids: list[int]) -> np.ndarray | None:
        """The mean profile vector of the given players — a "players like these" query.

        Embedding the question text instead makes proper nouns dominate the vector, which
        retrieves players with similar-*looking names* rather than a similar playing style.
        """
        mask = np.isin(self.ids, player_ids)
        if not mask.any():
            return None
        centroid = self.vectors[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm else centroid

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
