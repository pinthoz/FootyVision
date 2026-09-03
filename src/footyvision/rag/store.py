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
from footyvision.rag.constraints import Constraints


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


# Ordinary words that also happen to be surnames. Without this, "a **young** winger"
# pins Ashley Young, and because the assistant then queries with the centroid of the
# pinned players instead of the question, one accidental collision hijacks the whole
# retrieval. Filtering the *question* side is safe: someone actually asking about Ashley
# Young still writes "Ashley", which is distinctive and pins him on its own.
_COMMON_WORDS: frozenset[str] = frozenset(
    """
    young older elder best good great strong quick fast pace power powerful
    back backs wing wings wide side sides ball balls foot feet left right
    play plays player players playing style forward forwards striker
    winger wingers defender defenders midfield midfielder keeper goalkeeper
    find give need want show tell like similar compare best most more less than
    that this they them with from what which have take takes make makes
    someone somebody anyone scoring passing tackling dribbling pressing
    creative defensive attacking complete season minutes team teams club
    """.split()
)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self, ids, names, texts, vectors: np.ndarray, attrs: dict | None = None) -> None:
        self.ids = np.asarray(ids)
        self.names = np.asarray(names, dtype=object)
        self.texts = np.asarray(texts, dtype=object)
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        # Hard attributes kept beside the vectors so retrieval can narrow the pool before
        # ranking it. An index built before these existed simply has none, and every
        # filter then matches everything rather than failing.
        attrs = attrs or {}
        n = len(self.ids)
        self.foot = np.asarray(attrs.get("foot", [None] * n), dtype=object)
        self.age = np.asarray(attrs.get("age", [np.nan] * n), dtype=np.float32)
        self.position_group = np.asarray(attrs.get("position_group", [None] * n), dtype=object)

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
            attrs={
                "foot": [d.get("foot") for d in docs],
                "age": [d.get("age", np.nan) for d in docs],
                "position_group": [d.get("position_group") for d in docs],
            },
        )

    def mentioned(self, question: str) -> list[Hit]:
        """Players named in the question, matched on a distinctive part of their name.

        Pure embedding search does not reliably retrieve the players a question names —
        "like Bale and Bakambu" can miss both. Pinning them is the lexical half of a
        hybrid retrieval: cheap, exact, and it makes comparisons actually answerable.
        """
        asked = set(_name_tokens(question)) - _COMMON_WORDS
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

    def matching(self, constraints: Constraints) -> np.ndarray:
        """Boolean mask of the players satisfying every hard constraint.

        Two rules, and the difference between them matters:

        * A *player* whose attribute is unknown fails the constraint. Saying "he is
          left-footed" of someone whose foot was never recorded is exactly the invention
          the assistant exists to avoid.
        * A *dimension* the index knows nothing about is not a filter at all. An index
          built before these attributes existed would otherwise exclude everybody and
          leave the assistant answering every positional question with silence.
        """
        mask = np.ones(len(self.ids), dtype=bool)
        if constraints.foot is not None and self._knows(self.foot):
            mask &= self.foot == constraints.foot
        if constraints.position_group is not None and self._knows(self.position_group):
            mask &= self.position_group == constraints.position_group
        if self._knows(self.age):
            if constraints.max_age is not None:
                mask &= np.nan_to_num(self.age, nan=np.inf) <= constraints.max_age
            if constraints.min_age is not None:
                mask &= np.nan_to_num(self.age, nan=-np.inf) >= constraints.min_age
        return mask

    @staticmethod
    def _knows(values: np.ndarray) -> bool:
        """Whether the index recorded this attribute for anyone at all."""
        if values.dtype == object:
            return any(v is not None for v in values)
        return bool(np.isfinite(values).any())

    def search(
        self,
        query_vector: list[float],
        k: int = 6,
        mask: np.ndarray | None = None,
    ) -> list[Hit]:
        """Rank by cosine similarity, optionally within a pre-filtered subset.

        Filtering before ranking rather than after is the point: an embedding cannot
        enforce "left-footed", so asking it to rank the whole pool and hoping the
        constraint survives the top-k does not work.
        """
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q
        candidates = np.arange(len(self.ids)) if mask is None else np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(scores[candidates])[::-1][:k]]
        return [
            Hit(int(self.ids[i]), str(self.names[i]), str(self.texts[i]), float(scores[i]))
            for i in order
        ]

    def save_db(self, session, embed_model: str) -> int:
        """Replace the stored index with this one, in a single transaction."""
        from footyvision.db.models import PlayerVector

        session.query(PlayerVector).delete()
        session.bulk_save_objects(
            [
                PlayerVector(
                    player_id=int(self.ids[i]),
                    name=str(self.names[i]),
                    text=str(self.texts[i]),
                    vector=self.vectors[i].astype(np.float32).tobytes(),
                    dim=int(self.vectors.shape[1]),
                    embed_model=embed_model,
                    foot=self.foot[i] if self.foot[i] else None,
                    age=None if not np.isfinite(self.age[i]) else float(self.age[i]),
                    position_group=self.position_group[i] if self.position_group[i] else None,
                )
                for i in range(len(self.ids))
            ]
        )
        session.commit()
        return len(self.ids)

    @classmethod
    def load_db(cls, session) -> tuple[VectorStore, str] | None:
        """Read the stored index back, with the name of the model that built it."""
        from footyvision.db.models import PlayerVector

        rows = session.query(PlayerVector).all()
        if not rows:
            return None
        vectors = np.vstack(
            [np.frombuffer(r.vector, dtype=np.float32).reshape(1, r.dim) for r in rows]
        )
        store = cls(
            [r.player_id for r in rows],
            [r.name for r in rows],
            [r.text for r in rows],
            vectors,
            attrs={
                "foot": [r.foot for r in rows],
                "age": [np.nan if r.age is None else r.age for r in rows],
                "position_group": [r.position_group for r in rows],
            },
        )
        return store, rows[0].embed_model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ids=self.ids,
            names=self.names,
            texts=self.texts,
            vectors=self.vectors,
            foot=self.foot,
            age=self.age,
            position_group=self.position_group,
        )

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        data = np.load(path, allow_pickle=True)
        # Indexes written before the attributes existed have none; degrade, do not crash.
        attrs = {k: data[k] for k in ("foot", "age", "position_group") if k in data.files}
        return cls(data["ids"], data["names"], data["texts"], data["vectors"], attrs)
