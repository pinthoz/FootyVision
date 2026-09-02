"""RAG scouting assistant: retrieve relevant player profiles, then answer grounded in them."""

from __future__ import annotations

from typing import Any

from footyvision.llm.client import LLMClient
from footyvision.rag.store import VectorStore


def build_prompt(question: str, hits: list) -> tuple[str, str]:
    system = (
        "You are a football scouting assistant. Answer in English, concisely, even when "
        "the question is asked in another language. Base your answer EXCLUSIVELY on the "
        "retrieved players below — never invent players or statistics. Name the players "
        "you use, and if none of them fit, say so honestly. Every profile carries per-90 "
        "values and percentiles: when the question compares players or asks who is best "
        "at some metric, compare those numbers and justify your pick with them. If the "
        "metric asked about is not in the profiles, say you do not have it rather than "
        "guessing."
    )
    context = "\n".join(f"- {h.text}" for h in hits)
    user = f"Question: {question}\n\nRetrieved players (context):\n{context}\n\nAnswer:"
    return system, user


class ScoutAssistant:
    def __init__(self, store: VectorStore, client: LLMClient | None = None) -> None:
        self.store = store
        self.client = client or LLMClient()

    def answer(self, question: str, k: int = 6) -> dict[str, Any]:
        # Hybrid retrieval: players named in the question are pinned, the rest of the
        # budget is filled by embedding similarity.
        pinned = self.store.mentioned(question)
        # When the question names players, fill the remaining slots with players whose
        # *style* resembles theirs. Embedding the raw question makes the proper nouns
        # dominate, which retrieves similar-sounding names instead of similar players.
        centroid = self.store.style_centroid([h.player_id for h in pinned]) if pinned else None
        query_vector = (
            centroid if centroid is not None else self.client.embed([question], kind="query")[0]
        )
        seen = {h.player_id for h in pinned}
        hits = list(pinned)
        for hit in self.store.search(query_vector, k=k + len(pinned)):
            if len(hits) >= max(k, len(pinned)):
                break
            if hit.player_id not in seen:
                hits.append(hit)
                seen.add(hit.player_id)
        system, user = build_prompt(question, hits)
        answer = self.client.chat(system, user, max_tokens=1400)
        return {
            "answer": answer,
            "sources": [
                {"player_id": h.player_id, "name": h.name, "score": round(h.score, 3)} for h in hits
            ],
        }
