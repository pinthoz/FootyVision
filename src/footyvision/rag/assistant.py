"""RAG scouting assistant: retrieve relevant player profiles, then answer grounded in them."""
from __future__ import annotations

from typing import Any

from footyvision.llm.client import LLMClient
from footyvision.rag.store import VectorStore


def build_prompt(question: str, hits: list) -> tuple[str, str]:
    system = (
        "És um assistente de scouting de futebol. Respondes em português de Portugal, de "
        "forma concisa. Baseia-te EXCLUSIVAMENTE nos jogadores recuperados abaixo — não "
        "inventes jogadores nem estatísticas. Cita pelo nome os jogadores que usares e, se "
        "nenhum servir, di-lo honestamente."
    )
    context = "\n".join(f"- {h.text}" for h in hits)
    user = f"Pergunta: {question}\n\nJogadores recuperados (contexto):\n{context}\n\nResposta:"
    return system, user


class ScoutAssistant:
    def __init__(self, store: VectorStore, client: LLMClient | None = None) -> None:
        self.store = store
        self.client = client or LLMClient()

    def answer(self, question: str, k: int = 6) -> dict[str, Any]:
        query_vector = self.client.embed([question])[0]
        hits = self.store.search(query_vector, k=k)
        system, user = build_prompt(question, hits)
        answer = self.client.chat(system, user, max_tokens=1400)
        return {
            "answer": answer,
            "sources": [{"player_id": h.player_id, "name": h.name, "score": round(h.score, 3)}
                        for h in hits],
        }
