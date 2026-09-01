"""Grounded scouting reports: the LLM explains numbers we computed, never invents them.

Pipeline: feature frame -> radar percentiles + similar players -> a compact factual
context -> a prompt that forbids inventing stats -> the local LLM writes the prose.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from footyvision.llm.client import LLMClient
from footyvision.ml.features import load_feature_frame
from footyvision.ml.similarity import find_similar, radar_percentiles

# Human-readable metric labels for the report context.
READABLE: dict[str, str] = {
    "goals_per90": "golos/90",
    "assists_per90": "assistências/90",
    "shots_per90": "remates/90",
    "xg_per90": "xG/90",
    "passes_per90": "passes/90",
    "passes_completed_per90": "passes completos/90",
    "progressive_passes_per90": "passes progressivos/90",
    "dribbles_per90": "dribles/90",
    "dribbles_completed_per90": "dribles completos/90",
    "carries_per90": "conduções/90",
    "progressive_carries_per90": "conduções progressivas/90",
    "tackles_per90": "desarmes/90",
    "interceptions_per90": "interceções/90",
    "blocks_per90": "bloqueios/90",
    "clearances_per90": "alívios/90",
    "ball_recoveries_per90": "recuperações/90",
    "pressures_per90": "pressões/90",
}


def assemble_context(
    target: pd.Series,
    group: str,
    metrics: dict[str, dict[str, float]],
    similar: list[dict[str, Any]],
    n_strengths: int = 5,
    n_weaknesses: int = 4,
) -> dict[str, Any]:
    """Turn computed features into a compact, JSON-serialisable scouting context."""
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["percentile"], reverse=True)

    def fmt(items: list[tuple[str, dict[str, float]]]) -> list[dict[str, Any]]:
        return [
            {
                "metric": READABLE.get(k, k),
                "value": round(v["value"], 2),
                "percentile": v["percentile"],
            }
            for k, v in items
        ]

    return {
        "name": target["name"],
        "position": target.get("primary_position"),
        "position_group": group,
        "minutes": int(target["minutes"]),
        "matches_played": int(target.get("matches_played", 0)),
        "strengths": fmt(ranked[:n_strengths]),
        "weaknesses": fmt(ranked[-n_weaknesses:][::-1]),
        "similar_players": similar,
    }


def build_prompt(context: dict[str, Any]) -> tuple[str, str]:
    """Return (system, user) messages. Percentiles are vs peers in the same position."""
    system = (
        "És um scout de futebol profissional. Escreves relatórios de scouting concisos, "
        "objetivos e em português de Portugal. Baseia-te EXCLUSIVAMENTE nos dados fornecidos "
        "— não inventes números, clubes, idades nem estatísticas que não estejam no contexto. "
        "Os percentis são relativos a jogadores da mesma posição. Estrutura o relatório com as "
        "secções: Resumo, Pontos Fortes, Pontos Fracos, Enquadramento Tático, "
        "Potencial de Desenvolvimento, Risco."
    )

    def lines(items: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"  - {i['metric']}: {i['value']} (percentil {i['percentile']})" for i in items
        )

    sims = (
        ", ".join(f"{s['name']} ({s['similarity']:.2f})" for s in context["similar_players"])
        or "n/d"
    )

    user = (
        f"Jogador: {context['name']}\n"
        f"Posição: {context['position']} (grupo {context['position_group']})\n"
        f"Minutos: {context['minutes']} em {context['matches_played']} jogos\n\n"
        f"Pontos fortes (percentis mais altos):\n{lines(context['strengths'])}\n\n"
        f"Pontos fracos (percentis mais baixos):\n{lines(context['weaknesses'])}\n\n"
        f"Jogadores mais semelhantes (similaridade cosseno): {sims}\n\n"
        "Escreve o relatório de scouting."
    )
    return system, user


def build_report_context(
    session: Session,
    player_id: int,
    min_minutes: float | None = None,
    competition_id: int | None = None,
    season_id: int | None = None,
    top_similar: int = 5,
) -> dict[str, Any] | None:
    """Load data and assemble the factual context, or None if the player is absent."""
    frame = load_feature_frame(session, min_minutes, competition_id, season_id)
    radar = radar_percentiles(frame, player_id)
    if radar is None:
        return None
    target, group, metrics = radar

    sim = find_similar(frame, player_id, top_n=top_similar)
    similar: list[dict[str, Any]] = []
    if sim is not None:
        _, results = sim
        similar = [
            {"name": r["name"], "similarity": round(float(r["similarity"]), 2)}
            for _, r in results.iterrows()
        ]
    return assemble_context(target, group, metrics, similar)


def generate_report(
    session: Session,
    player_id: int,
    client: LLMClient | None = None,
    **context_kwargs: Any,
) -> dict[str, Any] | None:
    """Full pipeline: context + LLM prose. Returns {context, report} or None if absent."""
    context = build_report_context(session, player_id, **context_kwargs)
    if context is None:
        return None
    client = client or LLMClient()
    system, user = build_prompt(context)
    # Generous budget: a 6-section report plus a reasoning model's hidden thinking.
    report = client.chat(system, user, max_tokens=3000)
    return {"context": context, "report": report}
