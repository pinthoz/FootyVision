"""Natural-language → structured PlayerQuery via the LLM.

The LLM only ever emits JSON for the PlayerQuery schema; Pydantic validation rejects
anything outside the whitelist. No SQL is ever produced by the model.
"""

from __future__ import annotations

import json

from footyvision.llm.client import LLMClient
from footyvision.search.query import SEARCHABLE_FIELDS, PlayerQuery


class NLParseError(ValueError):
    """The LLM output could not be parsed into a valid PlayerQuery."""


def _fields_help() -> str:
    return ", ".join(sorted(SEARCHABLE_FIELDS))


def build_nl_prompt(text: str) -> tuple[str, str]:
    system = (
        "You translate a football scout's request into a JSON query object. "
        "Output ONLY a JSON object, no prose, no code fences.\n"
        "Schema:\n"
        "{\n"
        '  "position_group": one of "GK"|"DEF"|"MID"|"FWD" or null,\n'
        '  "competition": league-name substring or null,\n'
        '  "min_minutes": number or null,\n'
        '  "conditions": [{"field": <field>, "op": "gt"|"gte"|"lt"|"lte"|"eq", "value": number}],\n'
        '  "order_by": <field> or null,\n'
        '  "order_desc": boolean,\n'
        '  "limit": integer (1-100)\n'
        "}\n"
        f"Allowed <field> values (per-90 unless stated): {_fields_help()}.\n"
        "There is NO age/date-of-birth data, so ignore any age constraint (e.g. 'under 23') "
        "and do not invent a field for it. Map metric names to the closest allowed field "
        "(e.g. 'xG per 90' -> 'xg_per90', 'progressive passes' -> 'progressive_passes_per90')."
    )
    user = f"Request: {text}\nJSON:"
    return system, user


def _extract_json(raw: str) -> str:
    """Pull the first JSON object out of the model output (tolerates fences/prose)."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise NLParseError(f"No JSON object found in LLM output: {raw[:200]!r}")
    return raw[start : end + 1]


def parse_nl(text: str, client: LLMClient | None = None) -> PlayerQuery:
    """Ask the LLM to structure `text`, then validate into a PlayerQuery."""
    client = client or LLMClient()
    system, user = build_nl_prompt(text)
    # Headroom for reasoning models that "think" before emitting the JSON.
    raw = client.chat(system, user, temperature=0.1, max_tokens=1500)
    payload = _extract_json(raw)
    try:
        return PlayerQuery.model_validate(json.loads(payload))
    except (json.JSONDecodeError, ValueError) as exc:
        raise NLParseError(f"Invalid query from LLM: {exc}. Raw: {payload[:200]!r}") from exc
