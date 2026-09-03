"""Per-client rate limiting for the endpoints that spend money.

`/assistant`, `/search` and the report endpoint each call an LLM, so an unthrottled
public deployment lets anyone drain the API key attached to it. This is the actual
defence: CORS restricts which *sites* a browser will let call the API, but it does
nothing about curl, so it is a good hygiene measure and a bad security control.

Deliberately in-process and dependency-free. A single Render instance has one process,
so a shared counter is exact there; behind several replicas each would hold its own
window and the effective limit multiplies by the replica count. Redis is the swap-in if
that day comes — the interface here would not change.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from footyvision.config import get_settings

WINDOW_SECONDS = 60.0

# client key -> timestamps of the calls still inside the window.
_HITS: defaultdict[str, deque[float]] = defaultdict(deque)


def client_key(request: Request) -> str:
    """Identify the caller, trusting the proxy header only for its first hop.

    Render and Vercel both terminate TLS in front of the app, so `request.client.host`
    is the proxy and every caller would share one bucket. `X-Forwarded-For` is spoofable
    by a determined client, which is why this throttles cost rather than enforcing
    identity — the leftmost entry is the closest thing to the origin available.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def reset() -> None:
    """Drop every window. Tests need this; nothing in the app calls it."""
    _HITS.clear()


def rate_limit(request: Request) -> None:
    """FastAPI dependency: allow N calls per minute per client, then 429.

    Set `RATE_LIMIT_PER_MINUTE=0` to disable entirely, which is what local development
    and the test suite do.
    """
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return

    now = time.monotonic()
    hits = _HITS[client_key(request)]
    while hits and now - hits[0] > WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= limit:
        retry_after = int(WINDOW_SECONDS - (now - hits[0])) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit of {limit} requests per minute exceeded.",
            headers={"Retry-After": str(retry_after)},
        )

    hits.append(now)
