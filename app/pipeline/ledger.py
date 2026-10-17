"""Cost and latency accounting, per stage. Written to sweeps/<id>/metrics.json.

Prices are paid-tier list prices (USD per 1M tokens) from ai.google.dev/gemini-api/docs/pricing,
retrieved 2026-10-17. The free tier costs nothing; we report what a paid sweep would cost.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

PRICES_PER_M = {
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.6-flash": {"in": 0.75, "out": 3.75},
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50},
    "gemini-3.1-flash-lite": {"in": 0.25, "out": 1.50},
}
LIVE_PRICES_PER_M = {"text_in": 0.75, "audio_in": 3.00, "video_in": 1.00, "image_in": 1.00, "text_out": 4.50, "audio_out": 12.00}
SERPER_USD_PER_QUERY = 0.001  # $50 / 50k-credit pack; the first 2,500 are free


class Ledger:
    def __init__(self) -> None:
        self.stages: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"seconds": 0.0, "calls": 0, "in_tokens": 0, "out_tokens": 0, "usd": 0.0, "http_calls": 0, "cache_hits": 0}
        )

    @contextmanager
    def stage(self, name: str):
        t = time.monotonic()
        try:
            yield
        finally:
            self.stages[name]["seconds"] += time.monotonic() - t

    def add_gemini(self, stage: str, model: str, usage: Any) -> None:
        s = self.stages[stage]
        s["calls"] += 1
        if usage is None:
            return
        tin = usage.prompt_token_count or 0
        tout = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        s["in_tokens"] += tin
        s["out_tokens"] += tout
        p = PRICES_PER_M.get(model, PRICES_PER_M["gemini-3.8-flash"])
        s["usd"] += tin / 1e6 * p["in"] + tout / 1e6 * p["out"]

    def add_http(self, stage: str, cached: bool, usd: float = 0.0) -> None:
        s = self.stages[stage]
        if cached:
            s["cache_hits"] += 1
        else:
            s["http_calls"] += 1
            s["usd"] += usd

    def to_dict(self) -> dict[str, Any]:
        stages = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in self.stages.items()}
        return {"stages": stages, "total_usd": round(sum(v["usd"] for v in self.stages.values()), 4)}


def live_session_cost(events: list[dict]) -> dict[str, Any]:
    """Cost of the live voice session, from the usage records the bridge logged.

    The Live API reports cumulative usage per turn; we take the per-modality token counts of each
    usage message and sum them.
    """
    totals: dict[str, int] = defaultdict(int)
    for e in events:
        if e.get("kind") != "usage":
            continue
        for key, direction in (("prompt_tokens_details", "in"), ("response_tokens_details", "out")):
            for d in e.get(key) or []:
                modality = str(d.get("modality", "TEXT")).lower()
                totals[f"{modality}_{direction}"] += int(d.get("token_count", 0))
    usd = sum(n / 1e6 * LIVE_PRICES_PER_M.get(k, 0.0) for k, n in totals.items())
    return {"tokens": dict(totals), "usd": round(usd, 4)}
