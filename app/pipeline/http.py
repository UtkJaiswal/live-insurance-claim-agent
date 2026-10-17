"""Cached HTTP for catalogue, price and FX lookups.

Every response is stored on disk with the time it was retrieved. Re-running a sweep re-uses the
stored answer (and its original retrieval date), so tuning does not spend search credits and the
packet can always show exactly what a source said and when.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.pipeline.ledger import Ledger


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CachedHTTP:
    def __init__(self, cache_dir: Path, ledger: Ledger, timeout: float = 20.0):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": "live-room-claims/0.1"})

    def _path(self, key: str) -> Path:
        return self.dir / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    async def request_json(
        self,
        stage: str,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        headers: dict | None = None,
        usd_per_call: float = 0.0,
        max_age_days: float | None = None,
    ) -> dict[str, Any]:
        """Returns {"data": <parsed JSON>, "retrieved_at": iso, "url": url, "cached": bool}."""
        key = json.dumps([method, url, params, json_body], sort_keys=True)
        path = self._path(key)
        if path.exists():
            entry = json.loads(path.read_text())
            fresh = True
            if max_age_days is not None:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(entry["retrieved_at"])
                fresh = age.total_seconds() < max_age_days * 86400
            if fresh:
                self.ledger.add_http(stage, cached=True)
                return {**entry, "cached": True}
        resp = await self.client.request(method, url, params=params, json=json_body, headers=headers)
        resp.raise_for_status()
        entry = {"key": key, "url": str(resp.url), "retrieved_at": utc_now(), "data": resp.json()}
        path.write_text(json.dumps(entry))
        self.ledger.add_http(stage, cached=False, usd=usd_per_call)
        return {**entry, "cached": False}

    async def close(self) -> None:
        await self.client.aclose()
