"""Thin async wrapper for structured vision calls: JSON-schema output, retries, usage accounting."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import errors, types

from app.pipeline.ledger import Ledger

log = logging.getLogger("gemini")

RETRYABLE = {429, 500, 502, 503, 504}


class QuotaExhausted(RuntimeError):
    """The project's daily request quota for the model is used up; retrying will not help."""


def image_part(jpeg: bytes) -> types.Part:
    return types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")


class Vision:
    def __init__(self, api_key: str, model: str, ledger: Ledger, max_concurrency: int = 6):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.ledger = ledger
        self.sem = asyncio.Semaphore(max_concurrency)

    async def json(
        self,
        stage: str,
        parts: list[Any],
        schema: dict[str, Any],
        system: str | None = None,
        thinking: str = "low",
        media_resolution: str = "high",
    ) -> Any:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=schema,
            media_resolution=getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{media_resolution.upper()}"),
            thinking_config=types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, thinking.upper())),
        )
        contents = [types.Content(role="user", parts=[p if isinstance(p, types.Part) else types.Part(text=p) for p in parts])]
        for attempt in range(6):
            try:
                async with self.sem:
                    resp = await self.client.aio.models.generate_content(model=self.model, contents=contents, config=config)
                self.ledger.add_gemini(stage, self.model, resp.usage_metadata)
                return json.loads(resp.text)
            except errors.APIError as e:
                if getattr(e, "code", None) == 402:
                    raise QuotaExhausted(f"Gemini prepaid credit is used up ({self.model}): top up billing in AI Studio") from e
                if getattr(e, "code", None) == 429 and "PerDay" in str(e):
                    raise QuotaExhausted(
                        f"daily request quota for {self.model} is exhausted (free tier allows ~20 requests/day per model); "
                        "enable billing on the Gemini project or set VISION_MODEL to another model"
                    ) from e
                if getattr(e, "code", None) in RETRYABLE and attempt < 5:
                    delay = min(60, 2 ** attempt * 2)
                    log.warning("%s: %s, retrying in %ss", stage, e.code, delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            except json.JSONDecodeError:
                if attempt < 2:
                    continue
                raise
