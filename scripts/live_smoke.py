"""Checks that the configured key and live model work, without a phone.

  uv run python scripts/live_smoke.py

Opens a Live session with the real agent config, asks it to greet, and reports what came back.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402

from app.capture.recorder import SweepRecorder  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.live.bridge import LiveBridge  # noqa: E402


async def main() -> None:
    s = load_settings()
    if not s.gemini_api_key:
        sys.exit("GEMINI_API_KEY is not set (copy .env.example to .env)")
    config = LiveBridge(None, s, SweepRecorder(Path(tempfile.mkdtemp())))._config()  # type: ignore[arg-type]
    client = genai.Client(api_key=s.gemini_api_key)
    audio_bytes, text = 0, []
    async with client.aio.live.connect(model=s.live_model, config=config) as session:
        await session.send_realtime_input(text="The user has opened the app. Greet them.")
        async for msg in session.receive():
            sc = msg.server_content
            if msg.tool_call:
                print("tool call:", [(f.name, f.args) for f in msg.tool_call.function_calls])
                break
            if sc and sc.model_turn:
                audio_bytes += sum(len(p.inline_data.data) for p in sc.model_turn.parts or [] if p.inline_data)
            if sc and sc.output_transcription and sc.output_transcription.text:
                text.append(sc.output_transcription.text)
    print(f"model: {s.live_model}")
    print(f"audio: {audio_bytes / 48000:.1f} s")
    print(f"agent said: {''.join(text).strip()}")


if __name__ == "__main__":
    asyncio.run(main())
