"""Replays a recorded sweep into the live server, as the phone would send it.

  uv run python scripts/replay_live.py sweeps/<recorded id> [--url ws://127.0.0.1:8000/ws/live]

Streams the recorded microphone audio (16 kHz PCM) in 100 ms chunks, every recorded keyframe with its
original motion header at its original time, and a 1 fps preview for the live model. Ends the sweep
when the recording ends (if the agent has not already), then waits for the claim packet and prints
what the agent said. An end-to-end test of the live agent, the background scanner and the pipeline
without walking a room.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import websockets

MSG_AUDIO, MSG_PREVIEW, MSG_KEYFRAME = 1, 2, 3


def preview(jpeg_path: Path) -> bytes:
    img = cv2.imread(str(jpeg_path))
    s = 768 / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/live")
    ap.add_argument("--packet-timeout", type=float, default=420)
    args = ap.parse_args()
    d = args.sweep_dir
    events = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines() if line.strip()]
    meta = json.loads((d / "meta.json").read_text())
    with wave.open(str(d / "audio_in.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    frames = [e for e in events if e["kind"] == "frame"]
    end_t = max(len(audio) / 16000, frames[-1]["t"] if frames else 0)
    agent_text, state = [], {"ended": False, "packet": None, "audio_s": 0.0}

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "device": f"replay of {d.name}", "camera": meta.get("camera", {})}))

        async def receive():
            async for msg in ws:
                if isinstance(msg, bytes):
                    state["audio_s"] += len(msg) / 2 / 24000
                    continue
                ev = json.loads(msg)
                if ev["type"] == "transcript" and ev["role"] == "agent":
                    agent_text.append(ev["text"])
                elif ev["type"] == "turn_complete" and agent_text:
                    print("AGENT:", "".join(agent_text).strip())
                    agent_text.clear()
                elif ev["type"] == "sweep" and ev["state"] == "ended":
                    state["ended"] = True
                    print(f"-- sweep ended at {ev['t']:.1f}s")
                elif ev["type"] in ("status", "nudge", "item", "region", "statement", "locale", "scan"):
                    print(f"   [{ev['type']}] " + json.dumps({k: v for k, v in ev.items() if k != 'type'}, ensure_ascii=False)[:160])
                elif ev["type"] == "packet":
                    state["packet"] = ev
                    print(f"-- PACKET in {ev['time_to_packet_s']} s: {ev['report']}")

        rx = asyncio.create_task(receive())
        t0 = time.monotonic()
        a_pos, f_idx, next_preview = 0, 0, 0.0
        while True:
            now = time.monotonic() - t0
            if now > end_t:
                break
            while a_pos < len(audio) and a_pos / 16000 <= now:
                await ws.send(bytes([MSG_AUDIO]) + audio[a_pos : a_pos + 1600].tobytes())
                a_pos += 1600
            while f_idx < len(frames) and frames[f_idx]["t"] <= now:
                fe = frames[f_idx]
                header = json.dumps({k: v for k, v in fe.get("client", {}).items() if k not in ("region", "sweeping")}).encode()
                jpeg = (d / fe["ref"]).read_bytes()
                await ws.send(bytes([MSG_KEYFRAME]) + struct.pack(">I", len(header)) + header + jpeg)
                if now >= next_preview:
                    await ws.send(bytes([MSG_PREVIEW]) + preview(d / fe["ref"]))
                    next_preview = now + 1.0
                f_idx += 1
            await asyncio.sleep(0.02)
        silence = np.zeros(1600, np.int16).tobytes()
        if not state["ended"]:
            print("-- recording finished; pressing End sweep")
            await ws.send(json.dumps({"type": "end_sweep"}))
        deadline = time.monotonic() + args.packet_timeout
        while state["packet"] is None and time.monotonic() < deadline:
            await ws.send(bytes([MSG_AUDIO]) + silence)  # keep the mic stream alive
            await asyncio.sleep(0.1)
        await asyncio.sleep(25)  # let the agent read the summary back
        print(f"-- agent audio received: {state['audio_s']:.1f} s")
        await ws.send(json.dumps({"type": "stop"}))
        rx.cancel()
    if state["packet"] is None:
        sys.exit("no packet within timeout")
    print("SUMMARY:", state["packet"]["summary"])


if __name__ == "__main__":
    asyncio.run(main())
