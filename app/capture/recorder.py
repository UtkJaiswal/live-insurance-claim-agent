"""Writes everything a sweep produces to disk so the offline pipeline can be replayed and audited.

Layout of sweeps/<id>/:
  meta.json          sweep id, device, locale, start/end times, camera settings
  frames/f_000123.jpg  high-resolution keyframes (these are what frame_ref points to)
  events.jsonl       one JSON object per event: frame, transcript, tool call, nudge, usage, ...
  audio_in.wav       user microphone (16 kHz mono)
  audio_out.wav      agent voice (24 kHz mono)
"""

from __future__ import annotations

import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SweepRecorder:
    def __init__(self, sweeps_dir: Path, sweep_id: str | None = None):
        self.id = sweep_id or datetime.now().strftime("sw_%Y%m%d_%H%M%S")
        self.dir = sweeps_dir / self.id
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.monotonic()
        self._frame_seq = 0
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        self._audio_in = open(self.dir / "audio_in.pcm", "ab")
        self._audio_out = open(self.dir / "audio_out.pcm", "ab")
        self.meta: dict[str, Any] = {
            "id": self.id,
            "captured_at": utc_now(),
            "device": "",
            "country": "",
            "currency": "",
            "sweep_started_t": None,
            "sweep_ended_t": None,
        }
        self._closed = False
        self.write_meta()

    def t(self) -> float:
        """Seconds since the session opened. All event timestamps use this clock."""
        return round(time.monotonic() - self._t0, 3)

    def log(self, kind: str, **data: Any) -> None:
        if self._closed:
            return
        self._events.write(json.dumps({"t": self.t(), "kind": kind, **data}, default=str) + "\n")

    def save_keyframe(self, jpeg: bytes, client_meta: dict[str, Any], quality: dict[str, Any]) -> str:
        """Saves a high-res frame and returns its frame_ref (path relative to the sweep dir)."""
        self._frame_seq += 1
        name = f"f_{self._frame_seq:06d}.jpg"
        (self.frames_dir / name).write_bytes(jpeg)
        ref = f"frames/{name}"
        self.log("frame", ref=ref, client=client_meta, quality=quality)
        return ref

    @property
    def frame_count(self) -> int:
        return self._frame_seq

    def audio_in(self, pcm: bytes) -> None:
        if not self._closed:
            self._audio_in.write(pcm)

    def audio_out(self, pcm: bytes) -> None:
        if not self._closed:
            self._audio_out.write(pcm)

    def write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._events.close()
        self._audio_in.close()
        self._audio_out.close()
        _pcm_to_wav(self.dir / "audio_in.pcm", 16000)
        _pcm_to_wav(self.dir / "audio_out.pcm", 24000)
        self.meta["closed_t"] = self.t()
        self.write_meta()


def _pcm_to_wav(pcm_path: Path, rate: int) -> None:
    data = pcm_path.read_bytes()
    with wave.open(str(pcm_path.with_suffix(".wav")), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
    pcm_path.unlink()


def read_events(sweep_dir: Path) -> list[dict[str, Any]]:
    path = sweep_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
