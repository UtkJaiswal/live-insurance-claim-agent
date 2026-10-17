"""Loads a recorded sweep and chooses which frames the vision stages look at."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.capture.quality import FrameQuality, Thresholds
from app.capture.recorder import read_events

MAX_ANALYSED_FRAMES = 90


@dataclass
class Frame:
    i: int  # index among all frames in the sweep
    ref: str
    t: float
    quality: dict[str, Any]
    client: dict[str, Any]
    region: str | None
    sweeping: bool

    def path(self, sweep_dir: Path) -> Path:
        return sweep_dir / self.ref


@dataclass
class SweepData:
    dir: Path
    meta: dict[str, Any]
    events: list[dict[str, Any]]
    frames: list[Frame]
    statements: list[dict[str, Any]] = field(default_factory=list)
    live_items: list[dict[str, Any]] = field(default_factory=list)


def load_sweep(sweep_dir: Path) -> SweepData:
    meta = json.loads((sweep_dir / "meta.json").read_text())
    events = read_events(sweep_dir)
    frames = [
        Frame(i=n, ref=e["ref"], t=e["t"], quality=e.get("quality", {}), client=e.get("client", {}),
              region=e.get("client", {}).get("region"), sweeping=bool(e.get("client", {}).get("sweeping", True)))
        for n, e in enumerate(x for x in events if x["kind"] == "frame")
    ]
    statements, live_items = [], []
    for e in events:
        if e["kind"] == "tool_call" and (e.get("result") or {}).get("ok"):
            if e["name"] == "record_user_statement":
                statements.append({**e["args"], "t": e["t"]})
            elif e["name"] == "log_item":
                live_items.append({**e["args"], "t": e["t"]})
    return SweepData(sweep_dir, meta, events, frames, statements, live_items)


def in_sweep(f: Frame, meta: dict[str, Any]) -> bool:
    start, end = meta.get("sweep_started_t"), meta.get("sweep_ended_t")
    if start is not None and f.t < start:
        return False
    if end is not None and f.t > end:
        return False
    return True


def usable(f: Frame, th: Thresholds = Thresholds()) -> bool:
    q = f.quality
    rot = q.get("rotation_dps")
    if rot is not None and rot > th.fast_pan_dps * 1.5:
        return False
    if q.get("brightness", 255) < th.dark * 0.6:
        return False
    s, cov = q.get("sharpness"), q.get("edge_coverage", 0)
    if s is not None and cov >= th.min_texture and s < th.blur * 0.85:
        return False
    return True


GAP_FILL_S = 3.0


def select_frames(sweep: SweepData, max_frames: int = MAX_ANALYSED_FRAMES, prefer: set[int] | None = None) -> list[Frame]:
    """Usable in-sweep frames; if there are too many, the sharpest frame per equal time window.

    `prefer` holds frames already analysed live during the sweep. They are kept, and other frames are
    added only where no preferred frame lies within GAP_FILL_S seconds, so post-sweep work is small.
    """
    cands = [f for f in sweep.frames if in_sweep(f, sweep.meta) and usable(f)]
    if prefer:
        chosen = [f for f in cands if f.i in prefer]
        times = [f.t for f in chosen]
        for f in sorted(cands, key=lambda f: -(f.quality.get("sharpness") or 0)):
            if len(chosen) >= max_frames:
                break
            if f.i not in prefer and all(abs(f.t - t) > GAP_FILL_S for t in times):
                chosen.append(f)
                times.append(f.t)
        return sorted(chosen, key=lambda f: f.t)
    if len(cands) <= max_frames:
        return cands
    t0, t1 = cands[0].t, cands[-1].t
    width = (t1 - t0) / max_frames
    best: dict[int, Frame] = {}
    for f in cands:
        k = min(int((f.t - t0) / width), max_frames - 1)
        if k not in best or (f.quality.get("sharpness") or 0) > (best[k].quality.get("sharpness") or 0):
            best[k] = f
    return [best[k] for k in sorted(best)]


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


def jpeg_bytes(img: np.ndarray, max_side: int | None = None, quality: int = 90) -> bytes:
    if max_side is not None:
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


__all__ = ["Frame", "SweepData", "load_sweep", "select_frames", "read_image", "jpeg_bytes", "FrameQuality"]
