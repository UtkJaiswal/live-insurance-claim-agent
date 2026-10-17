"""Builds a simulated sweep from one high-resolution photo, in the same on-disk format as a real one.

  uv run python scripts/make_test_sweep.py testdata/source/philosophy_shelf.jpg --frames 6

A portrait 2160x3840 "camera" pans left to right across the photo with ~55% overlap and small
random rotation, scale and perspective jitter, as a hand-held phone would. Used to test detection,
spine reading and de-duplication before a real library sweep exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.capture.quality import assess_jpeg  # noqa: E402

OUT_W, OUT_H = 2160, 3840


def make_frames(src: np.ndarray, n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    H, W = src.shape[:2]
    win_h = H
    win_w = int(round(win_h * OUT_W / OUT_H))
    if win_w > W:
        win_w = W
        win_h = int(round(win_w * OUT_H / OUT_W))
    xs = np.linspace(0, W - win_w, n)
    frames = []
    for x in xs:
        cx, cy = x + win_w / 2, H / 2
        s = rng.uniform(0.96, 1.04)
        ang = np.deg2rad(rng.uniform(-2, 2))
        hw, hh = win_w / 2 / s, win_h / 2 / s
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        corners = corners @ rot.T + [cx, cy]
        corners += rng.normal(0, win_w * 0.01, corners.shape)  # mild perspective wobble
        dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        frames.append(cv2.warpPerspective(src, M, (OUT_W, OUT_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE))
    return frames


def write_sweep(frames: list[np.ndarray], out_dir: Path, sweep_id: str, source: str) -> Path:
    d = out_dir / sweep_id
    (d / "frames").mkdir(parents=True, exist_ok=True)
    events = []
    for k, img in enumerate(frames, 1):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpeg = buf.tobytes()
        ref = f"frames/f_{k:06d}.jpg"
        (d / ref).write_bytes(jpeg)
        q = assess_jpeg(jpeg, 5.0).as_dict()
        t = 10.0 + 2.0 * k
        events.append({"t": t, "kind": "frame", "ref": ref, "client": {"w": OUT_W, "h": OUT_H, "rotation_dps": 5.0, "sweeping": True}, "quality": q})
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    meta = {
        "id": sweep_id, "captured_at": "2026-10-17T12:00:00+00:00", "device": f"simulated from {source}",
        "country": "IN", "currency": "INR", "sweep_started_t": 10.0, "sweep_ended_t": 10.0 + 2.0 * (len(frames) + 1),
        "frame_count": len(frames), "simulated": True,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("testdata/sweeps"))
    ap.add_argument("--id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = cv2.imread(str(args.source))
    frames = make_frames(src, args.frames, args.seed)
    d = write_sweep(frames, args.out, args.id or f"sim_{args.source.stem}", args.source.name)
    print(d)


if __name__ == "__main__":
    main()
