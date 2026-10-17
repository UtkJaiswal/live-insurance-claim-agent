"""Per-frame capture quality and the monitor that turns persistent problems into spoken nudges.

Sharpness is the mean |Laplacian| / mean |gradient| over strong-edge pixels. Blur spreads an edge
out, so its second derivative falls faster than its first; the ratio is close to independent of
how much texture the scene has. Frames with too few strong edges (a plain wall) get no verdict:
variance-of-Laplacian scored those as "blurry" and caused false nudges in the first real sweep.

The checks are cheap (one 640-px grayscale pass) so they keep up with the stream. Raw scores are
recorded with every frame so thresholds can be recalibrated from real sweeps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np

ANALYSIS_WIDTH = 640
EDGE_MAGNITUDE = 60.0  # Sobel magnitude that counts as a strong edge at ANALYSIS_WIDTH


@dataclass
class FrameQuality:
    sharpness: float | None  # edge-sharpness ratio; None when there is too little texture to judge
    edge_coverage: float  # share of pixels on strong edges (spines give lots, plain walls almost none)
    glare_frac: float  # share of near-saturated pixels
    brightness: float  # mean grey level 0-255
    rotation_dps: float | None  # phone rotation rate from the gyro, degrees/second

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def assess_jpeg(jpeg: bytes, rotation_dps: float | None = None) -> FrameQuality:
    # Half-resolution decode is several times faster on 4K frames; we downscale to 640 px anyway.
    grey = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if grey is None:
        raise ValueError("not a decodable image")
    return assess_grey(grey, rotation_dps)


def assess_grey(grey: np.ndarray, rotation_dps: float | None = None) -> FrameQuality:
    h, w = grey.shape[:2]
    if w != ANALYSIS_WIDTH:
        grey = cv2.resize(grey, (ANALYSIS_WIDTH, int(h * ANALYSIS_WIDTH / w)), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    edges = mag > EDGE_MAGNITUDE
    coverage = float(edges.mean())
    sharpness = None
    if edges.sum() >= 200:
        lap = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
        sharpness = float(lap[edges].mean() / mag[edges].mean())
    return FrameQuality(
        sharpness=sharpness,
        edge_coverage=coverage,
        glare_frac=float((grey >= 250).mean()),
        brightness=float(grey.mean()),
        rotation_dps=rotation_dps,
    )


@dataclass(frozen=True)
class Thresholds:
    blur: float = 0.10  # sharpness ratio below this is blurred (calibrated on sweep sw_20261017_155806)
    min_texture: float = 0.04  # edge coverage needed before judging blur
    glare: float = 0.03
    dark: float = 45.0
    fast_pan_dps: float = 45.0


# What the agent is told for each issue. The agent rephrases it in its own voice.
ISSUE_HINTS = {
    "fast_pan": "The camera is panning too fast; spines are smeared. Ask the user to slow down and hold steady on each shelf.",
    "blur": "Recent frames are blurry. Ask the user to hold the phone steady, or step a little closer so the camera can focus.",
    "glare": "There is strong glare or reflection in recent frames. Ask the user to tilt the phone slightly or change angle.",
    "dark": "Recent frames are too dark to read spines. Ask the user to turn on a light or open a curtain.",
}

REPEAT_RESET_S = 45.0  # an issue not nudged for this long starts again at the base cooldown
MAX_COOLDOWN_S = 40.0


class CaptureMonitor:
    """Flags an issue when most frames in a short window share it.

    A global cooldown separates any two nudges, and each repeat of the same issue doubles its own
    cooldown, so a user who cannot fix something is not told the same thing every few seconds.
    """

    def __init__(self, thresholds: Thresholds, window: int = 4, cooldown_s: float = 6.0):
        self.th = thresholds
        self.window: deque[list[str]] = deque(maxlen=window)
        self.cooldown_s = cooldown_s
        self._last_any_t = -1e9
        self._last_t: dict[str, float] = {}
        self.repeats: dict[str, int] = {}

    def issues(self, q: FrameQuality) -> list[str]:
        out = []
        if q.rotation_dps is not None and q.rotation_dps > self.th.fast_pan_dps:
            out.append("fast_pan")
        elif q.sharpness is not None and q.edge_coverage >= self.th.min_texture and q.sharpness < self.th.blur:
            out.append("blur")
        if q.glare_frac > self.th.glare:
            out.append("glare")
        if q.brightness < self.th.dark:
            out.append("dark")
        return out

    def _ready(self, issue: str, t: float) -> bool:
        last = self._last_t.get(issue)
        if last is None or t - last > REPEAT_RESET_S:
            self.repeats[issue] = 0
            return True
        gap = min(self.cooldown_s * 2 ** self.repeats.get(issue, 0), MAX_COOLDOWN_S)
        return t - last >= gap

    def observe(self, q: FrameQuality, t: float) -> str | None:
        """Returns the issue to nudge about now, or None. self.repeats[issue] says how many times before."""
        self.window.append(self.issues(q))
        if len(self.window) < self.window.maxlen or t - self._last_any_t < self.cooldown_s:
            return None
        counts: dict[str, int] = {}
        for frame_issues in self.window:
            for issue in frame_issues:
                counts[issue] = counts.get(issue, 0) + 1
        majority = len(self.window) // 2 + 1
        for issue in ISSUE_HINTS:
            if counts.get(issue, 0) >= majority and self._ready(issue, t):
                if issue in self._last_t and t - self._last_t[issue] <= REPEAT_RESET_S:
                    self.repeats[issue] = self.repeats.get(issue, 0) + 1
                self._last_t[issue] = t
                self._last_any_t = t
                self.window.clear()
                return issue
        return None
