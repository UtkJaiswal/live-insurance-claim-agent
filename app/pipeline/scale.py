"""Stage 5: metric scale and spine measurement.

Where scale comes from, best first:
  1. A4 sheet (29.7 x 21.0 cm) if one is visible: an explicit reference object.
  2. Identified books whose matched edition lists physical dimensions (catalogue anchors).
  3. Doors: standard leaf height, 203 cm +/- 5% (weak, used only if nothing better exists).
  4. Standard book formats: the median upright book is assumed 21 cm (last resort, low confidence).

Two ways of applying it:
  * With a 3D reconstruction (app/geometry/da3.py), one global factor k converts reconstruction units
    to cm. A box of h pixels at depth z with focal length f spans h*z/f units, so each anchor gives
    k = known_cm / (h*z/f). Every spine and the room are then measured with the same k.
  * Without one, each shelf segment is measured in its reference-frame pixels (track.py) with
    cm-per-pixel = median(known_cm / box_px) over the segment's anchors.

A book's own catalogue height is never used to measure that book (leave-one-out), otherwise the
spine-dimension check against ground truth would be circular.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.detect import Detection
from app.pipeline.identify import Identification
from app.pipeline.track import Track

A4_LONG_CM = 29.7
DOOR_CM = 205.0
FORMAT_PRIOR_CM = 21.0


@dataclass
class Anchor:
    source: str  # a4 | catalogue | door
    ref: str  # detection id or track id
    known_cm: float
    weight: float
    track_id: str | None = None  # set for catalogue anchors (for leave-one-out)


@dataclass
class Measurement:
    track_id: str
    height_cm: float | None = None
    thickness_cm: float | None = None
    method: str = ""
    n_anchors: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScaleModel:
    """k: cm per reconstruction unit (3D mode) or None; per-segment cm per ref-pixel (2D mode)."""

    k_samples: list[tuple[Anchor, float]] = field(default_factory=list)
    seg_samples: dict[int, list[tuple[Anchor, float]]] = field(default_factory=dict)
    method: str = ""

    def k(self, exclude_track: str | None = None) -> tuple[float | None, int]:
        return _robust([(a, v) for a, v in self.k_samples if a.track_id is None or a.track_id != exclude_track])

    def seg(self, s: int, exclude_track: str | None = None) -> tuple[float | None, int]:
        return _robust([(a, v) for a, v in self.seg_samples.get(s, []) if a.track_id is None or a.track_id != exclude_track])


def _robust(samples: list[tuple[Anchor, float]]) -> tuple[float | None, int]:
    if not samples:
        return None, 0
    # An A4 sheet or catalogue anchor outranks doors; use the best tier available.
    best_w = max(a.weight for a, _ in samples)
    vals = np.array([v for a, v in samples if a.weight == best_w])
    med = float(np.median(vals))
    if len(vals) >= 4:  # drop outliers beyond 3 MADs (a mis-matched edition, a leaning book)
        mad = float(np.median(np.abs(vals - med))) or med * 0.02
        vals = vals[np.abs(vals - med) <= 3 * 1.4826 * mad]
        med = float(np.median(vals))
    return med, len(vals)


def touches_edge(d: Detection, margin: float = 0.006) -> bool:
    """True if the box runs into the top or bottom of the frame: its true height is unknown.

    The detector's own 'fully_visible' flag was wrong in testing (a door cut by the frame bottom was
    reported fully visible), so the geometry is checked here. Frame size comes from the detection.
    """
    H = d.attrs.get("_frame_h")
    if not H:
        return False
    return d.box[1] <= margin * H or d.box[3] >= (1 - margin) * H


def spine_px(d: Detection) -> tuple[float, float]:
    """(length along the spine, thickness across it) in pixels. Uses the edge-refined thickness when set."""
    across = d.attrs.get("refined_thickness_px")
    if d.attrs.get("orientation") == "flat":
        return d.w, across or d.h
    return d.h, across or d.w


def refine_thickness(gray: np.ndarray, d: Detection, search: float = 0.15) -> float | None:
    """Snaps the two long edges of a spine box to the strongest image edges near them.

    Detection boxes are a few pixels loose; on a 2 cm spine that is a 20-30% thickness error. The
    spine's sides are strong straight edges parallel to the spine, so the column (or row) profile of
    the gradient across the spine peaks there.
    """
    flat = d.attrs.get("orientation") == "flat"
    x0, y0, x1, y1 = (int(round(v)) for v in d.box)
    if flat:  # work in the transposed image so the spine runs vertically
        gray, (x0, y0, x1, y1) = gray.T, (y0, x0, y1, x1)
    H, W = gray.shape
    w, h = x1 - x0, y1 - y0
    if w < 6 or h < 20:
        return None
    m = int(search * w) + 2
    xa, xb = max(0, x0 - m), min(W, x1 + m)
    ya, yb = max(0, y0 + int(0.2 * h)), min(H, y1 - int(0.2 * h))
    strip = gray[ya:yb, xa:xb].astype(np.float32)
    if strip.shape[0] < 10 or strip.shape[1] < 6:
        return None
    prof = np.abs(np.diff(strip, axis=1)).mean(axis=0)  # mean |d/dx| per column boundary
    base = float(np.median(prof)) + 1e-6

    def peak(lo: int, hi: int) -> int | None:
        lo, hi = max(0, lo - xa), min(len(prof), hi - xa)
        if hi - lo < 2:
            return None
        k = lo + int(np.argmax(prof[lo:hi]))
        return k + xa if prof[k] >= 1.8 * base else None

    left = peak(x0 - m, x0 + m)
    right = peak(x1 - m, x1 + m)
    # Detection boxes err on the loose side, so a valid refinement only ever tightens the box. A wider
    # result means an edge of the neighbouring book was picked up (seen in testing), so it is rejected.
    if left is None or right is None or not (0.6 * w <= right - left <= w):
        return None
    return float(right - left)


def collect_anchors(tracks: list[Track], idents: dict[str, Identification], anchor_dets: list[Detection]) -> list[tuple[Anchor, Detection]]:
    out: list[tuple[Anchor, Detection]] = []
    for d in anchor_dets:
        if not d.attrs.get("fully_visible") or touches_edge(d):
            continue
        if d.attrs.get("kind") == "door" and not 1.6 <= d.h / max(d.w, 1) <= 4.5:
            continue  # not the shape of a door leaf: partly hidden or mis-detected
        if d.attrs.get("kind") == "a4_sheet":
            out.append((Anchor("a4", d.id, A4_LONG_CM, 3.0), d))
        elif d.attrs.get("kind") == "door":
            out.append((Anchor("door", d.id, DOOR_CM, 1.0), d))
    for t in tracks:
        ident = idents.get(t.id)
        h = (ident.dims_cm or {}).get("height") if ident else None
        if not h or t.fragment or t.orientation == "flat":
            continue
        for d in t.dets:
            if not d.attrs.get("cut_off"):
                out.append((Anchor("catalogue", t.id, float(h), 2.0, track_id=t.id), d))
    return out


def build_scale(tracks: list[Track], idents: dict[str, Identification], anchor_dets: list[Detection], recon=None) -> ScaleModel:
    model = ScaleModel()
    anchors = collect_anchors(tracks, idents, anchor_dets)
    if recon is not None:
        # With a reconstruction, doors are measured by the height of their top edge above the floor
        # (app/geometry/room.door_top_scales), not by box height: furniture often hides a door's bottom.
        anchors = [(a, d) for a, d in anchors if a.source != "door"]
    frame_index = {ref: n for n, ref in enumerate(recon.frame_refs)} if recon is not None else {}
    for a, d in anchors:
        length_px = d.h if a.source != "catalogue" else spine_px(d)[0]
        if recon is not None and d.ref in frame_index:
            units = _units(recon, frame_index[d.ref], d, length_px)
            if units:
                model.k_samples.append((a, a.known_cm / units))
    for t in tracks:
        if t.ref_box is None:
            continue
        for a, d in anchors:
            if a.track_id == t.id:
                h_ref = t.ref_box[3] - t.ref_box[1]
                model.seg_samples.setdefault(t.segment, []).append((a, a.known_cm / h_ref))
                break
    # A4 sheets and doors are not tracked. Their frame's cm-per-pixel is converted to the segment's
    # reference-frame pixels through the books seen in that same frame (ref height / frame height).
    for a, d in anchors:
        if a.source == "catalogue":
            continue
        ratios: dict[int, list[float]] = {}
        for t in tracks:
            if t.ref_box is None or t.fragment:
                continue
            for bd in t.dets:
                if bd.frame_i == d.frame_i and not bd.attrs.get("cut_off"):
                    ratios.setdefault(t.segment, []).append((t.ref_box[3] - t.ref_box[1]) / bd.h)
        for seg, rs in ratios.items():
            frame_cm_per_px = a.known_cm / d.h
            model.seg_samples.setdefault(seg, []).append((a, frame_cm_per_px / float(np.median(rs))))
    return model


def _units(recon, i: int, d: Detection, length_px: float) -> float | None:
    """Length in reconstruction units of `length_px` pixels at the depth of the object.

    Books and A4 sheets: median depth over the central 40% of the box. Doors: the box centre can look
    through an open doorway into the next room (seen in testing), so depth comes from the two jambs,
    the outer 12% of the box on each side.
    """
    h, w = recon.proc_size
    H, W = recon.orig_size
    y0 = int((d.box[1] + 0.3 * d.h) * h / H)
    y1 = max(y0 + 1, int((d.box[3] - 0.3 * d.h) * h / H))
    if d.kind == "anchor" and d.attrs.get("kind") == "door":
        strips = [(d.box[0], d.box[0] + 0.12 * d.w), (d.box[2] - 0.12 * d.w, d.box[2])]
    else:
        strips = [(d.box[0] + 0.3 * d.w, d.box[2] - 0.3 * d.w)]
    vals = []
    for a, b in strips:
        x0 = int(a * w / W)
        x1 = max(x0 + 1, int(b * w / W))
        vals.append(recon.depth[i, max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)].ravel())
    patch = np.concatenate(vals) if vals else np.array([])
    if patch.size == 0:
        return None
    z = float(np.median(patch))
    f_orig = float(recon.intrinsics[i][1, 1]) * H / h  # vertical focal length in original pixels
    if z <= 0 or f_orig <= 0:
        return None
    return length_px * z / f_orig


def measure(tracks: list[Track], model: ScaleModel, recon=None) -> dict[str, Measurement]:
    frame_index = {ref: n for n, ref in enumerate(recon.frame_refs)} if recon is not None else {}
    out: dict[str, Measurement] = {}
    for t in tracks:
        if t.fragment:
            continue
        m = Measurement(t.id)
        dets = [d for d in t.best(3) if not d.attrs.get("cut_off")] or t.best(1)
        # 3D: global k, leave-one-out
        k, n = model.k(exclude_track=t.id)
        heights, thicks = [], []
        if k is not None:
            for d in dets:
                if d.ref in frame_index:
                    L, T = spine_px(d)
                    u_l = _units(recon, frame_index[d.ref], d, L)
                    u_t = _units(recon, frame_index[d.ref], d, T)
                    if u_l and u_t:
                        heights.append(k * u_l)
                        thicks.append(k * u_t)
        if heights:
            m.height_cm, m.thickness_cm = round(float(np.median(heights)), 1), round(float(np.median(thicks)), 1)
            m.method, m.n_anchors = "reconstruction depth x global anchor scale (leave-one-out)", n
        elif t.ref_box is not None:
            s, n = model.seg(t.segment, exclude_track=t.id)
            method = "shelf-plane pixels x segment anchor scale (leave-one-out)"
            if s is None:
                s, n, method = _format_prior(tracks, t.segment), 0, "standard-format prior (median upright book = 21 cm)"
            if s is not None:
                bw, bh = t.ref_box[2] - t.ref_box[0], t.ref_box[3] - t.ref_box[1]
                L, T = (bw, bh) if t.orientation == "flat" else (bh, bw)
                # apply the best detection's edge refinement to the reference-frame thickness
                ratios = [d.attrs["refined_thickness_px"] / (d.h if t.orientation == "flat" else d.w)
                          for d in dets if d.attrs.get("refined_thickness_px")]
                if ratios:
                    T *= float(np.median(ratios))
                m.height_cm, m.thickness_cm = round(L * s, 1), round(T * s, 1)
                m.method, m.n_anchors = method, n
        if t.orientation == "flat" and m.thickness_cm is not None:
            # A flat book in a stack is seen at an angle: its box includes the top cover, so the box
            # height is not the spine thickness (testing gave 10-30 cm). Leave it unmeasured.
            m.thickness_cm = None
            m.method += "; flat book: thickness not measured (box includes the cover seen at an angle)"
        m.confidence = _confidence(m)
        out[t.id] = m
    return out


def refine_tracks(sweep_dir: Path, tracks: list[Track], per_track: int = 3) -> int:
    """Edge-refines the thickness of each track's best detections in place. Returns how many changed."""
    by_frame: dict[str, list[Detection]] = {}
    for t in tracks:
        if t.fragment:
            continue
        for d in t.best(per_track):
            by_frame.setdefault(d.ref, []).append(d)
    n = 0
    for ref, dets in by_frame.items():
        gray = cv2.imread(str(sweep_dir / ref), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for d in dets:
            r = refine_thickness(gray, d)
            if r is not None:
                d.attrs["refined_thickness_px"] = r
                n += 1
    return n


def _format_prior(tracks: list[Track], segment: int) -> float | None:
    hs = [t.ref_box[3] - t.ref_box[1] for t in tracks if t.segment == segment and t.ref_box and not t.fragment and t.orientation != "flat"]
    return FORMAT_PRIOR_CM / float(np.median(hs)) if hs else None


def _confidence(m: Measurement) -> float:
    if m.height_cm is None:
        return 0.0
    if m.method.startswith("standard-format"):
        return 0.3
    base = 0.85 if m.method.startswith("reconstruction") else 0.7
    return round(min(0.95, base * (0.6 + 0.4 * min(m.n_anchors, 5) / 5)), 2)
