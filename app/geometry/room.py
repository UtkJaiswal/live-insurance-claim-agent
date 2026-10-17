"""Room dimensions from the sweep's reconstruction.

1. Up direction: the phone's accelerometer gives gravity in device axes for every frame; rotated by
   that frame's camera pose it gives "up" in the reconstruction. iOS and Android disagree on the sign
   of accelerationIncludingGravity, so the sign is chosen from the data: a phone held upright sees
   up along the image's -y axis. Without motion data, the mean camera up-axis is used.
2. Floor and ceiling: peaks in the histogram of point heights near the bottom and top.
3. Walls: wall points projected onto the floor plane form lines. Rooms are assumed to have right
   angles (Manhattan): the rotation that makes the x and z histograms sharpest aligns the walls with
   the axes, and the outermost strong peaks on each axis are the walls.
4. Metric scale: the reconstruction is unscaled; k (cm per unit) comes from the same anchors as the
   spine measurements (app/pipeline/scale.py).

A room that is not a rectangle is reported as its bounding rectangle with a shape note (and the
convex-hull floor area for comparison) and goes to the review queue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SQFT_PER_M2 = 10.7639


@dataclass
class RoomResult:
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    floor_area_m2: float | None = None
    wall_area_m2: float | None = None
    shelved_wall_area_m2: float | None = None
    floor_area_ft2: float | None = None
    wall_area_ft2: float | None = None
    scale_method: str = ""
    confidence: float = 0.0
    shape_notes: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def up_vector(recon, clients: dict[str, dict[str, Any]]) -> tuple[np.ndarray, str]:
    ups = []
    for i, ref in enumerate(recon.frame_refs):
        g = (clients.get(ref) or {}).get("gravity")
        if not g or len(g) != 3 or not np.isfinite(g).all() or np.linalg.norm(g) < 5:
            continue
        # device axes (x right, y toward top edge, z out of the screen) -> OpenCV camera axes of the
        # rear camera (x right, y down, z forward)
        cam = np.array([g[0], -g[1], -g[2]], float)
        ups.append((i, cam / np.linalg.norm(cam)))
    if len(ups) >= 3:
        sign = 1.0 if np.mean([u[1] for _, u in ups]) < 0 else -1.0  # up should point along -y (image up)
        world = [recon.cam_to_world(i)[0] @ (sign * u) for i, u in ups]
        v = np.mean(world, axis=0)
        return v / np.linalg.norm(v), f"accelerometer gravity ({len(ups)} frames)"
    v = np.mean([recon.cam_to_world(i)[0] @ np.array([0.0, -1.0, 0.0]) for i in range(len(recon.frame_refs))], axis=0)
    return v / np.linalg.norm(v), "mean camera up-axis (no motion data)"


def rotation_to_y(up: np.ndarray) -> np.ndarray:
    """Rotation R with R @ up = (0, 1, 0)."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    c = float(up @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def _peak(values: np.ndarray, lo: float, hi: float, bins: int) -> tuple[float | None, float]:
    """Location and strength (count / median count) of the strongest histogram bin within [lo, hi]."""
    sel = values[(values >= lo) & (values <= hi)]
    if len(sel) < 50:
        return None, 0.0
    counts, edges = np.histogram(sel, bins=bins)
    k = int(np.argmax(counts))
    strength = counts[k] / (np.median(counts[counts > 0]) + 1e-9)
    return float((edges[k] + edges[k + 1]) / 2), float(strength)


def manhattan_angle(xz: np.ndarray) -> float:
    best, best_score = 0.0, -1.0
    span = np.ptp(xz, axis=0).max() + 1e-9
    for deg in np.arange(0, 90, 1.0):
        a = np.deg2rad(deg)
        r = xz @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        score = 0.0
        for axis in (0, 1):
            c, _ = np.histogram(r[:, axis], bins=int(np.ptp(r[:, axis]) / span * 300) + 10)
            p = c / c.sum()
            score += float((p ** 2).sum())
        if score > best_score:
            best, best_score = deg, score
    return float(best)


HANDHELD_M = (0.9, 2.0)  # a phone held by a standing or walking person
ARM_M = 0.3  # a phone is held at least this far from any wall
WALL_BIN_M = 0.05
DOOR_TOP_CM = 205.0  # door leaf height (tape-measured 205 cm in testing; standard Indian leaves are 200-210)


def wall_beyond(v: np.ndarray, cam_edge: float, direction: int) -> tuple[float | None, float]:
    """The wall on one side: the strongest plane of points beyond where the phone ever was.

    The phone was inside the room, so no wall lies between camera positions; furniture faces can.
    Beyond the outermost camera position the strongest histogram bin is the wall. Tested rules on a
    real sweep against tape: "farthest dense plane" picked a wardrobe mirror's reflection 0.5 m behind
    the wall; "nearest" picks furniture; "strongest" found both walls within 4%. Returns
    (position, strength) or (None, 0) if that side has too few points (the wall was not captured).
    """
    side = v[v <= cam_edge + 0.15] if direction < 0 else v[v >= cam_edge - 0.15]
    if len(side) < 300:
        return None, 0.0
    edges = np.arange(side.min(), side.max() + WALL_BIN_M, WALL_BIN_M)
    if len(edges) < 3:
        return None, 0.0
    counts, edges = np.histogram(side, bins=edges)
    k = int(counts.argmax())
    strength = counts[k] / (np.median(counts[counts > 0]) + 1e-9)
    return float((edges[k] + edges[k + 1]) / 2), float(strength)


def floor_level(points: np.ndarray, up: np.ndarray) -> float | None:
    """Height of the floor plane along `up`, in reconstruction units."""
    h = (points @ rotation_to_y(up).T)[:, 1]
    h = h[np.isfinite(h)]
    lo, hi = np.percentile(h, [0.5, 99.5])
    floor, _ = _peak(h, lo, lo + 0.3 * (hi - lo), 80)
    return floor


def door_top_scales(recon, door_dets: list, up: np.ndarray, floor: float, door_cm: float = DOOR_TOP_CM) -> list[tuple[str, float]]:
    """cm-per-unit samples from doors: the top edge of a door leaf stands door_cm above the floor.

    Uses only the top edge, at the depth of the door's jambs, and the floor plane. The bottom of a door
    is often hidden by furniture, which made the earlier full-height method read the door as short and
    inflate everything by about 40% in a real sweep. Detections whose top edge touches the frame are
    skipped.
    """
    R = rotation_to_y(up)
    idx = {r: i for i, r in enumerate(recon.frame_refs)}
    hh, ww = recon.proc_size
    H, W = recon.orig_size
    out = []
    for d in door_dets:
        if d.ref not in idx or d.box[1] <= 0.006 * H:
            continue
        i = idx[d.ref]
        x0, y0, x1, y1 = d.box
        zs = []
        for c in (int(x0 * ww / W), int(x1 * ww / W) - 1):
            patch = recon.depth[i, int(y0 * hh / H) : max(int(y0 * hh / H) + 1, int(y1 * hh / H)), max(0, c - 2) : c + 3]
            if patch.size:
                zs.append(float(np.median(patch)))
        if not zs:
            continue
        z = float(np.median(zs))
        K = recon.intrinsics[i]
        u, v = (x0 + x1) / 2 * ww / W, y0 * hh / H
        cam = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        Rwc, c = recon.cam_to_world(i)
        top_h = float(((Rwc @ cam + c) @ R.T)[1] - floor)
        if top_h > 0:
            out.append((d.id, door_cm / top_h))
    return out


def fit_room(points: np.ndarray, up: np.ndarray, k_cm_per_unit: float | None, scale_method: str,
             cameras: np.ndarray | None = None) -> RoomResult:
    res = RoomResult(scale_method=scale_method)
    if k_cm_per_unit is None:
        res.notes.append("no metric scale anchor in the sweep: room size cannot be stated")
        return res
    m = k_cm_per_unit / 100.0  # metres per unit
    Q = points @ rotation_to_y(up).T * m
    Q = Q[np.isfinite(Q).all(axis=1)]
    h = Q[:, 1]
    lo, hi = np.percentile(h, [0.5, 99.5])
    rng = hi - lo
    if rng <= 0.5:
        res.notes.append("reconstruction too flat to find floor and ceiling (was the whole room swept?)")
        return res
    floor, fs = _peak(h, lo, lo + 0.3 * rng, 80)
    ceil, cs = _peak(h, hi - 0.25 * rng, hi, 80)
    if floor is None:
        floor, fs = float(np.percentile(h, 2)), 0.0
        res.notes.append("floor plane not distinct; used the 2nd percentile of point heights")
    ceiling_seen = ceil is not None and cs >= 3.0 and ceil - floor > 2.0
    top = ceil if ceiling_seen else float(np.percentile(h, 99))
    if cameras is not None and len(cameras):
        # 0.9-2.0 m above the floor: walls, doors, wardrobes; not beds, tables or chairs, whose sides
        # would otherwise compete with the walls behind them.
        walls = Q[(h > floor + 0.9) & (h < floor + min(2.0, top - floor - 0.1))][:, [0, 2]]
    else:
        walls = Q[(h > floor + 0.15 * (top - floor)) & (h < top - 0.1 * (top - floor))][:, [0, 2]]
    if len(walls) < 500:
        res.notes.append("too few wall points to fit walls")
        return res
    ang = manhattan_angle(walls[np.random.default_rng(0).choice(len(walls), min(len(walls), 40000), replace=False)])
    a = np.deg2rad(ang)
    W = walls @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    C = None
    if cameras is not None and len(cameras):
        C = (cameras @ rotation_to_y(up).T * m)[:, [0, 2]] @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dims, strengths, missing = [], [], []
    for axis in (0, 1):
        v = W[:, axis]
        if C is not None:
            lo_w, s1 = wall_beyond(v, float(C[:, axis].min()), -1)
            hi_w, s2 = wall_beyond(v, float(C[:, axis].max()), +1)
            if lo_w is None:
                lo_w, s1 = float(C[:, axis].min()) - ARM_M, 0.0
                missing.append(axis)
            if hi_w is None:
                hi_w, s2 = float(C[:, axis].max()) + ARM_M, 0.0
                missing.append(axis)
            dims.append(hi_w - lo_w)
            strengths += [s1, s2]
            continue
        p1, p99 = np.percentile(v, [1, 99])
        span = p99 - p1
        bins = max(20, int(span / 0.03))  # ~3 cm bins
        left, ls = _peak(v, p1 - 0.05 * span, p1 + 0.35 * span, bins)
        right, rs = _peak(v, p99 - 0.35 * span, p99 + 0.05 * span, bins)
        if left is None or right is None:
            res.notes.append("could not find both walls on one axis")
            return res
        dims.append(right - left)
        strengths += [ls, rs]
    if missing:
        res.notes.append(f"{len(missing)} wall(s) not captured: that side is placed {ARM_M} m beyond the phone's furthest "
                         "position, so the size on that axis is a lower bound")
    length, width = max(dims), min(dims)
    height = top - floor
    implausible = False
    if cameras is not None and len(cameras):
        # Independent check on the scale: the phone was held by someone standing in the room.
        cam_h = float(np.median((cameras @ rotation_to_y(up).T * m)[:, 1] - floor))
        res.notes.append(f"check: with this scale the phone was held {cam_h:.2f} m above the floor")
        if not HANDHELD_M[0] <= cam_h <= HANDHELD_M[1]:
            implausible = True
            res.notes[-1] = (f"scale implausible: it puts the phone {cam_h:.2f} m above the floor (expected "
                             f"{HANDHELD_M[0]}-{HANDHELD_M[1]} m); room figures are unreliable")
    res.length_m, res.width_m, res.height_m = round(length, 2), round(width, 2), round(height, 2)
    res.floor_area_m2 = round(length * width, 2)
    res.wall_area_m2 = round(2 * (length + width) * height, 2)
    res.floor_area_ft2 = round(res.floor_area_m2 * SQFT_PER_M2, 1)
    res.wall_area_ft2 = round(res.wall_area_m2 * SQFT_PER_M2, 1)

    hull = _hull_area(W)
    rect = length * width
    shape = "rectangular (walls at right angles, fitted with sharpest-projection alignment)"
    if hull < 0.8 * rect:
        shape = (f"possibly not rectangular: wall points cover {hull:.1f} m2 of the {rect:.1f} m2 bounding rectangle; "
                 "the bounding rectangle is reported")
        res.notes.append("room shape may not be rectangular")
    res.shape_notes = shape + (". Wall area is gross (perimeter x height), openings not subtracted.")
    if not ceiling_seen:
        res.notes.append("ceiling not captured: height is the highest point seen (a lower bound)")
    weak = sum(1 for s in strengths if 0 < s < 3.0)
    if weak:
        res.notes.append(f"{weak} of 4 walls only weakly visible in the sweep")
    res.confidence = round(max(0.1, 0.85 - 0.12 * weak - 0.1 * len(missing) - (0.15 if not ceiling_seen else 0)
                               - (0.1 if hull < 0.8 * rect else 0)), 2)
    if implausible:
        # An empty field is correct when the system does not know: keep the rejected figures only in
        # the note, for audit, and report no room size.
        res.notes.append(f"rejected figures: {res.length_m} x {res.width_m} x {res.height_m} m")
        for f in ("length_m", "width_m", "height_m", "floor_area_m2", "wall_area_m2", "floor_area_ft2", "wall_area_ft2"):
            setattr(res, f, None)
        res.confidence = 0.0
    return res


def _hull_area(xy: np.ndarray) -> float:
    import cv2

    pts = xy[np.random.default_rng(1).choice(len(xy), min(len(xy), 20000), replace=False)].astype(np.float32)
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull))


def shelved_wall_area(items: list[Any]) -> float | None:
    """Front-face area of shelving units, m2. None if no shelving unit could be measured."""
    areas = [it.dims_cm["w"] * it.dims_cm["h"] / 1e4 for it in items
             if it.category == "shelving" and it.dims_cm.get("w") and it.dims_cm.get("h")]
    return round(sum(areas), 2) if areas else None


MAX_ENSEMBLE_SPREAD = 0.40  # (max - min) / median of floor area across reconstructions


def combine_estimates(estimates: list[RoomResult], scale_method: str) -> RoomResult:
    """Median room over several reconstructions of the same sweep, gated by their agreement.

    One reconstruction of a hand-held walking sweep is not stable: on a real bedroom, six
    reconstructions from staggered frame subsets gave floor areas from 7.0 to 19.6 m2 (tape: 9.4).
    Reporting one of them would be a number without evidence. The median is reported only when the
    floor areas agree within MAX_ENSEMBLE_SPREAD; otherwise the room is left empty with the range.
    """
    ok = [e for e in estimates if e.floor_area_m2]
    res = RoomResult(scale_method=scale_method)
    if not ok:
        res.notes = (estimates[0].notes if estimates else []) or ["room not measured"]
        return res
    if len(ok) < 3:
        best = max(ok, key=lambda e: e.confidence)
        best.notes = [*best.notes, f"room size from {len(ok)} reconstruction(s) only: not cross-checked"]
        best.confidence = round(min(best.confidence, 0.4), 2)
        best.scale_method = scale_method
        return best
    floors = np.array([e.floor_area_m2 for e in ok])
    spread = float(np.ptp(floors) / np.median(floors))
    pick = lambda f: float(np.median([getattr(e, f) for e in ok]))
    L, W, H = pick("length_m"), pick("width_m"), pick("height_m")
    res.notes = sorted({n for e in ok for n in e.notes if not n.startswith("check:")})
    res.notes.append(f"{len(ok)} reconstructions from staggered frame subsets: floor area {floors.min():.1f}-{floors.max():.1f} m2 "
                     f"(spread {spread:.0%})")
    if spread > MAX_ENSEMBLE_SPREAD:
        res.notes.append(f"room geometry unstable across reconstructions (spread {spread:.0%} > {MAX_ENSEMBLE_SPREAD:.0%}): "
                         f"room size not reported; median would have been {L:.2f} x {W:.2f} x {H:.2f} m")
        return res
    res.length_m, res.width_m, res.height_m = round(L, 2), round(W, 2), round(H, 2)
    res.floor_area_m2 = round(L * W, 2)
    res.wall_area_m2 = round(2 * (L + W) * H, 2)
    res.floor_area_ft2 = round(res.floor_area_m2 * SQFT_PER_M2, 1)
    res.wall_area_ft2 = round(res.wall_area_m2 * SQFT_PER_M2, 1)
    res.shape_notes = ok[0].shape_notes
    res.confidence = round(max(0.1, min(np.median([e.confidence for e in ok]), 1 - spread)), 2)
    return res
