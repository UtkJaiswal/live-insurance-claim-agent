"""Stage 2: link detections of the same book across frames, so each physical book is counted once.

A shelf face is close to a plane, so two overlapping frames of it are related by a homography. We
estimate it from SIFT matches (RANSAC), warp one frame's book boxes into the other, and link boxes
that overlap. Linked detections form a track; one track is one book. Frames connected by successful
homographies form a segment (one continuous stretch of shelving); within a segment, every book gets
coordinates in a reference frame, which gives shelf rows and left-to-right positions.

Known limit: if the camera crosses a gap with no shared texture (for example a bare wall between two
bookcases) the segment splits. That is intended. If it splits in the middle of one bookcase because
of blur, books in the overlap can be counted twice; the review queue flags tracks that sit at a
segment boundary.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.detect import Detection
from app.pipeline.frames import Frame

FEATURE_MAX_SIDE = 1280
MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.25
LINK_IOU = 0.3
FRAME_WINDOW = 4  # try linking each frame with the next N frames
FRAGMENT_HEIGHT_RATIO = 0.35


@dataclass
class FrameFeatures:
    scale: float  # feature image px per original px
    keypoints: np.ndarray  # [K, 2] in feature-image px
    descriptors: np.ndarray | None


@dataclass
class Track:
    id: str
    dets: list[Detection]
    segment: int = -1
    ref_box: tuple[float, float, float, float] | None = None  # box in the segment reference frame
    shelf_row: int | None = None
    position: int | None = None
    fragment: bool = False  # only ever seen as a small cut-off piece
    at_segment_edge: bool = False

    @property
    def orientation(self) -> str:
        votes = defaultdict(int)
        for d in self.dets:
            votes[d.attrs.get("orientation", "upright")] += 1
        return max(votes, key=votes.get)

    @property
    def spine_facing(self) -> bool:
        return sum(1 for d in self.dets if d.attrs.get("spine_facing", True)) * 2 >= len(self.dets)

    def best(self, k: int = 2) -> list[Detection]:
        """The k detections most useful for reading: not cut off, largest box first."""
        return sorted(self.dets, key=lambda d: (bool(d.attrs.get("cut_off")), -(d.w * d.h)))[:k]


def frame_features(path: Path) -> FrameFeatures:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None:
        raise ValueError(f"cannot read {path}")
    full_w = img.shape[1] * 2
    s = min(1.0, FEATURE_MAX_SIDE / max(img.shape))
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    kps, desc = cv2.SIFT_create(nfeatures=4000).detectAndCompute(img, None)
    return FrameFeatures(img.shape[1] / full_w, np.array([k.pt for k in kps], np.float32).reshape(-1, 2), desc)


def homography(a: FrameFeatures, b: FrameFeatures) -> np.ndarray | None:
    """H mapping original-pixel coordinates of frame a to frame b, or None if they do not overlap."""
    if a.descriptors is None or b.descriptors is None or len(a.keypoints) < 8 or len(b.keypoints) < 8:
        return None
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    pairs = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = a.keypoints[[m.queryIdx for m in good]]
    dst = b.keypoints[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None
    inliers = int(mask.sum())
    if inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
        return None
    S_a = np.diag([a.scale, a.scale, 1.0])
    S_b_inv = np.diag([1 / b.scale, 1 / b.scale, 1.0])
    return S_b_inv @ H @ S_a


def warp_box(H: np.ndarray, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    p = np.c_[pts, np.ones(4)] @ H.T
    p = p[:, :2] / p[:, 2:3]
    return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def containment(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


class _Groups:
    """Union-find that refuses to merge two groups containing detections from the same frame."""

    def __init__(self, dets: list[Detection]):
        self.parent = list(range(len(dets)))
        self.frames = [{d.frame_i} for d in dets]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> bool:
        a, b = self.find(i), self.find(j)
        if a == b:
            return True
        if self.frames[a] & self.frames[b]:
            return False
        self.parent[b] = a
        self.frames[a] |= self.frames[b]
        return True


@dataclass
class TrackResult:
    tracks: list[Track]
    segments: list[list[int]]  # frame indices per segment
    links: list[tuple[int, int]] = field(default_factory=list)  # frame pairs with a homography


def build_tracks(
    sweep_dir: Path,
    frames: list[Frame],
    detections: dict[int, list[Detection]],
    kind: str = "book",
    features: dict[int, FrameFeatures] | None = None,
) -> TrackResult:
    frames = sorted(frames, key=lambda f: f.t)
    feats = features if features is not None else {f.i: frame_features(f.path(sweep_dir)) for f in frames}
    dets = [d for f in frames for d in detections.get(f.i, []) if d.kind == kind]
    by_frame: dict[int, list[int]] = defaultdict(list)
    for n, d in enumerate(dets):
        by_frame[d.frame_i].append(n)

    groups = _Groups(dets)
    H_pairs: dict[tuple[int, int], np.ndarray] = {}
    for a_pos, fa in enumerate(frames):
        for fb in frames[a_pos + 1 : a_pos + 1 + FRAME_WINDOW]:
            H = homography(feats[fa.i], feats[fb.i])
            if H is None:
                continue
            H_pairs[(fa.i, fb.i)] = H
            cands = []
            for i in by_frame[fa.i]:
                wb = warp_box(H, dets[i].box)
                for j in by_frame[fb.i]:
                    if (dets[i].attrs.get("orientation") == "flat") != (dets[j].attrs.get("orientation") == "flat"):
                        continue
                    o = iou(wb, dets[j].box)
                    if dets[i].attrs.get("cut_off") or dets[j].attrs.get("cut_off"):
                        # A book cut by the frame edge is a sliver of its full box in the other frame:
                        # IoU is small, but the sliver lies almost entirely inside the full box.
                        o = max(o, containment(wb, dets[j].box) - 0.3)
                    if o >= LINK_IOU:
                        cands.append((o, i, j))
            used_i, used_j = set(), set()
            for _, i, j in sorted(cands, reverse=True):
                if i in used_i or j in used_j:
                    continue
                if groups.union(i, j):
                    used_i.add(i)
                    used_j.add(j)

    members: dict[int, list[Detection]] = defaultdict(list)
    for n, d in enumerate(dets):
        members[groups.find(n)].append(d)
    tracks = [Track(id="", dets=sorted(m, key=lambda d: d.frame_i)) for m in members.values()]

    segments = _segments(frames, H_pairs)
    _mark_fragments(tracks, dets)
    _place(tracks, frames, segments, H_pairs)
    # real books first, in shelf order; fragments get their own prefix
    ordered = sorted(tracks, key=lambda t: (t.fragment, t.segment, t.shelf_row or 0, t.position or 0))
    for n, tr in enumerate(t for t in ordered if not t.fragment):
        tr.id = f"{kind}_{n + 1:03d}"
    for n, tr in enumerate(t for t in ordered if t.fragment):
        tr.id = f"{kind}_fragment_{n + 1:03d}"
    tracks.sort(key=lambda t: t.id)
    return TrackResult(tracks, segments, sorted(H_pairs))


def _segments(frames: list[Frame], H_pairs: dict[tuple[int, int], np.ndarray]) -> list[list[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in H_pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen, out = set(), []
    for f in frames:
        if f.i in seen:
            continue
        comp, q = [], deque([f.i])
        seen.add(f.i)
        while q:
            x = q.popleft()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        out.append(sorted(comp))
    return out


def _to_ref(ref: int, comp: list[int], H_pairs: dict[tuple[int, int], np.ndarray]) -> dict[int, np.ndarray]:
    """Homography from each frame in the segment to the reference frame (BFS over links)."""
    to_ref = {ref: np.eye(3)}
    q = deque([ref])
    comp_set = set(comp)
    while q:
        x = q.popleft()
        for (a, b), H in H_pairs.items():
            # H maps a -> b. to_ref[x] maps x -> ref.
            if a == x and b in comp_set and b not in to_ref:
                to_ref[b] = to_ref[x] @ np.linalg.inv(H)  # b -> a -> ref
                q.append(b)
            elif b == x and a in comp_set and a not in to_ref:
                to_ref[a] = to_ref[x] @ H  # a -> b -> ref
                q.append(a)
    return to_ref


def _place(tracks: list[Track], frames: list[Frame], segments: list[list[int]], H_pairs) -> None:
    seg_of = {i: s for s, comp in enumerate(segments) for i in comp}
    for s, comp in enumerate(segments):
        seg_tracks = [t for t in tracks if seg_of.get(t.dets[0].frame_i) == s]
        if not seg_tracks:
            continue
        counts = defaultdict(int)
        for t in seg_tracks:
            for d in t.dets:
                counts[d.frame_i] += 1
        ref = max(comp, key=lambda i: counts.get(i, 0))
        to_ref = _to_ref(ref, comp, H_pairs)
        edge_frames = {comp[0], comp[-1]} if len(segments) > 1 else set()
        for t in seg_tracks:
            t.segment = s
            boxes = [warp_box(to_ref[d.frame_i], d.box) for d in t.dets if d.frame_i in to_ref]
            if boxes:
                t.ref_box = tuple(float(np.median([b[k] for b in boxes])) for k in range(4))  # type: ignore[assignment]
            t.at_segment_edge = bool(edge_frames & {d.frame_i for d in t.dets}) and len(t.dets) == 1
        _rows(seg_tracks)


def _rows(tracks: list[Track]) -> None:
    """Groups books into shelf rows by the height of their bottom edge, then orders left to right."""
    placed = [t for t in tracks if t.ref_box is not None and not t.fragment]
    if not placed:
        return
    heights = [t.ref_box[3] - t.ref_box[1] for t in placed if t.orientation != "flat"] or [
        t.ref_box[3] - t.ref_box[1] for t in placed
    ]
    med = float(np.median(heights))
    gap = 0.4 * med
    # Rows come from full-height books standing on a shelf board. A short box (a book mostly hidden
    # behind a neighbour, seen only by its top) does not end at the board, so it joins whichever row's
    # vertical span contains its centre instead of starting a row of its own.
    full = sorted((t for t in placed if t.ref_box[3] - t.ref_box[1] >= 0.5 * med), key=lambda t: t.ref_box[3])
    short = [t for t in placed if t.ref_box[3] - t.ref_box[1] < 0.5 * med]
    row, last = 0, None
    for t in full:
        if last is not None and t.ref_box[3] - last > gap:
            row += 1
        t.shelf_row = row
        last = t.ref_box[3]
    spans = {}
    for t in full:
        top, bottom = spans.get(t.shelf_row, (t.ref_box[1], t.ref_box[3]))
        spans[t.shelf_row] = (min(top, t.ref_box[1]), max(bottom, t.ref_box[3]))
    for t in short:
        cy = (t.ref_box[1] + t.ref_box[3]) / 2
        inside = [r for r, (top, bottom) in spans.items() if top <= cy <= bottom]
        if inside:
            t.shelf_row = inside[0]
        elif spans:  # nearest row by bottom edge
            t.shelf_row = min(spans, key=lambda r: abs(spans[r][1] - t.ref_box[3]))
        else:
            t.shelf_row = 0
    for r in {t.shelf_row for t in placed}:
        in_row = sorted((t for t in placed if t.shelf_row == r), key=lambda t: t.ref_box[0])
        for p, t in enumerate(in_row, 1):
            t.position = p


def _mark_fragments(tracks: list[Track], dets: list[Detection]) -> None:
    by_frame: dict[int, list[float]] = defaultdict(list)
    for d in dets:
        if not d.attrs.get("cut_off") and d.attrs.get("orientation") != "flat":
            by_frame[d.frame_i].append(d.h)
    for t in tracks:
        if not all(d.attrs.get("cut_off") for d in t.dets):
            continue
        ratios = [d.h / np.median(by_frame[d.frame_i]) for d in t.dets if by_frame.get(d.frame_i)]
        if ratios and max(ratios) < FRAGMENT_HEIGHT_RATIO:
            t.fragment = True


MERGE_WINDOW_S = 10.0
MERGE_TEXT_SCORE = 88


def merge_across_segments(tracks: list[Track], texts: dict[str, str], frame_t: dict[int, float]) -> list[str]:
    """Merges tracks that are the same book seen in two capture segments that failed to link.

    When the camera moves fast or the view changes sharply, consecutive frames do not get a
    homography and the same stack appears in two segments (seen in a real sweep: a blurred oblique
    frame of a 7-book stack added 6 duplicates). Within MERGE_WINDOW_S seconds, two tracks from
    different segments whose spine text agrees, and which never appear in the same frame, are one
    book. Pairs are matched one-to-one, best text agreement first. Unreadable spines cannot be
    matched this way and are left alone. Returns notes for the metrics.
    """
    from rapidfuzz import fuzz

    from app.pipeline.identify import norm

    live = [t for t in tracks if not t.fragment and texts.get(t.id)]
    span = {t.id: (min(frame_t.get(d.frame_i, 0.0) for d in t.dets), max(frame_t.get(d.frame_i, 0.0) for d in t.dets)) for t in live}
    frames_of = {t.id: {d.frame_i for d in t.dets} for t in live}
    pairs = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a.segment == b.segment or frames_of[a.id] & frames_of[b.id]:
                continue
            gap = max(span[a.id][0], span[b.id][0]) - min(span[a.id][1], span[b.id][1])
            if gap > MERGE_WINDOW_S:
                continue
            ta, tb = norm(texts[a.id]), norm(texts[b.id])
            score = fuzz.token_sort_ratio(ta, tb)
            # One read often has an extra word or is cut short ("Sharma | 5th EDITION | How to Prepare
            # for DAT" vs "5TH EDITION | How to Prepare for DATA INTERPRETATION"): allow a subset match,
            # but only for reads of 3+ words so "ROBIN SHARMA" cannot match every Sharma book.
            if min(len(ta.split()), len(tb.split())) >= 3:
                score = max(score, fuzz.token_set_ratio(ta, tb) - 4)
            # A crop can also catch a neighbour's text. One distinctive segment of 4+ words that matches
            # ("How to Prepare for DATA INTERPRETATION for CAT") is the same spine.
            segs_a = [norm(x) for x in texts[a.id].split("|") if len(norm(x).split()) >= 4]
            segs_b = [norm(x) for x in texts[b.id].split("|") if len(norm(x).split()) >= 4]
            if segs_a and segs_b:
                score = max(score, max(fuzz.token_sort_ratio(x, y) for x in segs_a for y in segs_b))
            if score >= MERGE_TEXT_SCORE:
                pairs.append((score, a, b))
    notes, gone = [], set()
    for score, a, b in sorted(pairs, key=lambda p: -p[0]):
        if a.id in gone or b.id in gone:
            continue
        keep, drop = (a, b) if len(a.dets) >= len(b.dets) else (b, a)
        if frames_of[keep.id] & frames_of[drop.id]:
            continue
        keep.dets.extend(drop.dets)
        frames_of[keep.id] |= frames_of[drop.id]
        drop.fragment = True  # removed from the count; its detections now belong to `keep`
        gone.add(drop.id)
        notes.append(f"{drop.id} merged into {keep.id}: same spine text {texts[drop.id][:40]!r} in segments {drop.segment} and {keep.segment}")
    return notes


DUPLICATE_BOX_IOU = 0.55
MIN_BOOK_HEIGHT_FRAC = 0.10


def clean_book_detections(detections: dict, frame_h: dict[int, float] | None = None) -> tuple[dict, list[str]]:
    """Per-frame clean-up of book boxes before tracking.

    - Two boxes on one spine (IoU >= 0.55): the looser (larger) one is dropped. In a real sweep the
      detector drew 2-3 overlapping boxes on thin spines, and each crop read the same title.
    - Books under 10% of the frame height are dropped: seen from across the room they cannot be read
      or measured, and in testing a distant view of a shelf added a duplicate copy of every book on it.
      The live scanner asks the user to step closer instead.
    Returns the cleaned detections and notes.
    """
    out, notes = {}, []
    for fi, dets in detections.items():
        books = [d for d in dets if d.kind == "book"]
        H = (frame_h or {}).get(fi) or max((d.attrs.get("_frame_h") or 0) for d in dets) if dets else 0
        keep = []
        for d in sorted(books, key=lambda d: d.w * d.h):
            if H and d.h < MIN_BOOK_HEIGHT_FRAC * H and d.attrs.get("orientation") != "flat":
                notes.append(f"{d.id}: dropped, too small to read ({d.h / H:.0%} of frame height)")
                continue
            if any(iou(d.box, k.box) >= DUPLICATE_BOX_IOU for k in keep):
                notes.append(f"{d.id}: dropped, second box on the same spine")
                continue
            keep.append(d)
        out[fi] = keep + [d for d in dets if d.kind != "book"]
    return out, notes


WEAK_DETS = 2
STRONG_DETS = 4


def absorb_weak_tracks(tracks: list[Track]) -> list[str]:
    """A track seen in 1-2 frames whose shelf position lies mostly inside a well-seen track's (4+ frames)
    is the same book, missed by box linking after the camera moved. Seen in a real sweep: frames late in
    a shelf pass added 5 single-sighting duplicates of books already tracked across 12 frames."""
    notes = []
    strong = [t for t in tracks if not t.fragment and len(t.dets) >= STRONG_DETS and t.ref_box]
    for w in [t for t in tracks if not t.fragment and len(t.dets) <= WEAK_DETS and t.ref_box]:
        frames_w = {d.frame_i for d in w.dets}
        cands = [(containment(w.ref_box, s.ref_box), s) for s in strong
                 if s.segment == w.segment and not frames_w & {d.frame_i for d in s.dets}]
        cands = [(c, s) for c, s in cands if c >= 0.5]
        if not cands:
            continue
        c, s = max(cands, key=lambda x: x[0])
        s.dets.extend(w.dets)
        w.fragment = True
        notes.append(f"{w.id} absorbed into {s.id}: seen in {len(frames_w)} frame(s), {c:.0%} inside its shelf position")
    return notes


def drop_lone_reviews(tracks: list[Track], segments: list[list[int]], frame_t: dict[int, float], texts: dict[str, str]) -> list[str]:
    """A segment of one frame whose books are mostly unreadable (a word or two at most), within 10 s of a multi-frame segment with
    at least as many books, is that shelf seen again from further away (a real sweep: a distant view
    through a wardrobe door added 5 duplicates). Its tracks are removed from the count."""
    notes = []
    live = [t for t in tracks if not t.fragment]
    by_seg: dict[int, list[Track]] = defaultdict(list)
    for t in live:
        by_seg[t.segment].append(t)
    for s_idx, comp in enumerate(segments):
        if len(comp) != 1 or s_idx not in by_seg:
            continue
        lone = by_seg[s_idx]
        if sum(1 for t in lone if len((texts.get(t.id) or "").split()) >= 3) > 0.4 * len(lone):
            continue  # mostly readable: merge_across_segments handles those by text
        t0 = frame_t.get(comp[0], 0.0)
        for o_idx, other in enumerate(segments):
            if o_idx == s_idx or len(other) < 3 or len(by_seg.get(o_idx, [])) < len(lone):
                continue
            if min(abs(frame_t.get(f, 0.0) - t0) for f in other) <= MERGE_WINDOW_S:
                for t in lone:
                    t.fragment = True
                notes.append(f"segment {s_idx} (1 frame, {len(lone)} unreadable books) dropped as a re-view of segment {o_idx}")
                break
    return notes
