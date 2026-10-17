"""Stage 7: non-book items. Merge sightings across frames, measure, decide art handling.

Furniture is seen from different angles as the user walks, so the planar linking used for shelves
does not apply. One vision call looks at a thumbnail of every sighting and groups the ones that show
the same physical object; the code checks the grouping (every sighting used exactly once, ids valid)
and falls back to one item per sighting cluster in time if the answer is malformed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from app.pipeline.detect import Detection
from app.pipeline.frames import jpeg_bytes, read_image
from app.pipeline.gemini import Vision, image_part
from app.pipeline.read import crop_box

MAX_SIGHTINGS = 120
THUMB = 384

GROUP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"ids": {"type": "array", "items": {"type": "integer"}}, "label": {"type": "string"}},
        "required": ["ids", "label"],
    },
}

GROUP_PROMPT = """\
Below are numbered crops of objects detected in frames of one continuous video walk through a single \
room, in time order, each with its detector label. Group the crops that show the SAME physical object \
(seen again from another angle or distance). Different objects of the same kind (two identical chairs) \
are separate groups. Every crop number must appear in exactly one group. Give each group a short plain \
label.
"""


@dataclass
class ItemRecord:
    id: str
    category: str
    label: str
    material: str
    brand_model: str | None
    is_artwork: bool
    frame_ref: str
    sightings: list[str]
    dims_cm: dict[str, float | None] = field(default_factory=lambda: {"w": None, "h": None, "d": None})
    dims_method: str = ""
    confidence: float = 0.0
    print_confirmed: bool = False
    statement: str | None = None
    product_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_groups(raw: Any, n: int) -> list[list[int]] | None:
    if not isinstance(raw, list):
        return None
    seen: set[int] = set()
    groups = []
    for g in raw:
        ids = [i for i in (g.get("ids") or []) if isinstance(i, int) and 1 <= i <= n]
        if not ids or seen & set(ids):
            return None
        seen |= set(ids)
        groups.append(ids)
    missing = set(range(1, n + 1)) - seen
    groups += [[i] for i in sorted(missing)]  # anything left out stays its own item
    return groups


def _time_fallback(dets: list[Detection], frame_t: dict[int, float]) -> list[list[int]]:
    """Same category and similar label within 8 s -> same object. Used only if grouping fails."""
    groups: list[list[int]] = []
    for n, d in enumerate(dets, 1):
        for g in groups:
            last = dets[g[-1] - 1]
            if (last.attrs.get("category") == d.attrs.get("category")
                    and fuzz.token_set_ratio(last.attrs.get("label", ""), d.attrs.get("label", "")) >= 70
                    and abs(frame_t.get(d.frame_i, 0) - frame_t.get(last.frame_i, 0)) <= 8):
                g.append(n)
                break
        else:
            groups.append([n])
    return groups


async def group_items(vision: Vision, sweep_dir: Path, dets: list[Detection], frame_t: dict[int, float]) -> list[list[Detection]]:
    dets = sorted(dets, key=lambda d: (frame_t.get(d.frame_i, 0), d.id))[:MAX_SIGHTINGS]
    if not dets:
        return []
    if len(dets) == 1:
        return [dets]
    images: dict[str, np.ndarray] = {}
    parts: list[Any] = [GROUP_PROMPT]
    for n, d in enumerate(dets, 1):
        if d.ref not in images:
            images[d.ref] = read_image(sweep_dir / d.ref)
        parts += [f"Crop {n} (t={frame_t.get(d.frame_i, 0):.0f}s, detector: {d.attrs.get('category')} / {d.attrs.get('label')}):",
                  image_part(jpeg_bytes(crop_box(images[d.ref], d.box, 0.02), THUMB, 85))]
    groups = None
    try:
        groups = _validate_groups(await vision.json("items", parts, GROUP_SCHEMA, thinking="low", media_resolution="low"), len(dets))
    except Exception:
        groups = None
    if groups is None:
        groups = _time_fallback(dets, frame_t)
    return [[dets[i - 1] for i in g] for g in groups]


def build_items(groups: list[list[Detection]], recon=None, k_cm_per_unit: float | None = None) -> list[ItemRecord]:
    out = []
    for n, g in enumerate(groups, 1):
        rep = max(g, key=lambda d: (not d.attrs.get("cut_off"), d.w * d.h))
        cat = Counter(d.attrs.get("category", "other") for d in g).most_common(1)[0][0]
        brand = next((d.attrs["brand_model"] for d in g if d.attrs.get("brand_model")), None)
        rec = ItemRecord(
            id=f"item_{n:03d}", category=cat, label=rep.attrs.get("label", cat), material=rep.attrs.get("material", ""),
            brand_model=brand, is_artwork=any(d.attrs.get("is_artwork") for d in g) or cat == "art",
            frame_ref=rep.ref, sightings=[d.id for d in g],
        )
        types = Counter(d.attrs.get("product_type", "").strip().lower() for d in g if d.attrs.get("product_type"))
        rec.product_type = types.most_common(1)[0][0] if types else ""
        rec.confidence = round(min(0.95, 0.5 + 0.1 * len(g) + (0.1 if not rep.attrs.get("cut_off") else -0.1)), 2)
        out.append(rec)
    add_dimensions(out, groups, recon, k_cm_per_unit)
    return out


def add_dimensions(items: list[ItemRecord], groups: list[list[Detection]], recon=None, k_cm_per_unit: float | None = None) -> None:
    """Width and height from reconstruction depth and the anchor scale (run once both exist)."""
    if recon is None or not k_cm_per_unit:
        return
    frame_index = {ref: n for n, ref in enumerate(recon.frame_refs)}
    for rec, g in zip(items, groups):
        whs = []
        for d in g:
            if d.ref in frame_index and not d.attrs.get("cut_off"):
                i = frame_index[d.ref]
                z = _depth(recon, i, d)
                if z:
                    f = float(recon.intrinsics[i][1, 1]) * recon.orig_size[0] / recon.proc_size[0]
                    whs.append((d.w * z / f * k_cm_per_unit, d.h * z / f * k_cm_per_unit))
        if whs:
            rec.dims_cm = {"w": round(float(np.median([w for w, _ in whs])), 0), "h": round(float(np.median([h for _, h in whs])), 0), "d": None}
            rec.dims_method = f"reconstruction depth x anchor scale, {len(whs)} view(s); depth not measured"


def _depth(recon, i: int, d: Detection) -> float | None:
    h, w = recon.proc_size
    H, W = recon.orig_size
    x0, x1 = int((d.box[0] + 0.3 * d.w) * w / W), int((d.box[2] - 0.3 * d.w) * w / W)
    y0, y1 = int((d.box[1] + 0.3 * d.h) * h / H), int((d.box[3] - 0.3 * d.h) * h / H)
    patch = recon.depth[i, max(0, y0) : max(y0 + 1, min(h, y1)), max(0, x0) : max(x0 + 1, min(w, x1))]
    return float(np.median(patch)) if patch.size else None


PRINT_RE = re.compile(r"\b(print|poster|reproduction|copy)\b", re.I)
ORIGINAL_RE = re.compile(r"\b(original|hand[- ]?painted|oil on|signed)\b", re.I)


def apply_statements(items: list[ItemRecord], statements: list[dict[str, Any]]) -> None:
    """Art defaults to appraisal; a policyholder's 'it is a print' about that piece lifts it."""
    art = [it for it in items if it.is_artwork]
    for s in statements:
        text = f"{s.get('subject', '')} {s.get('statement', '')}"
        if not PRINT_RE.search(text) or ORIGINAL_RE.search(s.get("statement", "")):
            continue
        cands = sorted(art, key=lambda it: fuzz.token_set_ratio(s.get("subject", ""), it.label), reverse=True)
        target = None
        if len(art) == 1:
            target = art[0]
        elif cands and fuzz.token_set_ratio(s.get("subject", ""), cands[0].label) >= 55:
            target = cands[0]
        if target:
            target.print_confirmed = True
            target.statement = s.get("statement")
