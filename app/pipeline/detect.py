"""Stage 1: per-frame detection of books, non-book items and scale anchors.

One vision call per frame. Boxes come back as [ymin, xmin, ymax, xmax] on a 0-1000 grid and are
converted to pixel boxes (x0, y0, x1, y1) in the original frame. Nothing here reads text or names a
book: reading happens on full-resolution crops in read.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.pipeline.frames import Frame, jpeg_bytes, read_image
from app.pipeline.gemini import QuotaExhausted, Vision, image_part
from app.schema import ITEM_CATEGORIES

DETECT_MAX_SIDE = 2048

BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
SCHEMA = {
    "type": "object",
    "properties": {
        "books": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": BOX,
                    "orientation": {"type": "string", "enum": ["upright", "flat", "leaning"]},
                    "spine_facing": {"type": "boolean", "description": "true if the spine faces the camera; false if page edges or a cover face out"},
                    "cut_off": {"type": "boolean", "description": "true if the book is cut by the image edge or hidden behind something"},
                },
                "required": ["box_2d", "orientation", "spine_facing", "cut_off"],
            },
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": BOX,
                    "category": {"type": "string", "enum": list(ITEM_CATEGORIES)},
                    "label": {"type": "string", "description": "short noun phrase, e.g. 'wooden two-door wardrobe'"},
                    "product_type": {"type": "string", "description": "what a shop would call it, without colours or contents, e.g. 'double bed', 'wardrobe', 'study table', 'floor lamp'"},
                    "material": {"type": "string"},
                    "brand_model": {"type": "string", "description": "only if printed and legible; else empty"},
                    "is_artwork": {"type": "boolean", "description": "painting, portrait, framed picture or sculpture"},
                    "cut_off": {"type": "boolean"},
                },
                "required": ["box_2d", "category", "label", "product_type", "material", "brand_model", "is_artwork", "cut_off"],
            },
        },
        "anchors": {
            "type": "array",
            "description": "objects of standard size used for scale",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": BOX,
                    "kind": {"type": "string", "enum": ["door", "a4_sheet"]},
                    "fully_visible": {"type": "boolean", "description": "top and bottom edges both in frame"},
                },
                "required": ["box_2d", "kind", "fully_visible"],
            },
        },
    },
    "required": ["books", "items", "anchors"],
}

PROMPT = """\
This is one frame from a phone video of a room, taken for a home-contents insurance inventory.

books: every individual physical book you can see, including books lying flat in stacks (one entry \
per book in the stack), books shelved backwards (page edges facing out), books seen only by their \
cover, and books partly hidden behind or between other books (a strip of spine or a top edge showing \
above a neighbour counts). One tight box per book around the part that is visible. Do not merge \
neighbouring books. Skip anything you are not sure is a book: boxes, folders, magazines, notebooks and diaries, \
booklets, brochures and pamphlets are not books.

items: every movable object of value that is not a book: shelving units and bookcases, furniture, \
lamps and light fittings, framed art, portraits and pictures, rugs and mats, electronics, appliances, \
decor. Do not list walls, floors, ceilings, doors, windows, switches, sockets, curtains or rods. One \
entry per physical object; label it plainly. Fill brand_model only from text you can actually read.

anchors: doors (the door leaf in its frame) and A4 sheets of paper, if present.

Boxes are [ymin, xmin, ymax, xmax] on a 0-1000 grid. Return empty lists when nothing qualifies.
"""


@dataclass
class Detection:
    id: str
    frame_i: int
    ref: str
    kind: str  # book | item | anchor
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in original pixels
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_pixels(box_2d: list[int], width: int, height: int) -> tuple[float, float, float, float]:
    y0, x0, y1, x1 = (max(0, min(1000, v)) for v in box_2d)
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 < x0:
        x0, x1 = x1, x0
    return (x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height)


def parse_detections(raw: dict[str, Any], frame: Frame, width: int, height: int) -> list[Detection]:
    out: list[Detection] = []
    for kind, key in (("book", "books"), ("item", "items"), ("anchor", "anchors")):
        for n, d in enumerate(raw.get(key) or []):
            box = to_pixels(d.get("box_2d") or [0, 0, 0, 0], width, height)
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            attrs = {k: v for k, v in d.items() if k != "box_2d"}
            attrs["_frame_w"], attrs["_frame_h"] = width, height
            out.append(Detection(f"{frame.ref.rsplit('/', 1)[-1][:-4]}_{kind[0]}{n:03d}", frame.i, frame.ref, kind, box, attrs))
    return out


async def detect_frame(vision: Vision, sweep_dir: Path, frame: Frame) -> list[Detection]:
    img = read_image(frame.path(sweep_dir))
    h, w = img.shape[:2]
    raw = await vision.json("detect", [image_part(jpeg_bytes(img, DETECT_MAX_SIDE)), PROMPT], SCHEMA, thinking="low")
    return parse_detections(raw, frame, w, h)


MAX_FAILED_SHARE = 0.25


class DetectionFailed(RuntimeError):
    pass


async def detect_frames(vision: Vision, sweep_dir: Path, frames: list[Frame], done: dict[int, list[Detection]] | None = None,
                        failures: list[str] | None = None) -> dict[int, list[Detection]]:
    """Detects every frame not already in `done` (frames processed live during the sweep).

    A frame whose call fails is NOT recorded as "no books": it is left out of the results and named in
    `failures`, so the packet can say which frames went unanalysed. Too many failures abort the run.
    """
    results = dict(done or {})
    todo = [f for f in frames if f.i not in results]
    outs = await asyncio.gather(*(detect_frame(vision, sweep_dir, f) for f in todo), return_exceptions=True)
    failed = []
    for f, r in zip(todo, outs):
        if isinstance(r, QuotaExhausted):
            raise r
        if isinstance(r, BaseException):
            failed.append(f"{f.ref}: {type(r).__name__}: {str(r)[:120]}")
        else:
            results[f.i] = r
    if failures is not None:
        failures.extend(failed)
    if todo and len(failed) > MAX_FAILED_SHARE * len(frames):
        raise DetectionFailed(f"{len(failed)} of {len(todo)} frames failed detection; first: {failed[0]}")
    return results
