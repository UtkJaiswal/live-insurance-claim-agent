"""Stage 3: read each book's spine from full-resolution crops.

The model transcribes printed text only. Every title/author it returns is checked in code against
its own transcription; a field that does not appear in the printed text is dropped, so a title cannot
be completed from memory. Books shelved backwards (page edges out) are not sent at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from app.pipeline.detect import Detection
from app.pipeline.frames import jpeg_bytes, read_image
from app.pipeline.gemini import Vision, image_part
from app.pipeline.track import Track

CROPS_PER_CALL = 8
CROP_MAX_SIDE = 1600
PAD = 0.06

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "crop": {"type": "integer"},
            "printed_text": {"type": "string", "description": "every word printed on the spine, exactly as printed, groups separated by ' | '"},
            "title": {"type": "string"},
            "author": {"type": "string"},
            "publisher": {"type": "string"},
            "volume": {"type": "string", "description": "volume number if printed (e.g. 'I', '2'), else empty"},
            "legibility": {"type": "string", "enum": ["clear", "partial", "none"]},
            "looks_antiquarian": {"type": "boolean", "description": "leather or old cloth binding, gilt tooling, raised bands, or clear signs of great age"},
        },
        "required": ["crop", "printed_text", "title", "author", "publisher", "volume", "legibility", "looks_antiquarian"],
    },
}

PROMPT = """\
Each image is a crop of one book spine (or cover) from a photo of a bookshelf. Text may run top to \
bottom or bottom to top. For each crop:
- printed_text: transcribe every word you can actually read, exactly as printed, in reading order.
- title, author, publisher, volume: copy them out of printed_text. Leave a field empty if it is not \
printed or you cannot read it. Do not complete cut-off or blurred words. Do not use what you know \
about books to fill anything in: if the spine shows only a surname, give only the surname.
- legibility: clear (title fully readable), partial (some words readable), none.
Ignore text from neighbouring books at the crop edges. Return one entry per crop, using its number.
"""

LEGIBILITY_RANK = {"clear": 2, "partial": 1, "none": 0}


@dataclass
class SpineRead:
    track_id: str
    det_id: str
    frame_ref: str
    printed_text: str = ""
    title: str = ""
    author: str = ""
    publisher: str = ""
    volume: str = ""
    legibility: str = "none"
    dropped: str = ""  # fields removed because they were not in printed_text
    rank: int = 0  # 0 = the track's best detection (not cut off, largest)
    looks_antiquarian: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def crop_box(img: np.ndarray, box: tuple[float, float, float, float], pad: float = PAD) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad + 6, (y1 - y0) * pad + 6
    return img[max(0, int(y0 - py)) : min(h, int(y1 + py)), max(0, int(x0 - px)) : min(w, int(x1 + px))]


def _supported(field_value: str, printed: str) -> bool:
    """True if every word of a field occurs (fuzzily) in the printed text."""
    if not field_value:
        return True
    words = [w for w in field_value.lower().replace("|", " ").split() if len(w) > 1]
    hay = printed.lower()
    return all(w in hay or fuzz.partial_ratio(w, hay) >= 90 for w in words)


def verify(read: SpineRead) -> SpineRead:
    dropped = []
    for name in ("title", "author", "publisher", "volume"):
        if not _supported(getattr(read, name), read.printed_text):
            dropped.append(f"{name}={getattr(read, name)!r}")
            setattr(read, name, "")
    read.dropped = "; ".join(dropped)
    if not read.printed_text.strip():
        read.legibility = "none"
    return read


async def read_spines(vision: Vision, sweep_dir: Path, tracks: list[Track], per_track: int = 2) -> dict[str, list[SpineRead]]:
    jobs: list[tuple[Track, Detection]] = []
    for t in tracks:
        if t.fragment or not t.spine_facing:
            continue
        jobs.extend((t, d) for d in t.best(per_track) if d.attrs.get("spine_facing", True))
    rank = {d.id: n for t in tracks for n, d in enumerate(t.best(per_track))}

    images: dict[str, np.ndarray] = {}

    def crop(d: Detection) -> bytes:
        if d.ref not in images:
            images[d.ref] = read_image(sweep_dir / d.ref)
        return jpeg_bytes(crop_box(images[d.ref], d.box), CROP_MAX_SIDE, 92)

    batches = [jobs[k : k + CROPS_PER_CALL] for k in range(0, len(jobs), CROPS_PER_CALL)]

    async def run(batch: list[tuple[Track, Detection]]) -> list[SpineRead]:
        parts: list[Any] = [PROMPT]
        for n, (_, d) in enumerate(batch, 1):
            parts += [f"Crop {n}:", image_part(crop(d))]
        raw = await vision.json("read", parts, SCHEMA, thinking="low")
        by_n = {r.get("crop"): r for r in raw if isinstance(r, dict)}
        out = []
        for n, (t, d) in enumerate(batch, 1):
            r = by_n.get(n, {})
            out.append(verify(SpineRead(
                track_id=t.id, det_id=d.id, frame_ref=d.ref, rank=rank.get(d.id, 0),
                printed_text=r.get("printed_text", ""), title=r.get("title", ""), author=r.get("author", ""),
                publisher=r.get("publisher", ""), volume=r.get("volume", ""), legibility=r.get("legibility", "none"),
                looks_antiquarian=bool(r.get("looks_antiquarian", False)),
            )))
        return out

    results: dict[str, list[SpineRead]] = {}
    for reads in await asyncio.gather(*(run(b) for b in batches)):
        for r in reads:
            results.setdefault(r.track_id, []).append(r)
    return results


def combine(reads: list[SpineRead]) -> tuple[SpineRead | None, float]:
    """Best reading for a track and a reading confidence in [0, 1].

    Two independent crops that agree on the title give high confidence; a single clear read gives
    moderate confidence; partial reads stay low and normally end in the review queue.
    """
    usable = [r for r in reads if r.legibility != "none" and (r.title or r.printed_text)]
    if not usable:
        return None, 0.0
    # Clearer read first; between equally clear reads, the one from the better detection. A crop of a
    # book cut by the frame edge can show its neighbour instead, so length of title is not a signal.
    best = min(usable, key=lambda r: (-LEGIBILITY_RANK[r.legibility], r.rank))
    others = [r for r in usable if r is not best and r.title]
    if best.title and others and max(fuzz.token_sort_ratio(best.title.lower(), o.title.lower()) for o in others) >= 85:
        conf = 0.95
    elif best.legibility == "clear":
        conf = 0.8 if not others else 0.6  # disagreeing second read lowers confidence
    else:
        conf = 0.45
    return best, conf


def drop_neighbour_spill(reads: dict[str, list[SpineRead]], rows: dict[str, tuple[int, int, int]]) -> list[str]:
    """Removes reads that repeat an adjacent book's title: the crop caught the neighbour's spine.

    rows maps track id -> (segment, shelf row, position). A read is dropped only if the neighbour's
    own reads agree on that title and this track has another read that does not. Returns notes.
    """
    notes = []
    consensus = {}
    for tid, rs in reads.items():
        titles = [r.title.lower() for r in rs if r.title]
        if len(titles) >= 2 and fuzz.token_sort_ratio(titles[0], titles[1]) >= 85:
            consensus[tid] = titles[0]
    where = {v: k for k, v in rows.items()}
    for tid, rs in reads.items():
        if tid not in rows or len(rs) < 2:
            continue
        seg, row, pos = rows[tid]
        for npos in (pos - 1, pos + 1):
            nb = where.get((seg, row, npos))
            if nb is None or nb not in consensus:
                continue
            spill = [r for r in rs if r.title and fuzz.token_sort_ratio(r.title.lower(), consensus[nb]) >= 85]
            if spill and len(spill) < len(rs):
                for r in spill:
                    rs.remove(r)
                notes.append(f"{tid}: dropped read of {consensus[nb]!r} from {spill[0].det_id} (neighbour {nb})")
    return notes
