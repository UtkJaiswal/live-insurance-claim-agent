"""Background detection while the user is still sweeping.

Every ~2 s of usable footage, one keyframe goes through the same detector the post-sweep pipeline
uses. Results fill the on-screen inventory live, prompt the agent to ask about art, and are handed to
the pipeline afterwards so those frames are not detected twice.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

from app.capture.quality import Thresholds
from app.pipeline.detect import Detection, detect_frame
from app.pipeline.frames import Frame, usable
from app.pipeline.gemini import Vision
from app.pipeline.ledger import Ledger

if TYPE_CHECKING:
    from app.live.bridge import LiveBridge

log = logging.getLogger("scanner")

MIN_GAP_S = 2.0  # consecutive scanned frames must overlap enough for tracking to link them
MAX_IN_FLIGHT = 3


class LiveScanner:
    def __init__(self, bridge: "LiveBridge"):
        self.bridge = bridge
        self.ledger = Ledger()
        self.vision = Vision(bridge.settings.gemini_api_key, bridge.settings.vision_model, self.ledger, max_concurrency=MAX_IN_FLIGHT)
        self.detections: dict[int, list[Detection]] = {}
        self.items_seen: list[tuple[str, str]] = []  # (category, label)
        self.art_asked: list[str] = []
        self._last_t = -1e9
        self._in_flight = 0
        self.max_books_in_view = 0
        self.closer_asked = False

    async def on_keyframe(self, ref: str, header: dict[str, Any]) -> None:
        b = self.bridge
        if not b.state.sweeping:
            return
        t = b.rec.t()
        m = re.search(r"f_(\d+)\.jpg$", ref)
        if not m or t - self._last_t < MIN_GAP_S or self._in_flight >= MAX_IN_FLIGHT:
            return
        if len(self.detections) + self._in_flight >= b.settings.max_analysed_frames:
            return  # frame budget for the sweep is used up
        quality = header.get("_quality", {})
        frame = Frame(i=int(m.group(1)) - 1, ref=ref, t=t, quality=quality, client=header,
                      region=header.get("region"), sweeping=True)
        if quality and not usable(frame, Thresholds()):
            return
        self._last_t = t
        self._in_flight += 1
        try:
            dets = await detect_frame(self.vision, b.rec.dir, frame)
        except Exception as e:
            log.warning("live detect failed on %s: %s", ref, e)
            return
        finally:
            self._in_flight -= 1
        self.detections[frame.i] = dets
        await self._report(frame, dets)

    async def _report(self, frame: Frame, dets: list[Detection]) -> None:
        b = self.bridge
        books = [d for d in dets if d.kind == "book" and not d.attrs.get("cut_off")]
        self.max_books_in_view = max(self.max_books_in_view, len(books))
        H = max((d.attrs.get("_frame_h") or 0) for d in dets) if dets else 0
        upright = [d.h / H for d in books if H and d.attrs.get("orientation") != "flat"]
        if len(upright) >= 3 and sorted(upright)[len(upright) // 2] < 0.12 and not self.closer_asked:
            self.closer_asked = True  # once per sweep; the capture monitor handles blur separately
            await b.nudge("Books are in view but too far away to read their spines. Ask the user to step "
                          "closer so a shelf fills most of the screen, then pan slowly along it.", "too_far")
        await b._send_json({"type": "scan", "ref": frame.ref, "books_in_view": len(books),
                            "frames_scanned": len(self.detections)})
        for d in dets:
            if d.kind != "item":
                continue
            cat, label = d.attrs.get("category", "other"), d.attrs.get("label", "")
            if any(c == cat and fuzz.token_set_ratio(label, l) >= 80 for c, l in self.items_seen):
                continue
            self.items_seen.append((cat, label))
            b.rec.log("live_item", category=cat, label=label, frame_ref=frame.ref)
            await b._send_json({"type": "item", "category": cat, "description": label, "brand_model": d.attrs.get("brand_model") or None})
            if (d.attrs.get("is_artwork") or cat == "art") and len(self.art_asked) < 3:
                self.art_asked.append(label)
                await b.nudge(f"A piece of art is in view ({label}). Ask the user whether it is an original or a print, "
                              "and record the answer with record_user_statement.", "art_question")
