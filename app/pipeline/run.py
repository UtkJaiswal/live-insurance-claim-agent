"""Runs every post-sweep stage and writes the claim packet.

  detect -> track -> read -> (identify || reconstruct) -> scale/measure -> items -> price -> room
  -> assemble -> integrity check -> report

Each stage's output is saved under sweeps/<id>/stages/ so it can be inspected and tested on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
from PIL import Image

from app.config import ROOT, Settings
from app.geometry import da3
from app.geometry.room import DOOR_TOP_CM, RoomResult, combine_estimates, door_top_scales, fit_room, floor_level, shelved_wall_area, up_vector
from app.packet.integrity import check_packet
from app.packet.report import render
from app.pipeline.assemble import assemble, statements_for_tracks, summary_text
from app.pipeline.detect import Detection, detect_frames
from app.pipeline.frames import load_sweep, select_frames
from app.pipeline.gemini import Vision
from app.pipeline.http import CachedHTTP
from app.pipeline.identify import Catalogue, identify
from app.pipeline.items import add_dimensions, apply_statements, build_items, group_items
from app.pipeline.ledger import Ledger, live_session_cost
from app.pipeline.price import FX, CURRENCY_OF, BookPrice, ItemPrice, Shopping, appraisal_reason, price_book, price_item
from app.pipeline.read import SpineRead, combine, drop_neighbour_spill, read_spines
from app.pipeline.scale import Anchor, build_scale, measure, refine_tracks
from app.pipeline.tinyfish import TinyFishShopping
from app.pipeline.track import absorb_weak_tracks, build_tracks, clean_book_detections, drop_lone_reviews, merge_across_segments
from app.schema import ClaimPacket, ReviewEntry

log = logging.getLogger("pipeline")

MAX_RECON_FRAMES = 48
LOCALE_SAMPLE = 10
Progress = Callable[[str], Awaitable[None]] | None


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def _load_dets(path: Path) -> dict[int, list[Detection]]:
    raw = json.loads(path.read_text())
    dets = {int(k): [Detection(**{**d, "box": tuple(d["box"])}) for d in v] for k, v in raw.items()}
    sizes: dict[str, tuple[int, int]] = {}
    for ds in dets.values():  # detections saved before frame size was recorded
        for d in ds:
            if "_frame_h" not in d.attrs:
                if d.ref not in sizes:
                    with Image.open(path.parent.parent / d.ref) as im:
                        sizes[d.ref] = im.size
                d.attrs["_frame_w"], d.attrs["_frame_h"] = sizes[d.ref]
    return dets


def room_ensemble(sweep, doors, clients, n: int) -> list:
    """Room estimates from n reconstructions of staggered frame subsets (every n-th in-sweep frame from
    a different offset, plus door frames), each scaled by its own door-top samples."""
    from app.pipeline.frames import in_sweep

    insweep = [f.ref for f in sweep.frames if in_sweep(f, sweep.meta)]
    door_refs = sorted({d.ref for d in doors})
    out = []
    for phase in range(n):
        base = insweep[phase::n][:36]
        refs = sorted(set(base) | set(door_refs[phase % 3 :: 3]), key=lambda r: int(r[-10:-4]))
        if len(refs) < 8:
            continue
        rec = da3.reconstruct(sweep.dir, refs)
        up, _ = up_vector(rec, clients)
        pts, _ = rec.point_cloud(conf_quantile=0.5)
        fl = floor_level(pts, up)
        ks = door_top_scales(rec, doors, up, fl) if fl is not None else []
        if not ks:
            continue
        k = float(np.median([v for _, v in ks]))
        cams = np.array([rec.cam_to_world(i)[1] for i in range(len(rec.frame_refs))])
        out.append(fit_room(pts, up, k, "", cams))
    return out


async def _timed(ledger: Ledger, name: str, aw: Awaitable[Any]) -> Any:
    t = time.monotonic()
    try:
        return await aw
    finally:
        ledger.stages[name]["seconds"] += time.monotonic() - t


async def process_sweep(
    sweep_dir: Path,
    settings: Settings,
    live_detections: dict[int, list[Detection]] | None = None,
    reuse: bool = False,
    progress: Progress = None,
    on_packet: Callable[[ClaimPacket, dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[ClaimPacket, dict[str, Any]]:
    t_start = time.monotonic()
    ledger = Ledger()
    http = CachedHTTP(ROOT / "cache" / "http", ledger)
    vision = Vision(settings.gemini_api_key, settings.vision_model, ledger)
    stages = sweep_dir / "stages"
    stages.mkdir(exist_ok=True)

    async def say(msg: str) -> None:
        log.info("%s: %s", sweep_dir.name, msg)
        if progress:
            await progress(msg)

    try:
        sweep = load_sweep(sweep_dir)
        meta = sweep.meta
        country = (meta.get("country") or settings.default_country).upper()
        currency = (meta.get("currency") or settings.default_currency).upper()
        meta.update(country=country, currency=currency)
        det_path = stages / "detect.json"
        saved = _load_dets(det_path) if reuse and det_path.exists() else None
        # With --reuse, analyse exactly the frames already detected: re-selecting would add new
        # frames and spend new model calls (seen in testing: 38 extra calls on a reused bedroom sweep).
        prefer = set(saved) if saved else set(live_detections or {})
        frames = select_frames(sweep, settings.max_analysed_frames, prefer=prefer)
        if saved:
            frames = [f for f in frames if f.i in saved]
        frame_t = {f.i: f.t for f in sweep.frames}
        frame_region = {f.i: f.region for f in sweep.frames}
        clients = {f.ref: f.client for f in sweep.frames}
        await say(f"analysing {len(frames)} of {len(sweep.frames)} frames")

        # 1. detect
        failed_frames: list[str] = []
        with ledger.stage("detect"):
            if saved:
                dets = saved
            else:
                dets = await detect_frames(vision, sweep_dir, frames, live_detections, failed_frames)
            _dump(det_path, {k: [d.to_dict() for d in v] for k, v in dets.items()})

        # 2. track books
        with ledger.stage("track"):
            dets, cleaned = clean_book_detections(dets)
            tr = build_tracks(sweep_dir, frames, dets, kind="book")
            refine_tracks(sweep_dir, tr.tracks)
            books = [t for t in tr.tracks if not t.fragment]
            _dump(stages / "track.json", {"segments": tr.segments, "links": tr.links, "tracks": [
                {"id": t.id, "segment": t.segment, "row": t.shelf_row, "position": t.position, "fragment": t.fragment,
                 "spine_facing": t.spine_facing, "orientation": t.orientation, "dets": [d.id for d in t.dets]} for t in tr.tracks]})
        await say(f"{len(books)} books found on {len(tr.segments)} shelf segment(s)")

        # 3. read spines
        read_path = stages / "read.json"
        with ledger.stage("read"):
            if reuse and read_path.exists():
                # re-key saved reads through their detection ids: track ids can change between runs
                owner = {d.id: t.id for t in books for d in t.dets}
                reads_raw = {}
                for rs in json.loads(read_path.read_text()).values():
                    for r in rs:
                        tid = owner.get(r["det_id"])
                        if tid:
                            reads_raw.setdefault(tid, []).append(SpineRead(**{**r, "track_id": tid}))
            else:
                reads_raw = await read_spines(vision, sweep_dir, books)
                _dump(read_path, {k: [r.to_dict() for r in v] for k, v in reads_raw.items()})
            spill = drop_neighbour_spill(reads_raw, {t.id: (t.segment, t.shelf_row, t.position) for t in books})
            reads = {t.id: combine(reads_raw.get(t.id, [])) for t in books}
            texts = {tid: (r.printed_text if r else "") for tid, (r, _) in reads.items()}
            spill += merge_across_segments(tr.tracks, texts, frame_t)
            spill += absorb_weak_tracks(tr.tracks)
            spill += drop_lone_reviews(tr.tracks, tr.segments, frame_t, texts)
            for t in tr.tracks:  # a merged track's reads join the track it merged into
                if t.fragment and t.id in reads_raw:
                    reads.pop(t.id, None)
            books = [t for t in tr.tracks if not t.fragment]
            reads = {t.id: combine(reads_raw.get(t.id, []) + [r for d in t.dets for x in reads_raw.values() for r in x if r.det_id == d.id and r.track_id != t.id]) for t in books}

        # 4-8 overlap: the reconstruction runs on the GPU, item grouping on the model, and each book is
        # priced the moment it is identified, so the price-search rate limit is spent while the rest runs.
        recon = None
        recon_frames = [f.ref for f in frames]
        if len(recon_frames) > MAX_RECON_FRAMES:
            recon_frames = [recon_frames[int(i)] for i in np.linspace(0, len(recon_frames) - 1, MAX_RECON_FRAMES)]
        recon_task = None
        if da3.available() and len(recon_frames) >= 3:
            recon_task = asyncio.create_task(_timed(ledger, "reconstruct", asyncio.to_thread(da3.reconstruct, sweep_dir, recon_frames)))

        prov, excluded = statements_for_tracks(tr.tracks, sweep.statements, frame_t, frame_region)
        fx = FX(http)
        if settings.tinyfish_api_key:
            extractor = Vision(settings.gemini_api_key, settings.price_extract_model, ledger, max_concurrency=8)
            shop = TinyFishShopping(http, settings.tinyfish_api_key, extractor, settings.tinyfish_search_rpm, kind="book")
            item_shop = TinyFishShopping(http, settings.tinyfish_api_key, extractor, settings.tinyfish_search_rpm, kind="item")
            item_shop.limiter = shop.limiter  # one per-minute budget for the account
            price_source = "TinyFish search + page fetch"
        else:
            shop = item_shop = Shopping(http, settings.serper_api_key)
            price_source = "Serper (Google Shopping)"
        sem = asyncio.Semaphore(6)
        deadline = t_start + settings.price_deadline_s

        def out_of_time(tid: str) -> BookPrice:
            bp = BookPrice(tid)
            bp.replacement.note = bp.used.note = (f"not priced: the sweep's price-search budget ({settings.tinyfish_search_rpm}/min) "
                                                  f"ran out before the {settings.price_deadline_s:.0f} s deadline")
            return bp

        async def pb(tid: str) -> BookPrice:
            i = idents[tid]
            if time.monotonic() > deadline:
                return out_of_time(tid)
            async with sem:
                if time.monotonic() > deadline:
                    return out_of_time(tid)
                return await price_book(shop, fx, tid, i.title or "", i.author, i.isbn, country, currency)

        async def pi(it) -> ItemPrice:
            reason = None
            if it.is_artwork and not it.print_confirmed:
                reason = "art or portrait: defaults to appraisal unless the policyholder says it is a print"
            # material helps for furniture ("wooden bookshelf"), misleads for electronics ("laptop metal")
            # Search what a shop calls it ("double bed"), not how it looks ("bed with patterned duvet", which
            # found duvet covers in testing). Material helps for furniture, misleads for electronics.
            noun = it.product_type or it.label
            q = f"{it.material} {noun}".strip() if it.category in ("furniture", "shelving", "decor", "rug") and it.material else noun
            if it.print_confirmed:
                q = f"framed art print {it.label}"
            if time.monotonic() > deadline and not reason:
                return ItemPrice(it.id, note="not priced: price-search budget ran out")
            async with sem:
                return await price_item(item_shop, fx, it.id, q, it.brand_model, country, currency, reason)

        appraisal: dict[str, str] = {}
        price_tasks: dict[str, asyncio.Task] = {}
        catalogue = Catalogue(http, settings.google_books_key)
        idents = {}
        t_price = time.monotonic()

        async def identify_then_price(tid: str) -> None:
            res = await identify(catalogue, {tid: reads[tid]})
            ident = idents[tid] = res[tid]
            if ident.status != "identified" or tid in excluded:
                return
            r = appraisal_reason(currency, None, ident.year, bool(reads[tid][0] and reads[tid][0].looks_antiquarian), prov.get(tid, []))
            if r:
                appraisal[tid] = r  # never auto-priced
                return
            price_tasks[tid] = asyncio.create_task(pb(tid))

        item_dets = [d for f in frames for d in dets.get(f.i, []) if d.kind == "item"]

        async def group_and_build():
            groups = await group_items(vision, sweep_dir, item_dets, frame_t)
            its = build_items(groups)
            apply_statements(its, sweep.statements)
            return groups, its

        # Items are grouped now but priced after the books: in a 380-book sweep, item searches took the
        # price slots first and every book reached the deadline unpriced. Books are the main claim.
        items_task = asyncio.create_task(_timed(ledger, "items", group_and_build()))
        await _timed(ledger, "identify", asyncio.gather(*(identify_then_price(t.id) for t in books)))
        _dump(stages / "identify.json", {k: v.to_dict() for k, v in idents.items()})
        await say(f"{sum(1 for v in idents.values() if v.status == 'identified')} books identified; pricing")
        if recon_task is not None:
            try:
                recon = await recon_task
            except Exception as e:  # geometry is optional; the packet says why it is missing
                log.warning("reconstruction failed: %s", e)
                recon = None

        # scale and measure (needs identifications and the reconstruction)
        anchors = [d for f in frames for d in dets.get(f.i, []) if d.kind == "anchor"]
        with ledger.stage("measure"):
            model = build_scale(tr.tracks, idents, anchors, recon)
            up = floor_units = None
            if recon is not None:
                pts_all, _ = recon.point_cloud(conf_quantile=0.5)
                up, up_method = up_vector(recon, clients)
                floor_units = floor_level(pts_all, up)
                doors = [d for d in anchors if d.attrs.get("kind") == "door"]
                if floor_units is not None and doors:
                    for det_id, kk in door_top_scales(recon, doors, up, floor_units):
                        model.k_samples.append((Anchor("door", det_id, DOOR_TOP_CM, 1.0), kk))
            measures = measure(tr.tracks, model, recon)
            k, nk = model.k() if recon is not None else (None, 0)
            used = Counter(a.source for a, _ in model.k_samples) if recon is not None else Counter(
                a.source for s in model.seg_samples.values() for a, _ in s)
            scale_method = ", ".join(f"{n} {src} anchor{'s' if n > 1 else ''}" for src, n in used.items()) or "none"
            scale_method = (f"3D reconstruction ({da3.MODEL_ID.split('/')[-1]}) scaled by {scale_method}" if recon is not None
                            else f"shelf-plane pixels scaled by {scale_method}")
            _dump(stages / "measure.json", {"scale_method": scale_method, "k_cm_per_unit": k,
                                            "measurements": {kk: v.to_dict() for kk, v in measures.items()}})

        groups, items = await items_task
        add_dimensions(items, groups, recon, k)
        _dump(stages / "items.json", [it.to_dict() for it in items])
        prices_list = await asyncio.gather(*price_tasks.values())
        item_prices_list = await asyncio.gather(*(pi(it) for it in items))
        ledger.stages["price"]["seconds"] += time.monotonic() - t_price
        prices = {p.track_id: p for p in prices_list}
        item_prices = {p.item_id: p for p in item_prices_list}
        for tid, p in prices.items():  # expensive finds (new or used) go to a human, not into the total
            top = max((x for x in (p.replacement.amount, p.used.amount) if x is not None), default=None)
            r = appraisal_reason(currency, top, None, False, [])
            if r:
                appraisal[tid] = r
        _dump(stages / "price.json", {"books": {k_: v.to_dict() for k_, v in prices.items()},
                                      "items": {k_: v.to_dict() for k_, v in item_prices.items()}, "appraisal": appraisal,
                                      "source": price_source})
        await say(f"priced {sum(1 for p in prices.values() if p.replacement.amount is not None)} books and "
                  f"{sum(1 for p in item_prices.values() if p.low is not None)} items")

        # 9. room
        with ledger.stage("room"):
            if recon is None:
                room = RoomResult(notes=["3D reconstruction unavailable (install the geometry extra): room not measured"])
            else:
                cams = np.array([recon.cam_to_world(i)[1] for i in range(len(recon.frame_refs))])
                estimates = [fit_room(pts_all, up, k, scale_method, cams)]
                doors = [d for d in anchors if d.attrs.get("kind") == "door"]
                if doors and settings.room_ensemble > 1:
                    estimates += await _timed(ledger, "reconstruct", asyncio.to_thread(
                        room_ensemble, sweep, doors, clients, settings.room_ensemble))
                room = combine_estimates(estimates, scale_method + f"; up from {up_method}")
            room.shelved_wall_area_m2 = shelved_wall_area(items)
            _dump(stages / "room.json", room.to_dict())

        # 10. packet
        with ledger.stage("assemble"):
            packet = assemble(meta, tr.tracks, reads, idents, measures, prices, items, item_prices, room, excluded, appraisal)
            for f in failed_frames:
                packet.review_queue.append(ReviewEntry(ref_id="sweep", reason=f"frame could not be analysed, contents may be missing: {f}"))
            defects = check_packet(packet, sweep_dir)

        def write_metrics(extra: dict | None = None) -> dict:
            m = ledger.to_dict()
            m["live"] = live_session_cost(sweep.events)
            m["total_usd"] = round(m["total_usd"] + m["live"]["usd"], 4)
            m["processing_s"] = round(time.monotonic() - t_start, 1)
            m["notes"] = spill + cleaned
            m["price_source"] = price_source
            m["prices_dropped_unverified"] = getattr(shop, "dropped_unverified", 0) + (
                getattr(item_shop, "dropped_unverified", 0) if item_shop is not shop else 0)
            m.update(extra or {})
            _dump(sweep_dir / "metrics.json", m)
            return m

        # The packet is complete here: write it and hand it over before the second-country check.
        (sweep_dir / "claim_packet.json").write_text(packet.model_dump_json(indent=2))
        metrics = write_metrics({"time_to_packet_s": round(time.monotonic() - t_start, 1)})
        render(packet, sweep_dir, defects, None, metrics)
        info = {"metrics": metrics, "defects": defects, "summary": summary_text(packet)}
        await say("packet ready")
        if on_packet is not None:
            await on_packet(packet, info)

        # Locale check (same books priced in a second country). Not part of time-to-packet.
        locale = None
        sample = [b for b in packet.books if b.status == "identified" and b.replacement_cost.amount is not None][:LOCALE_SAMPLE]
        if sample:
            other = "US" if country != "US" else "IN"
            other_cur = CURRENCY_OF.get(other, "USD")

            async def alt(b):
                async with sem:
                    return b, await price_book(shop, fx, b.id, b.title or "", b.author, b.isbn, other, other_cur)

            alts = await _timed(ledger, "price", asyncio.gather(*(alt(b) for b in sample)))
            locale = {"country_a": country, "currency_a": currency, "country_b": other, "currency_b": other_cur,
                      "books": [{"id": b.id, "title": b.title, "a": b.replacement_cost.amount, "a_conv": b.replacement_cost.converted,
                                 "b": p.replacement.amount, "b_conv": p.replacement.converted, "b_source": p.replacement.source,
                                 "b_url": p.replacement.url, "b_retrieved_at": p.replacement.retrieved_at} for b, p in alts]}
            _dump(sweep_dir / "locale_comparison.json", locale)

        metrics = write_metrics({"time_to_packet_s": metrics["time_to_packet_s"]})
        render(packet, sweep_dir, defects, locale, metrics)
        info["metrics"] = metrics
        return packet, info
    finally:
        await http.close()
