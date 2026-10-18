"""Stage 9: assemble the claim packet from stage outputs, apply the policyholder's statements, and
fill the review queue. Totals are computed by app.packet.totals, never by a model."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from rapidfuzz import fuzz

from app.geometry.room import RoomResult
from app.packet.totals import finalize
from app.pipeline.identify import Identification
from app.pipeline.items import ItemRecord
from app.pipeline.price import BookPrice, ItemPrice
from app.pipeline.read import SpineRead
from app.pipeline.scale import Measurement
from app.pipeline.track import Track
from app.schema import (
    Book, BookReplacementCost, BookUsedValue, ClaimPacket, Dimensions, Excluded, Item, ItemReplacementCost,
    ReviewEntry, Room, Sweep,
)

LOW_ID_CONF = 0.75
PROVENANCE = re.compile(r"\b(signed|autograph|first edition|1st edition|rare|inscribed|limited edition)\b", re.I)


def shelf_label(t: Track) -> str | None:
    if t.shelf_row is None:
        return None
    return f"Unit {chr(65 + t.segment) if t.segment < 26 else t.segment} shelf {t.shelf_row + 1}"


def frame_ref_with_box(ref: str, box: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return f"{ref}#xywh={x0},{y0},{x1 - x0},{y1 - y0}"


def _track_time(t: Track, frame_t: dict[int, float]) -> float:
    ts = sorted(frame_t.get(d.frame_i, 0.0) for d in t.dets)
    return ts[len(ts) // 2]


def statements_for_tracks(tracks: list[Track], statements: list[dict[str, Any]], frame_t: dict[int, float],
                          frame_region: dict[int, str | None]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """(provenance statements per track, exclusion statement per track).

    A statement about "this book" applies to the book in view when it was said: the track whose
    frames are closest in time. An exclusion ("skip that shelf") applies to every book seen within
    8 s of the statement.
    """
    prov: dict[str, list[str]] = {}
    excl: dict[str, str] = {}
    live = [t for t in tracks if not t.fragment]
    for s in statements:
        text = s.get("statement", "")
        if s.get("kind") == "exclusion":
            # Only books on screen when it was said (within 8 s). The agent's region label can be
            # stale (it rarely re-labelled shelves in testing), so it could exclude a whole bookcase.
            for t in live:
                if abs(_track_time(t, frame_t) - s["t"]) <= 8:
                    excl[t.id] = text
        elif s.get("kind") in ("provenance", "correction") and PROVENANCE.search(text) and live:
            t = min(live, key=lambda x: abs(_track_time(x, frame_t) - s["t"]))
            if abs(_track_time(t, frame_t) - s["t"]) <= 8:
                prov.setdefault(t.id, []).append(text)
    return prov, excl


def assemble(
    sweep_meta: dict[str, Any],
    tracks: list[Track],
    reads: dict[str, tuple[SpineRead | None, float]],
    idents: dict[str, Identification],
    measures: dict[str, Measurement],
    prices: dict[str, BookPrice],
    items: list[ItemRecord],
    item_prices: dict[str, ItemPrice],
    room: RoomResult,
    excluded: dict[str, str],
    appraisal: dict[str, str],
) -> ClaimPacket:
    packet = ClaimPacket(sweep=Sweep(
        id=sweep_meta["id"], captured_at=sweep_meta.get("captured_at", ""), device=sweep_meta.get("device", ""),
        duration_s=round((sweep_meta.get("sweep_ended_t") or 0) - (sweep_meta.get("sweep_started_t") or 0), 1),
        country=sweep_meta.get("country", ""), currency=sweep_meta.get("currency", ""),
    ))
    review: list[ReviewEntry] = []

    title_count = Counter((i.title or "").lower() for i in idents.values() if i.status == "identified" and i.title)

    for t in sorted((t for t in tracks if not t.fragment), key=lambda t: (t.segment, t.shelf_row or 0, t.position or 0)):
        best = t.best(1)[0]
        ref = frame_ref_with_box(best.ref, best.box)
        if t.id in excluded:
            packet.excluded_by_policyholder.append(Excluded(ref_id=t.id, frame_ref=ref, statement=excluded[t.id]))
            continue
        read, _ = reads.get(t.id, (None, 0.0))
        ident = idents.get(t.id)
        meas = measures.get(t.id)
        bp = prices.get(t.id)
        status = ident.status if ident else "unidentified"
        reason = appraisal.get(t.id)
        if reason and status == "identified":
            status = "needs_appraisal"
        book = Book(
            id=t.id, shelf=shelf_label(t), position=t.position, frame_ref=ref, status=status,
            title=ident.title if ident and status != "unidentified" else None,
            author=ident.author if ident and status != "unidentified" else None,
            edition=ident.edition if ident and status != "unidentified" else None,
            isbn=ident.isbn if ident and status != "unidentified" else None,
            spine_height_cm=meas.height_cm if meas else None, spine_thickness_cm=meas.thickness_cm if meas else None,
            id_confidence=ident.id_confidence if ident else 0.0,
            orientation="flat" if t.orientation == "flat" else "upright",
            spine_text_read=(read.printed_text or None) if read else None,
            catalogue_url=ident.catalogue_url if ident and status != "unidentified" else None,
            measurement_method=meas.method if meas else None,
            appraisal_reason=reason if status == "needs_appraisal" else None,
        )
        if status == "identified" and bp:
            r, u = bp.replacement, bp.used
            if r.amount is not None:
                book.replacement_cost = BookReplacementCost(amount=r.amount, source=r.source, url=r.url, retrieved_at=r.retrieved_at,
                                                            converted=r.converted, original_amount=r.original_amount,
                                                            original_currency=r.original_currency)
            if u.amount is not None:
                book.used_value = BookUsedValue(amount=u.amount, source=u.source, url=u.url, retrieved_at=u.retrieved_at,
                                                condition_assumed=u.condition_assumed, converted=u.converted,
                                                original_amount=u.original_amount, original_currency=u.original_currency)
        packet.books.append(book)

        # review queue
        if not t.spine_facing:
            review.append(ReviewEntry(ref_id=t.id, reason="shelved backwards (page edges facing out): spine not visible"))
        elif status == "unidentified":
            review.append(ReviewEntry(ref_id=t.id, reason="; ".join(ident.reasons) if ident and ident.reasons else "spine not readable"))
        if status == "needs_appraisal":
            review.append(ReviewEntry(ref_id=t.id, reason=f"needs human appraisal: {reason}"))
        if status == "identified":
            if book.id_confidence < LOW_ID_CONF:
                review.append(ReviewEntry(ref_id=t.id, reason=f"identification confidence {book.id_confidence:.2f}: " + "; ".join(ident.reasons if ident else [])))
            if bp and bp.used.amount is not None and bp.replacement.amount is not None and bp.used.amount > bp.replacement.amount:
                review.append(ReviewEntry(ref_id=t.id, reason="used value found above replacement cost: check the listings"))
            if bp and bp.replacement.amount is None and bp.used.amount is not None:
                review.append(ReviewEntry(ref_id=t.id, reason="only a used price was found; no new copy is on sale"))
            if bp and bp.replacement.amount is None and bp.used.amount is None:
                review.append(ReviewEntry(ref_id=t.id, reason=f"no price found: {bp.replacement.note}; excluded from totals"))
            if title_count.get((book.title or "").lower(), 0) > 1:
                review.append(ReviewEntry(ref_id=t.id, reason="same title appears more than once: a second copy, or counted twice"))
        if meas and meas.method.startswith("standard-format"):
            review.append(ReviewEntry(ref_id=t.id, reason="no scale anchor on this shelf: dimensions assume a 21 cm median book"))
        if t.at_segment_edge:
            review.append(ReviewEntry(ref_id=t.id, reason="seen once, at the edge of a capture segment: may duplicate a neighbour"))

    for it in items:
        ip = item_prices.get(it.id, ItemPrice(it.id))
        status = ip.status
        item = Item(
            id=it.id, category=it.category, description=f"{it.label}" + (f" ({it.material})" if it.material else ""),
            brand_model=it.brand_model, frame_ref=it.frame_ref,
            dimensions_cm=Dimensions(**it.dims_cm), status=status, confidence=it.confidence, material=it.material or None,
            dimensions_method=it.dims_method or None,
            appraisal_reason=ip.note if status == "needs_appraisal" else None,
            replacement_cost=ItemReplacementCost(low=ip.low, high=ip.high, source=ip.source, url=ip.url, retrieved_at=ip.retrieved_at)
            if ip.low is not None else ItemReplacementCost(),
        )
        packet.items.append(item)
        if status == "needs_appraisal":
            review.append(ReviewEntry(ref_id=it.id, reason=f"needs human appraisal: {ip.note}"))
        elif ip.low is None:
            review.append(ReviewEntry(ref_id=it.id, reason=f"no price: {ip.note}; excluded from totals"))
        if it.confidence < 0.6:
            review.append(ReviewEntry(ref_id=it.id, reason=f"seen in {len(it.sightings)} frame(s) only; detection confidence {it.confidence}"))
        if it.dims_cm.get("w") is None:
            review.append(ReviewEntry(ref_id=it.id, reason="dimensions not measured"))

    packet.room = Room(
        length_m=room.length_m, width_m=room.width_m, height_m=room.height_m, floor_area_m2=room.floor_area_m2,
        wall_area_m2=room.wall_area_m2, shelved_wall_area_m2=room.shelved_wall_area_m2, scale_method=room.scale_method,
        confidence=room.confidence, floor_area_ft2=room.floor_area_ft2, wall_area_ft2=room.wall_area_ft2,
        shape_notes=room.shape_notes,
    )
    for n in room.notes:
        review.append(ReviewEntry(ref_id="room", reason=n))

    packet.review_queue = review
    return finalize(packet)


def _n(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def summary_text(p: ClaimPacket) -> str:
    """What the agent reads back. Every number here comes from the packet, computed in code."""
    t, cur = p.totals, p.sweep.currency
    parts = [f"I counted {t.book_count} books: {t.books_identified} identified and {t.books_unidentified} I could not read."]
    if t.books_replacement_cost:
        parts.append(f"Replacing the priced books new would cost about {t.books_replacement_cost:,.0f} {cur}; "
                     f"their used value is about {t.books_used_value:,.0f} {cur}.")
    if p.items:
        if t.items_replacement_cost_high:
            parts.append(f"I logged {_n(len(p.items), 'other item')}, worth between {t.items_replacement_cost_low:,.0f} and "
                         f"{t.items_replacement_cost_high:,.0f} {cur} to replace.")
        else:
            parts.append(f"I logged {_n(len(p.items), 'other item')}.")
    r = p.room
    if r.floor_area_m2:
        parts.append(f"The room measures about {r.length_m} by {r.width_m} metres, a floor area of {r.floor_area_m2} square metres.")
    else:
        parts.append("I could not measure the room from this sweep.")
    n_review = len({e.ref_id for e in p.review_queue})
    parts.append(f"{_n(t.excluded_from_totals, 'line')} {'is' if t.excluded_from_totals == 1 else 'are'} not in the totals, "
                 f"and {_n(n_review, 'line')} need{'s' if n_review == 1 else ''} a human to review.")
    return " ".join(parts)


def match_score(a: str, b: str) -> float:
    return fuzz.token_set_ratio(a.lower(), b.lower())
