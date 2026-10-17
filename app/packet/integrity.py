"""Checks that every filled number in a packet has evidence behind it.

Returns a list of defects. An empty list means the packet honours the brief's three rules:
evidence for every filled field, every frame_ref openable, totals equal to the sum of the lines.
"""

from __future__ import annotations

from pathlib import Path

from app.packet.totals import compute_totals
from app.schema import ClaimPacket


def _frame_exists(frame_ref: str, sweep_dir: Path) -> bool:
    # frame_ref is "<relative path>" or "<relative path>#t=<seconds>"
    path = frame_ref.split("#", 1)[0]
    return bool(path) and (sweep_dir / path).is_file()


def check_packet(packet: ClaimPacket, sweep_dir: Path) -> list[str]:
    defects: list[str] = []

    for b in packet.books:
        if not _frame_exists(b.frame_ref, sweep_dir):
            defects.append(f"{b.id}: frame_ref {b.frame_ref!r} does not open")
        if b.status == "unidentified" and (b.title or b.author or b.isbn):
            defects.append(f"{b.id}: unidentified book carries a title/author/isbn")
        if b.status == "needs_appraisal" and (
            b.replacement_cost.amount is not None or b.used_value.amount is not None
        ):
            defects.append(f"{b.id}: flagged for appraisal but auto-priced")
        for label, price in (("replacement_cost", b.replacement_cost), ("used_value", b.used_value)):
            if price.amount is not None and not (price.source and price.url and price.retrieved_at):
                defects.append(f"{b.id}: {label} has an amount without source/url/retrieved_at")
        if b.used_value.amount is not None and not b.used_value.condition_assumed:
            defects.append(f"{b.id}: used_value has no condition_assumed")

    for it in packet.items:
        if not _frame_exists(it.frame_ref, sweep_dir):
            defects.append(f"{it.id}: frame_ref {it.frame_ref!r} does not open")
        rc = it.replacement_cost
        priced = rc.low is not None or rc.high is not None
        if it.status == "needs_appraisal" and priced:
            defects.append(f"{it.id}: flagged for appraisal but auto-priced")
        if priced and not (rc.source and rc.url and rc.retrieved_at):
            defects.append(f"{it.id}: price without source/url/retrieved_at")
        if rc.low is not None and rc.high is not None and rc.low > rc.high:
            defects.append(f"{it.id}: low > high")

    expected = compute_totals(packet.books, packet.items)
    if expected != packet.totals:
        defects.append(f"totals do not match lines: expected {expected.model_dump()}")

    ids = {b.id for b in packet.books} | {i.id for i in packet.items} | {"room", "sweep"}
    for entry in packet.review_queue:
        if entry.ref_id not in ids:
            defects.append(f"review_queue references unknown id {entry.ref_id!r}")

    return defects
