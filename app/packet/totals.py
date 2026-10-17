"""Totals are computed here, in code, from the lines. Never by a language model."""

from __future__ import annotations

from app.schema import Book, ClaimPacket, Item, Totals


def _r(x: float) -> float:
    return round(x + 0.0, 2)


def book_is_counted_in_value(book: Book) -> bool:
    """A book contributes money only if it is identified (not flagged for appraisal) and priced."""
    return book.status == "identified"


def item_bounds(item: Item) -> tuple[float, float] | None:
    """(low, high) contribution of an item, or None if it is excluded from totals."""
    if item.status == "needs_appraisal":
        return None
    rc = item.replacement_cost
    if rc.low is None and rc.high is None:
        return None
    low = rc.low if rc.low is not None else rc.high
    high = rc.high if rc.high is not None else rc.low
    return float(low), float(high)


def compute_totals(books: list[Book], items: list[Item]) -> Totals:
    excluded = 0
    repl = 0.0
    used = 0.0
    for b in books:
        if not book_is_counted_in_value(b):
            excluded += 1
            continue
        has_any = False
        if b.replacement_cost.amount is not None:
            repl += b.replacement_cost.amount
            has_any = True
        if b.used_value.amount is not None:
            used += b.used_value.amount
            has_any = True
        if not has_any:
            excluded += 1

    low = high = 0.0
    for it in items:
        bounds = item_bounds(it)
        if bounds is None:
            excluded += 1
            continue
        low += bounds[0]
        high += bounds[1]

    # Shelf run counts upright spines only; a flat stack occupies shelf by its depth, not its spines.
    run_cm = sum(b.spine_thickness_cm or 0 for b in books if b.orientation == "upright")

    return Totals(
        book_count=len(books),
        books_identified=sum(1 for b in books if b.title),
        books_unidentified=sum(1 for b in books if not b.title),
        shelf_run_m=_r(run_cm / 100),
        books_replacement_cost=_r(repl),
        books_used_value=_r(used),
        items_replacement_cost_low=_r(low),
        items_replacement_cost_high=_r(high),
        excluded_from_totals=excluded,
    )


def finalize(packet: ClaimPacket) -> ClaimPacket:
    """Recompute totals from the lines. Call this last, after every line edit."""
    packet.totals = compute_totals(packet.books, packet.items)
    return packet
