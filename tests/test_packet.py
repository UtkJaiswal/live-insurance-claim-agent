from app.packet.integrity import check_packet
from app.packet.totals import compute_totals, finalize
from app.schema import ClaimPacket, ReviewEntry, Sweep
from tests.conftest import make_book, make_item


def test_totals_sum_only_priced_identified_lines():
    books = [
        make_book(1, repl=500, used=200),
        make_book(2, repl=300),  # no used value: still counted for replacement
        make_book(3, status="needs_appraisal"),
        make_book(4, status="unidentified"),
        make_book(5),  # identified but unpriced
        make_book(6, orientation="flat", thickness=3.0, repl=100, used=50),
    ]
    items = [
        make_item(1, low=1000, high=1000),
        make_item(2, status="range", low=2000, high=3000),
        make_item(3, status="needs_appraisal", category="art"),
    ]
    t = compute_totals(books, items)
    assert t.book_count == 6
    assert t.books_identified == 5
    assert t.books_unidentified == 1
    assert t.books_replacement_cost == 900
    assert t.books_used_value == 250
    assert t.items_replacement_cost_low == 3000
    assert t.items_replacement_cost_high == 4000
    assert t.excluded_from_totals == 4  # appraisal, unidentified, unpriced book, appraisal item
    assert t.shelf_run_m == 0.1  # five upright books x 2 cm; the flat one does not add to run


def test_clean_packet_has_no_defects(sweep_dir):
    p = ClaimPacket(
        sweep=Sweep(id="s", captured_at="2026-10-18T10:00:00+00:00"),
        books=[make_book(1, repl=500, used=200), make_book(2, status="unidentified")],
        items=[make_item(1, low=10, high=20, status="range")],
        review_queue=[ReviewEntry(ref_id="b2", reason="spine unreadable")],
    )
    assert check_packet(finalize(p), sweep_dir) == []


def test_integrity_catches_unsourced_prices_and_bad_refs(sweep_dir):
    b = make_book(1, repl=500)
    b.replacement_cost.url = None
    flagged = make_book(2, status="needs_appraisal")
    flagged.used_value.amount = 999
    missing = make_book(3, repl=1)
    missing.frame_ref = "frames/nope.jpg"
    p = finalize(ClaimPacket(sweep=Sweep(id="s", captured_at="x"), books=[b, flagged, missing]))
    p.totals.books_replacement_cost += 1  # tampered total
    defects = "\n".join(check_packet(p, sweep_dir))
    assert "b1: replacement_cost has an amount without source" in defects
    assert "b2: flagged for appraisal but auto-priced" in defects
    assert "b3: frame_ref" in defects
    assert "totals do not match" in defects
