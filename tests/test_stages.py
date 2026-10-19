"""Offline tests for each pipeline stage: no network, no model calls."""

import asyncio

import cv2
import numpy as np

from app.pipeline.assemble import assemble, statements_for_tracks
from app.pipeline.detect import Detection, parse_detections, to_pixels
from app.pipeline.frames import Frame, SweepData, select_frames
from app.pipeline.identify import Edition, Identification, choose, parse_dims, split_volume
from app.pipeline.items import ItemRecord, _validate_groups, apply_statements
from app.pipeline.price import (
    FX, BookPrice, ItemPrice, Listing, PriceResult, appraisal_reason, listing_matches, parse_price, price_book,
    several_books,
)
from app.pipeline.read import SpineRead, combine, drop_neighbour_spill, verify
from app.pipeline.scale import Anchor, ScaleModel, build_scale, measure
from app.pipeline.track import Track, build_tracks
from app.geometry.room import RoomResult
from app.packet.integrity import check_packet


def _frame(i, t=0.0, ref=None, sharp=0.15):
    return Frame(i=i, ref=ref or f"frames/f_{i + 1:06d}.jpg", t=t, quality={"sharpness": sharp, "edge_coverage": 0.2, "brightness": 120},
                 client={}, region=None, sweeping=True)


def _det(i, frame_i, box, kind="book", **attrs):
    attrs = {"orientation": "upright", "spine_facing": True, "cut_off": False, **attrs}
    return Detection(f"d{frame_i}_{i}", frame_i, f"frames/f_{frame_i + 1:06d}.jpg", kind, box, attrs)


# ---------- detect ----------

def test_box_conversion_and_parsing():
    assert to_pixels([100, 200, 500, 300], 2000, 4000) == (400.0, 400.0, 600.0, 2000.0)
    assert to_pixels([500, 300, 100, 200], 1000, 1000) == (200.0, 100.0, 300.0, 500.0)  # swapped corners
    raw = {"books": [{"box_2d": [0, 0, 1000, 10], "orientation": "upright", "spine_facing": True, "cut_off": False},
                     {"box_2d": [0, 0, 1, 1], "orientation": "upright", "spine_facing": True, "cut_off": False}],
           "items": [], "anchors": []}
    dets = parse_detections(raw, _frame(0), 1000, 1000)
    assert len(dets) == 1 and dets[0].kind == "book"  # the 1x1 px box is dropped


# ---------- track ----------

def _shelf_image(seed=0, w=3000, h=1600):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    x, boxes = 40, []
    while x < w - 80:
        bw = int(rng.integers(50, 120))
        img[200:1400, x : x + bw] = rng.integers(30, 220, 3)
        for _ in range(12):  # text-like texture so features match
            y = int(rng.integers(260, 1340))
            cv2.putText(img, chr(int(rng.integers(65, 90))), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        boxes.append((x, 200, x + bw, 1400))
        x += bw + 4
    return img, boxes


def test_tracks_count_each_book_once_across_overlapping_frames(tmp_path):
    img, boxes = _shelf_image()
    (tmp_path / "frames").mkdir()
    frames, dets = [], {}
    for k, x0 in enumerate([0, 700, 1400]):
        crop = img[:, x0 : x0 + 1600]
        ref = f"frames/f_{k + 1:06d}.jpg"
        cv2.imwrite(str(tmp_path / ref), crop)
        frames.append(_frame(k, t=float(k), ref=ref))
        dets[k] = []
        for n, (a, y0, b, y1) in enumerate(boxes):
            if a >= x0 and b <= x0 + 1600:
                dets[k].append(_det(n, k, (a - x0, y0, b - x0, y1)))
            elif a < x0 + 1600 and b > x0:  # cut by the frame edge
                dets[k].append(_det(n, k, (max(a, x0) - x0, y0, min(b, x0 + 1600) - x0, y1), cut_off=True))
    res = build_tracks(tmp_path, frames, dets)
    visible = sum(1 for (a, _, b, _) in boxes if b > 0 and a < 3000)
    assert len(res.segments) == 1
    assert len([t for t in res.tracks if not t.fragment]) == visible
    positions = [t.position for t in sorted(res.tracks, key=lambda t: t.ref_box[0])]
    assert positions == sorted(positions)


# ---------- read ----------

def test_verify_drops_fields_not_in_printed_text():
    r = verify(SpineRead("t", "d", "f", printed_text="WILLIAMS | Shame and Necessity", title="Shame and Necessity",
                         author="Bernard Williams", legibility="clear"))
    assert r.title == "Shame and Necessity" and r.author == "" and "Bernard" in r.dropped


def test_combine_prefers_better_detection_and_agreement():
    a = SpineRead("t", "d1", "f", printed_text="Foo", title="The Foo Book", legibility="clear", rank=0)
    b = SpineRead("t", "d2", "f", printed_text="Bar", title="A Much Longer Neighbour Title", legibility="clear", rank=1)
    best, conf = combine([a, b])
    assert best is a and conf < 0.8  # disagreement lowers confidence; the longer title does not win
    best, conf = combine([a, SpineRead("t", "d3", "f", printed_text="Foo", title="The Foo Book", legibility="clear", rank=1)])
    assert conf >= 0.9


def test_neighbour_spill_is_removed():
    reads = {
        "b1": [SpineRead("b1", "x", "f", printed_text="A", title="Alpha Book", legibility="clear", rank=0),
               SpineRead("b1", "y", "f", printed_text="B", title="Beta Book", legibility="clear", rank=1)],
        "b2": [SpineRead("b2", "z", "f", printed_text="B", title="Beta Book", legibility="clear"),
               SpineRead("b2", "w", "f", printed_text="B", title="Beta Book", legibility="clear")],
    }
    notes = drop_neighbour_spill(reads, {"b1": (0, 0, 1), "b2": (0, 0, 2)})
    assert notes and [r.title for r in reads["b1"]] == ["Alpha Book"]


# ---------- identify ----------

def test_dims_and_volume_parsing():
    assert parse_dims("7.4 x 4.5 x 1.1 inches") == (18.8, 11.43, 2.79)
    assert parse_dims("24 x 16 x 2 centimeters") == (24.0, 16.0, 2.0)
    assert split_volume("Philosophen der Antike II") == ("Philosophen der Antike", "2")
    assert split_volume("Catch-22")[0] == "Catch-22" or split_volume("Catch-22")[1] == "22"


def _read(title, author="", volume="", legibility="clear", publisher=""):
    return SpineRead("t1", "d", "f", printed_text=f"{author} | {title} | {volume}", title=title, author=author,
                     volume=volume, publisher=publisher, legibility=legibility)


def test_choose_respects_volume_and_edition_evidence():
    cands = [Edition("openlibrary", "u1", "Philosophen der Antike I", ["Friedo Ricken"], "Kohlhammer", "1996", "111", 18.8, 11.4, 2.8),
             Edition("openlibrary", "u2", "Philosophen der Antike II", ["Friedo Ricken"], "Kohlhammer", "1996", "222")]
    ident = choose(_read("Philosophen der Antike", "Ricken", "II", publisher="Kohlhammer"), 0.9, cands, "t1", "2026-10-17")
    assert ident.title == "Philosophen der Antike II" and ident.isbn == "222"


def test_choose_without_catalogue_uses_printed_title_with_lower_confidence():
    ident = choose(_read("A Very Local Book Of Poems", "Sharma"), 0.95, [], "t1", None)
    assert ident.status == "identified" and ident.id_confidence < 0.7 and "no catalogue record" in ident.reasons[0]
    ident = choose(_read("Xyz", "", legibility="partial"), 0.45, [], "t1", None)
    assert ident.status == "unidentified" and ident.title is None


def test_short_spine_title_needs_unique_author_match():
    cands = [Edition("openlibrary", "u", "Die Natur und die Griechen", ["Erwin Schrödinger"])]
    ident = choose(_read("Griechen", "Schrödinger"), 0.95, cands, "t1", None)
    assert ident.title == "Die Natur und die Griechen" and ident.id_confidence <= 0.6


# ---------- scale ----------

def _track(tid, h_px, w_px=40.0, segment=0, frame_i=0, orientation="upright"):
    d = _det(0, frame_i, (0, 0, w_px, h_px), orientation=orientation)
    d.id = f"{tid}_d"
    t = Track(tid, [d], segment=segment, ref_box=(0, 0, w_px, h_px), shelf_row=0, position=1)
    return t


def test_leave_one_out_scale_and_format_prior():
    tracks = [_track("a", 1000), _track("b", 1000), _track("c", 500)]
    idents = {"a": Identification("a", "identified", dims_cm={"height": 20.0}),
              "b": Identification("b", "identified", dims_cm={"height": 22.0})}
    m = measure(tracks, build_scale(tracks, idents, []))
    assert m["a"].height_cm == 22.0  # measured with b's anchor only
    assert m["b"].height_cm == 20.0
    assert m["c"].height_cm == 10.5  # median of both anchors
    lone = [_track("x", 800, segment=3)]
    mm = measure(lone, ScaleModel())
    assert mm["x"].height_cm == 21.0 and mm["x"].confidence == 0.3


def test_a4_sheet_beats_catalogue_anchor():
    tracks = [_track("a", 1000)]
    idents = {"a": Identification("a", "identified", dims_cm={"height": 50.0})}  # a wrong catalogue entry
    a4 = Detection("s", 0, "frames/f_000001.jpg", "anchor", (0, 0, 700, 1485), {"kind": "a4_sheet", "fully_visible": True})
    model = build_scale(tracks, idents, [a4])
    s, n = model.seg(0, exclude_track="zzz")
    assert abs(s - 29.7 / 1485) < 1e-9 and n == 1


# ---------- price ----------

def test_parse_price_and_listing_match():
    assert parse_price("₹1,299.00", "INR") == (1299.0, "INR")
    assert parse_price("$12.99", "INR") == (12.99, "USD")
    assert parse_price("Rs. 450", "USD") == (450.0, "INR")
    assert listing_matches("Shame and Necessity by Bernard Williams (Paperback)", "Shame and Necessity", "Bernard Williams")
    assert not listing_matches("Shame and Necessity Kindle Edition", "Shame and Necessity", "Bernard Williams")
    assert not listing_matches("The Necessity of Art", "Shame and Necessity", "Bernard Williams")


def test_appraisal_rules():
    assert appraisal_reason("INR", 900, None, False, ["it's a signed copy"]).startswith("policyholder said")
    assert "1920" in appraisal_reason("INR", None, "1920", False, [])
    assert appraisal_reason("INR", 12000, None, False, []) and appraisal_reason("INR", 400, "2001", False, []) is None


class FakeShopping:
    enabled = True

    def __init__(self, by_market):
        self.by_market = by_market

    async def search(self, query, market):
        return self.by_market.get(market, []), "2026-10-17T10:00:00+00:00", f"https://shop/{market}?q={query}"


class FakeFX(FX):
    def __init__(self):
        pass

    async def rate(self, frm, to):
        return (95.0, "2026-10-16", "fx") if (frm, to) == ("USD", "INR") else None


def test_price_book_local_then_converted_fallback():
    local = [Listing("Shame and Necessity Williams paperback", "Store A", "https://a", 500, "INR", False, "IN"),
             Listing("Shame and Necessity Williams", "Store B", "https://b", 700, "INR", False, "IN"),
             Listing("Shame and Necessity Williams used", "Store C", "https://c", 300, "INR", True, "IN")]
    bp = asyncio.run(price_book(FakeShopping({"IN": local}), FakeFX(), "b1", "Shame and Necessity", "Bernard Williams", None, "IN", "INR"))
    assert bp.replacement.amount == 500 and not bp.replacement.converted and bp.replacement.n_listings == 2  # lower quartile
    assert bp.used.amount == 300 and bp.used.condition_assumed
    us = [Listing("Shame and Necessity Williams", "US Store", "https://us", 20.0, "USD", False, "US")]
    bp = asyncio.run(price_book(FakeShopping({"US": us}), FakeFX(), "b1", "Shame and Necessity", "Williams", None, "IN", "INR"))
    assert bp.replacement.converted and bp.replacement.amount == 1900.0 and bp.replacement.original_currency == "USD"
    assert bp.used.amount is None and bp.used.note == "no matching listing found"


# ---------- items ----------

def test_group_validation():
    assert _validate_groups([{"ids": [1, 2]}, {"ids": [3]}], 3) == [[1, 2], [3]]
    assert _validate_groups([{"ids": [1, 2]}, {"ids": [2, 3]}], 3) is None  # id used twice
    assert _validate_groups([{"ids": [1]}], 3) == [[1], [2], [3]]  # omitted ids stay separate items


def test_art_print_statement():
    art = ItemRecord("item_001", "art", "framed portrait above desk", "wood", None, True, "frames/f_000001.jpg", ["s"])
    apply_statements([art], [{"subject": "portrait above the desk", "statement": "that's just a print"}])
    assert art.print_confirmed
    art2 = ItemRecord("item_002", "art", "oil painting", "canvas", None, True, "frames/f_000001.jpg", ["s"])
    apply_statements([art2], [{"subject": "painting", "statement": "it is an original oil, not a print"}])
    assert not art2.print_confirmed


# ---------- frames ----------

def test_select_frames_prefers_live_frames_and_fills_gaps():
    frames = [_frame(i, t=float(i)) for i in range(20)]
    sweep = SweepData(dir=None, meta={}, events=[], frames=frames)
    chosen = select_frames(sweep, max_frames=90, prefer={0, 2, 4})
    ids = [f.i for f in chosen]
    assert {0, 2, 4} <= set(ids)
    assert all(abs(a.t - b.t) > 0 for a, b in zip(chosen, chosen[1:]))
    assert not any(i in ids for i in (1, 3, 5, 6, 7))  # within 3 s of a preferred frame


# ---------- assemble ----------

def test_assemble_end_to_end_is_integrity_clean(tmp_path):
    (tmp_path / "frames").mkdir()
    cv2.imwrite(str(tmp_path / "frames/f_000001.jpg"), np.zeros((100, 100, 3), np.uint8))
    tracks = [_track("book_001", 1000), _track("book_002", 900), _track("book_003", 800)]
    for n, t in enumerate(tracks, 1):
        t.position = n
    reads = {"book_001": (SpineRead("book_001", "d", "f", printed_text="X"), 0.9), "book_002": (None, 0.0), "book_003": (None, 0.0)}
    idents = {"book_001": Identification("book_001", "identified", title="X", author="Y", id_confidence=0.9),
              "book_002": Identification("book_002", "unidentified", reasons=["spine not readable"]),
              "book_003": Identification("book_003", "identified", title="Old", id_confidence=0.9)}
    prices = {"book_001": BookPrice("book_001", replacement=PriceResult(amount=500, currency="INR", source="S", url="https://s",
                                                                        retrieved_at="2026-10-17T10:00:00+00:00"))}
    items = [ItemRecord("item_001", "art", "portrait", "", None, True, "frames/f_000001.jpg", ["x"])]
    item_prices = {"item_001": ItemPrice("item_001", status="needs_appraisal", note="art")}
    p = assemble({"id": "s", "currency": "INR", "country": "IN"}, tracks, reads, idents, {}, prices, items, item_prices,
                 RoomResult(notes=["no scale"]), {}, {"book_003": "published 1920 (before 1950)"})
    assert check_packet(p, tmp_path) == []
    assert p.totals.books_replacement_cost == 500 and p.totals.excluded_from_totals == 3
    assert {b.id: b.status for b in p.books}["book_003"] == "needs_appraisal"
    reasons = " ".join(e.reason for e in p.review_queue)
    assert "appraisal" in reasons and "no scale" in reasons and "spine not readable" in reasons


def test_statements_map_to_books_by_time_and_region():
    a, b = _track("a", 100, frame_i=0), _track("b", 100, frame_i=5)
    frame_t = {0: 10.0, 5: 40.0}
    prov, excl = statements_for_tracks([a, b], [
        {"kind": "provenance", "statement": "this one is a signed first edition", "t": 41.0},
        {"kind": "exclusion", "statement": "skip that shelf, not mine", "t": 11.0, "region": None},
    ], frame_t, {0: None, 5: None})
    assert "b" in prov and "a" not in prov
    assert set(excl) == {"a"}


def test_generic_title_without_author_is_not_resolved():
    cands = [Edition("openlibrary", "u1", "Java", ["Herbert Schildt"]), Edition("openlibrary", "u2", "Java", ["Rogers Cadenhead"])]
    ident = choose(_read("Java", ""), 0.95, cands, "t1", None)
    assert ident.status == "unidentified" and ident.title is None and "cannot be determined" in ident.reasons[0]


def test_printed_publisher_rules_out_a_contradicting_work():
    cands = [Edition("openlibrary", "u1", "HTML5 and CSS3", ["Elizabeth Castro"], "Peachpit Press", "2012", "111"),
             Edition("openlibrary", "u2", "HTML5 and CSS3", ["Brian Hogan"], "Pragmatic Bookshelf", "2010", "222")]
    ident = choose(_read("HTML5 and CSS3", "Hogan", publisher="Manning"), 0.95, cands, "t1", None)
    assert ident.isbn is None and ident.author != "Elizabeth Castro"


def test_short_read_title_with_unexplained_words_is_not_matched():
    r = SpineRead("t1", "d", "f", printed_text="Photoshop CS6 | In Simple Steps", title="Photoshop CS6", legibility="partial")
    cands = [Edition("openlibrary", "u1", "Photoshop CS6", ["Mark Galer"])]
    assert choose(r, 0.8, cands, "t1", None).status == "unidentified"


def test_publisher_filter_cannot_leave_an_unverified_winner():
    r = SpineRead("t1", "d", "f", printed_text="In Color | HTML5 Canvas | FOR DUMMIES", title="HTML5 Canvas",
                  publisher="FOR DUMMIES", legibility="clear")
    cands = [Edition("openlibrary", "u1", "HTML5 Canvas", ["Steve Fulton"], "O'Reilly", "2011", "1"),
             Edition("openlibrary", "u2", "HTML5 Canvas", ["Oswald Campesato"])]
    assert choose(r, 0.95, cands, "t1", None).status == "unidentified"


def test_coauthors_in_different_order_are_one_authorship():
    from app.pipeline.identify import distinct_authorships
    cands = [Edition("ol", "u1", "Tika in Action", ["Chris A. Mattmann", "Jukka L. Zitting"]),
             Edition("ol", "u2", "Tika in Action", ["Jukka L. Zitting"])]
    assert distinct_authorships(cands) == 1
    ident = choose(_read("TIKA IN ACTION", ""), 0.95, cands, "t1", None)
    assert ident.status == "identified"


def test_short_titles_need_exact_words():
    from app.pipeline.identify import title_score
    assert title_score("HTML5", "HTML: Pocket Reference") < 85
    assert title_score("Java", "Java") == 100
    ident = choose(_read("HTML5", "", publisher=""), 0.95, [Edition("ol", "u", "HTML: Pocket Reference", ["Jennifer Robbins"])], "t1", None)
    assert ident.status == "unidentified"


def test_anchor_cut_by_frame_edge_is_rejected():
    from app.pipeline.scale import collect_anchors
    cut = Detection("door1", 0, "frames/f_000001.jpg", "anchor", (100, 146, 1600, 3840),
                    {"kind": "door", "fully_visible": True, "_frame_h": 3840})
    ok = Detection("door2", 0, "frames/f_000001.jpg", "anchor", (100, 200, 1000, 3000),
                   {"kind": "door", "fully_visible": True, "_frame_h": 3840})
    got = [a.ref for a, _ in collect_anchors([], {}, [cut, ok])]
    assert got == ["door2"]


def test_room_scale_checked_against_handheld_camera_height():
    from tests.test_room import synthetic_room
    from app.geometry.room import fit_room
    P, up = synthetic_room(tilt=False)
    cams = np.array([[2.0, 1.4, 1.5], [2.5, 1.5, 1.6]])
    good = fit_room(P, up, 100, "test", cams)
    assert good.confidence > 0.5 and any("1.4" in n or "1.5" in n for n in good.notes)
    bad = fit_room(P * 2, up, 100, "test", cams * 2)  # scale doubled: phone 2.9 m up
    assert bad.floor_area_m2 is None and bad.confidence == 0.0 and any("implausible" in n for n in bad.notes)


def test_item_range_needs_enough_comparables_and_brand_alone_is_not_a_model():
    from app.pipeline.price import price_item

    class Shop:
        enabled, label = True, "test"

        def __init__(self, ls):
            self.ls, self.queries = ls, []

        async def search(self, q, market):
            self.queries.append(q)
            return self.ls, "2026-10-17T10:00:00+00:00", "https://s"

    two = [Listing("laptop stand", "s", "u", 630, "INR", False, "IN"), Listing("laptop stand", "s", "u", 699, "INR", False, "IN")]
    shop = Shop(two)
    ip = asyncio.run(price_item(shop, FakeFX(), "i1", "laptop", "Apple", "IN", "INR"))
    assert shop.queries == ["Apple laptop"] and ip.low is None and "too few" in ip.note
    many = [Listing(f"bookcase {n}", "s", "u", p, "INR", False, "IN") for n, p in enumerate([2000, 3000, 4000, 5000, 9000])]
    ip = asyncio.run(price_item(Shop(many), FakeFX(), "i2", "wooden bookshelf", None, "IN", "INR"))
    assert ip.status == "range" and ip.low == 3000 and ip.high == 5000


def test_mixed_currency_listings_are_converted_one_by_one():
    from app.pipeline.price import price_item

    class Shop:
        enabled, label = True, "test"

        async def search(self, q, market):
            ls = [Listing("bed", "s", "u", p, "INR", False, "IN") for p in (10000, 12000, 14000, 16000)]
            ls.append(Listing("bed", "s", "u", 150.0, "USD", False, "IN"))  # 150 USD = 14,250 INR
            return ls, "2026-10-17T10:00:00+00:00", "https://s"

    ip = asyncio.run(price_item(Shop(), FakeFX(), "i1", "double bed", None, "IN", "INR"))
    assert ip.low == 12000 and ip.high == 14250 and "USD converted" in ip.note


def test_same_book_in_two_unlinked_segments_is_merged():
    from app.pipeline.track import merge_across_segments
    a = Track("a", [_det(0, 80, (0, 0, 10, 10))], segment=11)
    b = Track("b", [_det(0, 82, (0, 0, 10, 10)), _det(1, 84, (0, 0, 10, 10))], segment=12)
    c = Track("c", [_det(0, 200, (0, 0, 10, 10))], segment=13)  # same text but a minute later: another copy
    notes = merge_across_segments([a, b, c], {"a": "VISIONS OF THE FUTURE", "b": "Visions of the Future", "c": "VISIONS OF THE FUTURE"},
                                  {80: 50.0, 82: 52.0, 84: 54.0, 200: 120.0})
    assert a.fragment and not b.fragment and not c.fragment and len(b.dets) == 3 and notes


def test_ceiling_check_tracks_camera_pitch(tmp_path):
    from app.capture.recorder import SweepRecorder
    from app.config import load_settings
    from app.live.bridge import LiveBridge
    s = load_settings()
    b = LiveBridge(None, s.__class__(**{**s.__dict__, "gemini_api_key": "x"}), SweepRecorder(tmp_path))
    for _ in range(6):  # upright phone tilted slightly down (iOS sign: y negative when upright)
        b._track_pitch([0.0, -9.5, -2.5])
    assert b.max_pitch_deg < 0 and b.ceiling_missing()
    b._track_pitch([0.0, -6.0, 7.5])  # tilted ~50 degrees up
    assert b.max_pitch_deg > 40 and not b.ceiling_missing()
    b.rec.close()


def test_weak_duplicate_absorbed_and_lone_review_dropped():
    from app.pipeline.track import absorb_weak_tracks, drop_lone_reviews
    strong = Track("s", [_det(i, i, (100, 0, 140, 400)) for i in range(5)], segment=0, ref_box=(100, 0, 140, 400))
    weak = Track("w", [_det(0, 9, (105, 10, 138, 390))], segment=0, ref_box=(105, 10, 138, 390))
    other = Track("o", [_det(0, 3, (300, 0, 340, 400))] * 4, segment=0, ref_box=(300, 0, 340, 400))
    notes = absorb_weak_tracks([strong, weak, other])
    assert weak.fragment and len(strong.dets) == 6 and notes
    lone = [Track(f"l{i}", [_det(i, 20, (i * 50, 0, i * 50 + 40, 300))], segment=1, ref_box=(0, 0, 1, 1)) for i in range(2)]
    notes = drop_lone_reviews([strong, other, *lone], [[0, 1, 2, 3, 4], [20]], {**{i: float(i) for i in range(5)}, 20: 7.0}, {})
    assert all(t.fragment for t in lone) and notes


def test_old_edition_listing_is_not_the_replacement_price():
    ls = [Listing("(Old Edition) How To Prepare For Data Interpretation For CAT Sharma", "a", "u1", 270, "INR", False, "IN"),
          Listing("How to Prepare for Data Interpretation for CAT Sharma", "a", "u2", 349, "INR", False, "IN"),
          Listing("How to prepare for Data Interpretation for CAT Sharma", "a", "u3", 350, "INR", False, "IN")]
    bp = asyncio.run(price_book(FakeShopping({"IN": ls}), FakeFX(), "b", "How to Prepare for Data Interpretation for CAT", "Arun Sharma", None, "IN", "INR"))
    assert bp.replacement.amount == 349


def test_listing_match_rules_from_the_demo_price_audit():
    assert not listing_matches("Freemasonry at a Glance: Answers to 555 Questions", "555 Questions & Answers", None)
    assert listing_matches("555 Questions & Answers", "555 Questions & Answers", None)
    assert not listing_matches("Reading Scholars: Transportation : Satorre, Mark", "Transportation", "Lambert, Mark")
    assert listing_matches("Transportation : Lambert, Mark: Amazon.in: Books", "Transportation", "Lambert, Mark")
    assert not listing_matches("The Hunchback of Notre-Dame", "The Hunchback of Notre Dame", "Van Gool")
    assert listing_matches("The Incredible Book Eating Boy: Buy The Incredible Book Eating Boy by Oliver Jeffers", "The Incredible Book Eating Boy", "Oliver Jeffers")
    # second audit: volume numbers, series brackets, a different credited author, a series-name title
    assert not listing_matches("Phonics First: Bk. 1: Buy Phonics First: Bk. 1 by Calder Hunter", "Phonics First 6", "Hunter Calder")
    assert listing_matches("Quiz Whiz 3: 1,000 Super Fun Mind-Bending Trivia : National Geographic Kids", "Quiz whiz 3", "National Geographic Kids")
    assert not listing_matches("Buy Nature (Child's First Library of Learning S.) Book Online", "A Child's First Library of Learning", "C. E. Berry")
    assert listing_matches("Frog (See How They Grow) : Royston, Angela", "See How They Grow: Frog", "Angela Royston")
    assert listing_matches("Amazon.com: What Makes Me Me? (Big Questions) by Robert Winston", "What Makes Me Me?", "Robert Winston")
    assert not listing_matches("Life and Times in Ancient Greece: Charman, Andrew: Amazon.com", "Life and times in ancient Greece", "Julie Ferris")
    volumes = [Listing(t, "amazon.in", "", 100.0, "INR", False, "IN") for t in (
        "Science Starter: A Child's First Library of Learning : E. C. Berry",
        "Buy How Things Work: A Child's First Library of Learning Book Online")]
    assert several_books(volumes, "A Child's First Library of Learning")
    same = [Listing(t, "amazon.in", "", 100.0, "INR", False, "IN") for t in (
        "Kingfisher Knowledge: Microscopic Life : Walker", "Kingfisher Knowledge: Microscopic Life (Hardcover)")]
    assert not several_books(same, "Microscopic life")
    assert listing_matches("Buy TOP 10 OF EVERYTHING 2009 Book Online at Low Prices in India | Amazon.in", "Top 10 Of Everything 2009", "Russell Ash")
    # used-value audit: short titles must be the listing's own title; every title word must appear
    assert not listing_matches("The Wonderful World of Transportation book by David Lambert", "Transportation", "Lambert, Mark")
    assert not listing_matches("The Big Book of Search and Find by Tony Tallarico: Good Paperback", "The Big Book Of Search And Learn", "Tony Tallarico")
    assert not listing_matches("Big Book of Fantasy Quests Collection - Andy Dixon: 9780746070383", "Fantasy Quests", "Andy Dixon")
    assert listing_matches("Kingfisher Knowledge: Endangered Planet by Burnie, David: Very Good Hardback", "Endangered Planet", "David Burnie")
    assert listing_matches("Earthquakes book by Seymour Simon", "Earthquakes", "Seymour Simon")
    assert listing_matches("The Ship Book by Berenstain, Mike: Good Hardcover (1978)", "The ship book", "Michael Berenstain")
