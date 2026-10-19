# Ground truth and results

Scores come from `scripts/evaluate.py`, one row per pass bar. The matching rules are documented at the top of `app/evaluate.py`. Ground-truth files are in `ground_truth/` and `testdata/ground_truth/`.

## Where ground truth exists

| Test | Ground truth | What it scores |
|---|---|---|
| Demo run, library café (`sw_20261018_154322`) | **None collected** | Time to packet only |
| Bedroom 1 (`sw_20261017_204356`) and bedroom 2 (`sw_20261018_083258`) | Tape-measured room 3.37 × 2.80 × 2.85 m; door 205 cm; the 6 books listed by hand; 10 items | Count, titles, items, room, time |
| Hand-labelled shelf (simulated 6-frame pan of a 9415 × 6158 photo) | Every book labelled by hand: 25 books, 15 legible | Count, titles |
| Bookstore shelf (simulated, 137 detections) | None. Identifications reviewed by hand | Wrong-identification rate |

The 20 hand-measured spines and the 15 hand-checked prices were **not collected**. The spine-dimension and price bars are therefore not scored against hand measurements; see the notes on each below.

## Pass bars

| Measure | Pass bar | Result | Status |
|---|---|---|---|
| Book count | within 5% | Hand-labelled shelf: **24 vs 25 (4%)**. Bedroom 1: 9 vs 6. Bedroom 2: 11 vs 6. Demo: 380, unverified | PASS on the shelf; FAIL on hand-held bedroom video (duplicates) |
| Title identification | ≥ 70% correct, ≤ 3% confidently wrong | Hand-labelled shelf: **14/15 correct (93%), 0 wrong**. Bedroom 1 and 2: 4/6 correct; the only "wrong" line is a duplicate copy of a correct title. Bookstore: 0 wrong of 44 on review | PASS on the shelf |
| Spine dimensions | within 15% (20 hand-measured) | Not scored: no 20-spine sample. *The 5 AM Club* measured 30.5 cm against 21.6 with the old door scale, which led to the door-top fix. Flat books are now left unmeasured | not scored |
| Book prices | within 25% (15 hand-checked) | Not scored against hand-checked prices. Bedroom 2 prices audited against the cached retailer pages: after the fixes all 5 come from the correct listing (₹131, ₹210, ₹225, ₹319, ₹349). Demo re-run: all 82 priced books' source listings checked by hand, 81 right and 1 doubtful, after the wrong matches found went into new matching rules | not scored |
| Non-book items | ≥ 80% found, correct category | Bedroom 1: **10/10**. Bedroom 2: **8/10** | PASS |
| Floor area | within 10% | Bedroom: not reported (reconstructions disagree). The best single reconstruction gave **9.81 vs 9.44 m² (+3.9%)** | FAIL (left empty) |
| Wall area | within 15% | Not reported: the ceiling was not filmed in those sweeps | FAIL (left empty) |
| Time to packet | under 5 minutes | Bedroom 1: **61 s**. Bedroom 2: **113 s**. Hand-labelled shelf: 30 s. Demo (380 books): **391 s** | PASS except the demo |

## Demo run (no ground truth)

| Output | Value |
|---|---|
| Books counted / identified / unidentified | 380 / 134 / 246 |
| Books priced | Live run: 0 (item searches took the budget first; failure log error 1). Offline re-run with the fix: **82 of 134** identified books, 20 to appraisal, 0 unverified prices; ₹65,280 new, ₹29,341 used. Every price hand-checked against its source listing: 81 right, 1 doubtful |
| Locale comparison | 10 books priced in India and in the US (`sweeps/sw_20261018_154322_rerun/locale_comparison.json`) |
| Items found / priced | 27 / 21 (₹90,562–159,042 replacement) |
| Room | Not reported: 6 reconstructions gave 6.5–32.6 m² |
| Live features exercised | Locale confirmed by voice; fast-pan, step-closer and ceiling nudges; the art question ("original or print?") with the answer recorded |
| Time to packet; cost | 391 s; US$1.28 |

## Bedroom sweeps (tape-measured)

Both are in `ground_truth/bedroom_demo*/`.

**Bedroom 1:** 6 books lying flat in a wardrobe, plus a notebook.
- 14 counted, 9 after the de-duplication fixes.
- 4/6 titles correct. The scorer counts one duplicate *Visions of the Future* line as wrong (1/5); no title is misread.
- 10/10 items found.
- Packet in 61 s.

**Bedroom 2:** the same books standing upright.
- 22 counted, 11 after the fixes.
- 4/6 titles correct.
- 8/10 items found.
- Packet in 113 s.

**Room (tape: 3.37 × 2.80 × 2.85 m):**
- Old method: 5.54 × 1.10 m (floor −35%).
- After the door-top scale and camera-aware wall fixes, one reconstruction gave **3.39 × 2.89 m** (floor +3.9%).
- Other frame subsets of the same sweep gave 4.8–31.4 m². The system therefore reports a room size only when its reconstructions agree, and left the room empty on both bedroom sweeps.

## Hand-labelled shelf (simulated)

- **Source:** "Part of a bookshelf containing books about ancient philosophy (1)" (Wikimedia Commons, CC BY-SA 4.0).
- **Shelf:** 25 books. 6 are shelved backwards, 2 are hidden behind neighbours, and there are two pairs of volume I/II with identical titles. The I/II pairs are correctly kept as separate books with different ISBNs.

| Measure | Result |
|---|---|
| Book count | 24 vs 25 (4%), PASS. The miss is a book hidden behind its neighbour |
| Titles | 14/15 correct, 0 wrong, PASS. *Die Natur und die Griechen* (spine shows only "Griechen") goes to review instead of being guessed |
| Time to packet | 30 s, $0.06 |
