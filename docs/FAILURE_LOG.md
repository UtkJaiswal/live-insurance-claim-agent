# Failure log

**Demo run:** `sweeps/sw_20261018_154322`. Library café, about 3 minutes of sweep, 380 books on two long bookcases of thin children's books, 27 other items.

## The three worst errors in the demo run

### 1. No book was priced (0 of 134 identified books)

- **Root cause.** Item searches took the price-search slots first.
  - TinyFish allows 30 searches per minute for the whole account.
  - The 27 items were grouped and queued for pricing early. The books were queued as each was identified, which was late (see error 2).
  - By the time the books' turn came, the 210 s pricing deadline had passed, so all 134 were marked "not priced: budget ran out". Meanwhile 21 of the 27 items were priced.
  - Books are the main claim, so this ordering was a design bug, not bad luck.
- **Fix.** Books are now priced before items; items get whatever budget is left (`app/pipeline/run.py`).
- **Not yet measured.** A re-run of the same sweep with the fix (`sweeps/sw_20261018_154322_rerun`) still priced nothing, for a different reason: the Gemini prepaid credit ran out mid-run (HTTP 402), so offers could not be read off the fetched pages.
  - That run exposed a second defect: the failure was reported as "no matching listing found". It now reads "listings found but not read: Gemini prepaid credit is used up" (`app/pipeline/tinyfish.py`).
  - TinyFish found the right pages (for example amazon.in for *The Children's Dinosaur Encyclopedia*). On earlier sweeps, prices read from fetched pages were all verified on the page: 0 dropped.

### 2. Packet 391 s after the sweep (bar: 300 s)

- **Root cause.** The work grows with the number of books, and this shelf had 380.
  - Spine reading took 101 s (77 model calls).
  - Catalogue identification took 229 s: 688 Open Library requests at about 3 per second.
  - Pricing is capped at 30 searches per minute, so about 60 books can be priced inside 5 minutes whatever the ordering.
- **Effect on the video.** It is 13 minutes long, over the 6-minute limit, and is submitted unedited.
- **What would fix it** (not done):
  - Identify and price books *during* the sweep, as their spines are read (detection already runs live).
  - Cache prices by ISBN across sweeps.
  - Raise the TinyFish rate tier.
  - For a demo, sweep 60–80 books. Earlier 6–44-book runs finished in 30–174 s.

### 3. Room size not reported

- **Root cause.**
  - The space is an open-plan hall with pillars, not a closed room.
  - The ceiling line was filmed only briefly.
  - One reconstruction from a hand-held walking sweep is not stable: the 6 reconstructions from staggered frame subsets gave floor areas from 6.5 to 32.6 m², a 260% spread.
- **What the system did.** It declined, by design. It reports a room size only when the reconstructions agree within 40%, so the packet says "room geometry unstable" and gives the range instead of a wrong number.
- **Measured on tape-measured ground truth** (bedroom, 3.37 × 2.80 m), see "Fixes measured in testing" below.

Other weaknesses visible in the demo:
- **246 of 380 books unidentified.** Thin children's picture books often carry no readable spine text. Those books are counted and measured but not guessed.
- **The count of 380 is unverified.** No ground truth was collected at the library.

## Cost and latency per stage (demo run)

| Stage | Seconds | Calls | USD |
|---|---|---|---|
| Live voice session (`gemini-3.8-live`) | the sweep | — | 0.204 |
| Live detection during the sweep (`gemini-3.8-flash`) | during the sweep | 48 | 0.231 |
| Detection of remaining frames | 24 | 7 | 0.056 |
| Tracking | 7 | — | — |
| Spine reading | 101 | 77 | 0.681 |
| Identification (Open Library) | 229 | 688 HTTP | 0 |
| Item grouping | 61 (parallel) | 1 | 0.029 |
| Pricing: TinyFish search + fetch, offers read with `gemini-3.5-flash-lite` | 229 (overlaps identification) | 34 searches, 60 reads | 0.074 |
| 3D reconstruction + room ensemble (DA3-BASE, Apple M4) | 70 (parallel) | — | 0 |
| **Total** | **391 to packet** | | **1.28** |

A 60–80 book sweep costs about US$0.25–0.60 and finishes in about 2 minutes, from the 24- and 44-book test runs.

## Fixes measured in testing

| Error | Root cause | Fix | Before → after |
|---|---|---|---|
| Wrong authors on generic titles (137-book bookstore shelf) | With no author printed, the first catalogue work with a matching title won; "HTML5" vs "HTML" is 89% similar | Reject ambiguous authorship; exact words for short titles; reject unexplained spine words | about 7 wrong of 70 → **0 wrong of 44** (manual review) |
| A neighbour's spine read as this book | A frame-edge crop showed the neighbour's text, and the longer title was preferred | Prefer the better detection; drop reads that repeat a neighbour's title | 1/14 wrong → **0/14** (hand-labelled shelf) |
| Books counted 2–3 times | Unlinked segments, overlapping boxes on one spine, distant re-views | Text merge across segments; overlapping-box removal; small-box filter; weak-track absorption | bedroom 1: 14 → 9 (truth 6); bedroom 2: 22 → 11 (truth 6) |
| Room scale off by ~40% | Door height measured in pixels; the chair hid the door's bottom | Height of the door's **top edge** above the floor | 7 samples agree within 0.5%; phone height 1.52 m (realistic) |
| Walls placed on furniture and a mirror reflection | Wall search inside the camera path; "farthest plane" rule | Walls must lie beyond every camera position; strongest plane 0.9–2.0 m high | bedroom floor −35% → **+3.9%** for one reconstruction (other frame subsets: unstable, so now cross-checked) |
| *Monk* ₹682.50, *Word Power* sent to appraisal | Median of a real offer and a marketplace import (₹1,140; ₹11,397) | Lower quartile of new offers; ignore old-edition listings | ₹682.50 → **₹225**; appraisal → **₹131**; *Data Interpretation* ₹270 (old edition) → **₹349** |
| False "hold steady" nudges | Laplacian variance reads plain walls as blurry | Texture-independent edge-sharpness ratio | 8 → **0** false nudges on recorded sweeps |
| Agent set US/USD before the user spoke | The live model completed the dialogue itself | Server refuses `set_locale` without recent user speech | Next sessions confirmed by voice |
| Sessions dropped at 40 s in the library | 4K frames saturated a ~3 Mbit/s uplink; WebSocket pongs were delayed past the timeout | 2560 px frames, voice prioritised over frames, longer ping timeout | 40 s → **full 3-minute sweep** recorded |

## With two more weeks

- Identify and price books during the sweep, so the time to packet does not grow with shelf size.
- Make the room turn a guided, separate capture step (rotating in place), and fuse gyro rotations with the reconstruction's camera poses for a stable room size.
- Calibrate the capture thresholds and de-duplication rules on labelled real shelf video, rather than on two bedroom sweeps.
- An independent checker pass that re-reads a random 10% of spines and reports disagreement.
