# Architecture

```
PHONE (browser)                           SERVER (FastAPI)                                OUTSIDE SERVICES
mic, 16 kHz PCM ────────────┐
1 fps preview ──────────────┼──► LiveBridge ◄───────────────────────────────────────► Gemini Live (voice agent)
agent voice, 24 kHz ◄───────┘      │  tools: set_locale (refused until the user has answered), begin/end_sweep,
                                   │         mark_region, log_item, record_user_statement
keyframes 2560 px ───────────────► │  capture monitor: blur, glare, dark, pan speed, ceiling tilt
  + gyro / gravity                 │      └─► [monitor] nudges, spoken by the agent
                                   │  LiveScanner: detect a frame every ~2 s ────────► Gemini Flash
                                   ▼      (live inventory; "step closer"; "original or print?")
                        sweeps/<id>/  frames, events, audio, meta
                                   │ end of sweep
                                   ▼
 detect remaining frames ─► clean boxes ─► track (SIFT + homography) ─► read spines (crops) ─► de-duplicate
                                                                                                      │
     ┌─────────────────────────────── in parallel ──────────────────────────────────────────────────┤
     ▼                                   ▼                                    ▼                         │
 identify ──► Open Library      reconstruct (DA3-BASE) + 5-way           group items (vision)          │
     │        / Google Books    room ensemble                                 │                         │
     ▼                                   ▼                                    │                         │
 price each book ──► TinyFish  scale (A4 > catalogue > door top) ─► measure spines, items, room       │
 as identified       search + fetch, verify on the page, ECB FX                                       │
     └───────────────────────► items priced after the books ◄─────────────────┘                      │
                                   ▼                                                                    │
            assemble + review queue ─► totals in code ─► integrity check ─► claim_packet.json, report.html
                                                                           └─► [summary] spoken by the agent
```

## Which model does what

| Stage | Model or method | Why |
|---|---|---|
| Live conversation | `gemini-3.8-live`. Sliding-window compression and session resumption. | Real-time voice with camera. Audio+video sessions otherwise stop at 2 minutes. |
| Capture checks | OpenCV edge-sharpness ratio, saturated pixels, brightness; gyro rate; camera pitch from gravity | Cheap enough to run on every frame. Plain walls get no blur verdict. The sweep cannot end (once) until the ceiling line has been filmed. |
| Detection | `gemini-3.8-flash`, one call per frame | Finds books (upright, flat, backwards, partly hidden), items, and doors and A4 sheets used as scale anchors. Notebooks and pamphlets are excluded. |
| Counting | SIFT + RANSAC homographies between overlapping frames; union-find over linked boxes | A shelf face is close to a plane, so one book seen in five frames becomes one track. Clean-up: overlapping boxes removed; books under 10% of frame height dropped; weak tracks absorbed; matching spine text merged across segments; a lone distant view dropped. |
| Spine reading | `gemini-3.8-flash` on high-resolution crops, 8 per call | Transcription only. Code drops any title or author not in the transcription, and text picked up from a neighbouring spine. |
| Identification | Open Library (plus Google Books with a key) | Edition and ISBN are chosen only when the spine's publisher or volume supports them. A generic title without a printed author is not resolved. |
| 3D | Depth Anything 3, **DA3-BASE** (0.12B, Apache-2.0), up to 48 frames on Apple MPS / CUDA / CPU | Unscaled depth and camera poses. Never trusted for absolute size. |
| Item de-duplication | `gemini-3.8-flash` groups thumbnails of all sightings; code validates the grouping | Furniture seen from several angles does not fit planar linking. |
| Price reading | `gemini-3.5-flash-lite` on the price regions of fetched pages | Reads offers. It never supplies a price: code keeps only price strings that occur on the page. |

## Where metric scale comes from

Best first; the packet's `scale_method` names what was used.

1. **A4 sheet** (29.7 × 21.0 cm), standing in view: the explicit reference object.
2. **Catalogue dimensions** of identified books whose matched edition lists a height. Leave-one-out: a book's own catalogue height never measures that book.
3. **Door top edge**: a door leaf's top stands 205 cm above the floor. Only the top edge and the floor plane are used, because furniture often hides a door's bottom. That made the earlier full-height method inflate sizes by about 40%.
4. **Standard book format** (median upright book 21 cm): last resort, confidence 0.3, always sent to review.

**Applying the scale.**
- Each anchor gives `k = known_cm / (h·z/f)`, from its pixel size `h`, depth `z` and focal length `f`.
- The median `k`, with outliers rejected, scales every spine, every item and the room.
- **Sanity check:** the scaled camera must sit 0.9–2.0 m above the floor (a hand-held phone). If it does not, the room is left empty.

**Room.**
- *Up* comes from the accelerometer, sign self-calibrated because iOS and Android differ.
- The floor is a height-histogram peak.
- Walls are the strongest plane 0.9–2.0 m high **beyond every camera position**. The phone was inside the room, so this rejects furniture faces and mirror reflections.
- The room is reconstructed from 5 staggered frame subsets. The median is reported only if their floor areas agree within 40%; otherwise the room is left empty and the range goes to the review queue.
- Floor area = L × W. Wall area = perimeter × height, gross. Shelved wall area = the front faces of measured shelving units.

## Where prices come from

**TinyFish Search and Fetch** (free; 30 searches per minute), with Serper as a fallback.

1. **Search** the market's retailers for `title + author surname`. India: amazon.in, flipkart.com, bookswagon.com, sapnaonline.com, crossword.in, plus the used-book shops bookchor.com and kitabay.com.
2. **Fetch** the top 3 product pages. Each page is cached with its retrieval time.
3. **Read offers** from the price regions with `gemini-3.5-flash-lite`: price text verbatim, condition, format, seller.
4. **Verify in code** that each price string occurs on the fetched page. In every test run, 0 offers failed this check.

**Rules.**
- **Replacement cost** = the lower quartile of matching new print offers: what it costs to replace the book today. A median let a ₹11,397 import listing beat a ₹131 paperback.
- **Used value** = the median of used offers.
- **Matching a listing to the book:** every meaningful title word must be in the listing, and it must carry the author's surname unless the title is long and the listing credits no other author. A title of one or two words must be the listing's own title. Numbers in the title must agree ("Phonics First 6" is not "Bk. 1"). A title found only in a listing's series brackets is a different volume. If the matches are different books, the title is a series name and is not priced.
- **Used value** is capped at the replacement cost.
- **Excluded offers:** old-edition listings when current ones exist, e-books, audiobooks, and "New & Used from ₹X" summary lines.
- **Currency:** each listing is converted on its own at the ECB rate of the day.
- **Order:** books are priced as each is identified, before items. Books not priced by `PRICE_DEADLINE_S` (210 s) are marked unpriced, with the reason.

**Items.**
- Searched by product type ("buy wardrobe online India"). Category pages are allowed.
- Range = interquartile range of at least 4 comparable listings.
- A legible model number gives a single price.
- Art goes to `needs_appraisal` unless the policyholder says it is a print.

**Appraisal instead of a price:** the policyholder says signed, first edition or rare; the edition is from before 1950; the binding looks antiquarian; or any price found is above ₹5,000 (US$60, £50).
