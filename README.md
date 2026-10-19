# Library Contents Claim Agent

A live voice-and-vision agent for a home-contents claim. The policyholder walks their library once with a phone, talking to the agent. A few minutes later they get a claim packet: every book counted from its spine and identified where readable, priced at local market value with a source for each price, every other item priced, and the room's floor and wall area. Each number traces back to a saved frame and a price source, and anything the system is not sure of goes to a review queue instead of the totals.

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | pipeline diagram, which model does what, where metric scale and prices come from |
| [docs/FAILURE_LOG.md](docs/FAILURE_LOG.md) | the three worst errors in the demo run, root causes, fixes; cost and latency per stage |
| [docs/RESULTS.md](docs/RESULTS.md) | ground truth and the system's numbers against each pass bar |

## About the demo video

**Video:** [insurance-claim-video-demo.MP4 on Google Drive](https://drive.google.com/file/d/1Ct27fSrBMwOgKdoC9zlcPSaa2OZ_g6j-/view?usp=sharing) (13 min, 1.5 GB).

The video is one unedited take of **13 minutes**, over the 6-minute limit. It was not cut or sped up, as the brief requires. Where the time went (session clock from `events.jsonl` and `metrics.json`; the screen recording starts about 12 s into the session, so video times are about 12 s earlier):

| Session time | What happens |
|---|---|
| 0:00–0:25 | Greeting; country and currency confirmed by voice |
| 0:25–3:18 | **The sweep (2 min 52 s)**, deliberately covering the edge cases: "slow down" on a fast pan (0:30), "step closer" when spines were too far to read (0:43), the agent asking whether a framed picture is an original or a print (2:05) and recording the answer "it's a print" (3:06), and the ceiling check stopping the sweep from ending until the ceiling line was filmed (2:42) |
| 3:18–9:48 | **Processing: 6.5 minutes.** The shelves held about 380 thin children's books, and identifying that many books is the slow step (failure log, error 2) |
| 9:48–end | Packet ready; the agent reads the summary; walkthrough of the report |

**Why this take and not a shorter one.** The library café's network allowed only about 3 Mbit/s of upload. The first attempts were cut off at exactly 40 seconds: full-size frames saturated the connection and the server dropped the session. This was fixed on site (smaller frames, voice prioritised over frames, longer connection timeouts), and this take was the first one to run end to end. A second visit to re-record was not possible.

- **No book was priced in this run.** [docs/FAILURE_LOG.md](docs/FAILURE_LOG.md) (error 1) gives the cause and the fix. Re-running the same recorded sweep with the fix priced 82 of the 134 identified books and sent 20 to appraisal; that packet and report are in `sweeps/sw_20261018_154322_rerun/`.
- **No ground truth was collected at the library.** Measured accuracy comes from a hand-labelled shelf and a tape-measured bedroom ([docs/RESULTS.md](docs/RESULTS.md)).

The demo's packet, report and the frames it references are in `sweeps/sw_20261018_154322/`.

## Quick start (about 10 minutes)

### 1. Install the tools (once)

| Tool | macOS | Linux | Windows |
|---|---|---|---|
| [uv](https://docs.astral.sh/uv/) (Python package manager; installs Python 3.12 itself if needed) | `brew install uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) (HTTPS tunnel; phone browsers only allow the camera over HTTPS) | `brew install cloudflared` | the `.deb`/`.rpm` from the link | `winget install Cloudflare.cloudflared` |

You also need a phone with a browser. It was tested on an iPhone (Safari); Android Chrome should work the same way but was not tested.

### 2. Get the code and install

```bash
git clone <this repo> live-room && cd live-room
cp .env.example .env             # then add GEMINI_API_KEY and TINYFISH_API_KEY (see Keys below)
uv sync --extra geometry         # downloads ~0.8 GB (PyTorch); plain `uv sync` skips room geometry
uv run pytest                    # 55 offline tests, about 20 s: confirms the install
uv run python scripts/live_smoke.py   # optional: checks the Gemini key; prints the agent's greeting
```

### 3. Run

```bash
uv run uvicorn app.server:app --port 8000 --ws-ping-interval 30 --ws-ping-timeout 120
cloudflared tunnel --url http://localhost:8000   # in a second terminal; the URL is in the "Your quick Tunnel has been created!" box
```

The first sweep downloads the Depth Anything 3 weights from Hugging Face (about 0.5 GB, once). Room geometry runs on Apple Silicon (MPS) or an NVIDIA GPU. On a CPU-only machine, set `DA3_MODEL=depth-anything/DA3-SMALL` and `ROOM_ENSEMBLE=1` in `.env` to keep it fast.

Open the `https://….trycloudflare.com` URL on the phone, tap **Start**, and allow camera, microphone and motion access. The agent then:
1. confirms the country and currency
2. explains the sweep
3. directs it while you film

When the sweep ends, the packet link appears on the phone and the agent reads the summary aloud.

Optional URL parameters:
- `?kmax=3840` sends full 4K frames on a fast network (default 2560 px, which suits a ~3 Mbit/s uplink).
- `?kf=500` sends a frame every 500 ms (default 1000).

### Keys

| Variable | Needed for | Where |
|---|---|---|
| `GEMINI_API_KEY` | the live agent and all vision stages | aistudio.google.com/apikey. The free tier allows about 20 requests per day per model, and a sweep needs 100 or more, so enable billing. US$0.25–0.60 for a 60–80 book sweep; $1.28 for the 380-book demo. |
| `TINYFISH_API_KEY` | prices: retailer search and page fetch | tinyfish.ai. Search and Fetch are free on every plan, limited to 30 searches per minute. |
| `SERPER_API_KEY` | prices, used only when no TinyFish key is set | serper.dev. With no price key, prices stay empty and the reason goes to the review queue. |
| `GOOGLE_BOOKS_API_KEY` | optional: more catalogue dimensions, for better scale | Google Cloud console: enable the Books API. Open Library is used without a key. |

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| The session stops after about 40 s | Slow upload: frames back up and the connection times out | Start uvicorn with the `--ws-ping-*` flags above. Open the page with `?kmax=1920` for smaller frames. Watch "upload backlog" under the buttons |
| "SLOW NETWORK" under the buttons | Frames are being skipped to keep the voice real-time | Move closer to the Wi-Fi router, use a phone hotspot, or add `?kmax=1920` |
| The phone still behaves like an old version | Cached page | Close the browser tab and reopen the link (the server sends `no-store`) |
| Review queue says "listings found but not read: Gemini prepaid credit is used up" | Gemini billing ran out (HTTP 402) | Top up in AI Studio; re-run with `scripts/process_sweep.py sweeps/<id> --reuse` |
| The tunnel link stopped working | `cloudflared` was restarted (quick-tunnel links change) | Use the new link from the `cloudflared` window |

## Running without a phone

```bash
uv run python scripts/process_sweep.py sweeps/<id>            # re-run the whole post-sweep pipeline on a recorded sweep
uv run python scripts/process_sweep.py sweeps/<id> --reuse    # reuse saved detections and spine reads
uv run python scripts/replay_live.py sweeps/<id> --url ws://127.0.0.1:8000/ws/live   # replay a recorded session as the phone would
uv run python scripts/make_test_sweep.py testdata/source/philosophy_shelf.jpg --frames 6   # simulated sweep from a shelf photo
uv run python scripts/evaluate.py --packet sweeps/<id>/claim_packet.json --truth ground_truth/<room>   # one row per pass bar
uv run pytest     # 55 offline tests; no network or model calls
```

The test photos are on Wikimedia Commons and are not committed:

```bash
mkdir -p testdata/source
curl -L -o testdata/source/philosophy_shelf.jpg "https://upload.wikimedia.org/wikipedia/commons/f/f5/Part_of_a_bookshelf_containing_books_about_ancient_philosophy_%281%29.jpg"   # CC BY-SA 4.0
curl -L -o testdata/source/webdesign_books.jpg "https://upload.wikimedia.org/wikipedia/commons/3/32/WebDesignBooks.JPG"   # CC0
```

## What a sweep produces

Each session is recorded to `sweeps/<id>/`. Everything after the sweep reads only these files.

| Path | Contents |
|---|---|
| `claim_packet.json` | The output contract. Extra fields: catalogue URL, measurement method, appraisal reason, lines the policyholder excluded. |
| `report.html` | The human-readable report, rendered from the JSON, with an evidence crop per line. |
| `frames/f_000123.jpg` | Saved keyframes. Every `frame_ref` is `frames/f_….jpg#xywh=x,y,w,h`: the frame plus the box. |
| `crops/` | The evidence crop for each line. |
| `stages/*.json` | Each stage's own output: detect, track, read, identify, measure, items, price, room. |
| `locale_comparison.json` | The same 10 books priced in a second country (US, or India for a US sweep). |
| `metrics.json` | Seconds, model calls, tokens, HTTP calls and USD per stage, the live-session cost, and time to packet. |
| `events.jsonl`, `audio_*.wav`, `meta.json` | Transcripts, tool calls, nudges, per-frame quality and motion data, audio, and timings. |

Catalogue, price and exchange-rate lookups are cached in `cache/http/` with their original retrieval time. A re-run shows the date the source was actually read, and cached searches skip the search rate limit: re-pricing the 380-book demo sweep from the cache takes about a minute.

## Rules the code enforces

- **No model-made prices.** Prices come only from retailer pages fetched during the run, plus ECB exchange rates. A model reads the offers off the fetched page, and code keeps a price only if its exact text appears on that page. The integrity check (`app/packet/integrity.py`) flags any price without a source, URL and retrieval date.
- **No guessed titles.** The spine reader may only copy text it transcribed, and code drops any title or author that is not in the transcription. A title matching books by several authors, with no author printed on the spine, is left unidentified.
- **Totals are computed in code** (`app/packet/totals.py`). The integrity check recomputes them from the lines.
- **Art, signed or old books, and prices above ₹5,000 are not auto-priced.** They go to `needs_appraisal`, with the reason.
- **"I don't know" is an answer.** Unreadable spines, missing listings, a missing scale anchor, an uncaptured ceiling, unstable room geometry and frames that failed analysis all go to the review queue with the reason. They never become a guessed number.

## The reference app: kept, changed, thrown away

Reference: [Insurance Claim Live Agent Team](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/voice_ai_agents/insurance_claim_live_agent_team) (`awesome-llm-apps/voice_ai_agents/insurance_claim_live_agent_team`). I ran it locally before writing any code, and read its source (`live_demo/server.py`, `live_demo/live_tools.py`, `agent.py`) to see how the live session, the tools and the background graph fit together. **This project is written from scratch, not a fork.**

| | Reference app | This project |
|---|---|---|
| **Kept** | FastAPI server with one WebSocket per call; the browser streams 16 kHz PCM audio and JPEG frames into a Gemini Live session | Same transport (`app/live/bridge.py`), same model family (`gemini-3.8-live`) |
| **Kept** | Live input and output transcription, shown as a running transcript | Same |
| **Kept** | Tools the live model calls to record what it learns while the conversation continues | `set_locale`, `begin_sweep`, `mark_region`, `log_item`, `record_user_statement`, `end_sweep` |
| **Kept** | A claim file that fills in on screen during the call, then a packet for human review | A live inventory during the sweep, then `claim_packet.json` + `report.html` with a review queue |
| **Changed** | The agent describes damage in a frame | The agent **directs the capture**: a per-frame monitor (blur, glare, darkness, pan speed, books too far away, ceiling never filmed) turns problems into spoken instructions while the user is still at the shelf |
| **Changed** | One camera stream, at the size the live model receives | A low-resolution preview for the live model, plus separate high-resolution keyframes with gyro and gravity data, saved to disk for measurement and replay |
| **Changed** | Background work as an ADK graph of LLM agents (normalizer, classifier, validation, coverage, fraud checks) | Background work as **separate, testable, deterministic stages** (detect, track, read, identify, measure, price, room, assemble), each writing its own output. Models only detect and transcribe; code does the matching, scale, totals and integrity checks |
| **Changed** | No numbers that must be right | Counts, dimensions, prices and areas checked against ground truth (`scripts/evaluate.py`) |
| **Changed** | Non-blocking tools; sessions limited by Live API defaults | Blocking tools that return in milliseconds. Sliding-window context compression (audio+video sessions otherwise stop at 2 minutes) and session resumption on GoAway |
| **Changed** | Runs on localhost only | Reached from a phone through an HTTPS tunnel, with upload backpressure (frames are skipped before the voice lags) |
| **Thrown away** | Optional live avatar (video output) | Voice only |
| **Thrown away** | Incident sketches (image generation) | Not needed: every claim line is evidenced by a saved photo |
| **Thrown away** | Mock policy directory, policy lookup, coverage, fraud and document-checklist rules | Out of scope here ("a policy database is not scored") |
| **Thrown away** | ZIP packet of Markdown notes | Replaced by the JSON contract plus an HTML report rendered from it |

## Tools used

- **Gemini Live** (`gemini-3.8-live`): the voice agent.
- **Gemini Flash** (`gemini-3.8-flash`): detection, spine reading and item grouping.
- **`gemini-3.5-flash-lite`**: reads offers off fetched retailer pages.
- **Depth Anything 3** (DA3-BASE, Apache-2.0): multi-view room reconstruction.
- **OpenCV SIFT and RANSAC**: linking frames.
- **Open Library and Google Books**: identification.
- **TinyFish Search and Fetch**: retailer prices.
- **Frankfurter**: ECB exchange rates.
- **FastAPI**: the server.
- **Claude Code** was used as a coding assistant.
