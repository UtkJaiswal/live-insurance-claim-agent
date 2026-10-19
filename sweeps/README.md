# Recorded sweeps kept in the repo

Each sweep folder is what one session wrote: frames, events, audio, stage outputs, the claim packet and the report. Only the sweeps the documents cite are committed.

| Sweep | What it is | Kept |
|---|---|---|
| `sw_20261018_154322` | **The demo video's sweep** (library café, 380 books). Its `claim_packet.json`, `report.html` and every frame its `frame_ref`s point to | everything |
| `sw_20261017_204356` | Bedroom 1: 6 books lying flat, tape-measured room (`ground_truth/bedroom_demo`) | everything |
| `sw_20261018_083258` | Bedroom 2: the same books upright (`ground_truth/bedroom_demo2`); the price audit in `docs/RESULTS.md` | everything |
| `sw_20261017_155806` | First test sweep: the false blur nudges and the agent choosing US/USD before the user spoke (`docs/FAILURE_LOG.md`) | events and meta only |
| `sw_20261018_153515`, `sw_20261018_154011` | Library attempts cut off at exactly 40 s by the slow uplink (README, "About the demo video") | events and meta only |
| `sw_20261018_154322_rerun` | Offline re-run of the demo sweep with books priced first; stopped by the depleted Gemini credit (failure log, error 1) | metrics, meta and `stages/price.json` only |

The simulated test sweeps are in `testdata/sweeps/`, with their hand-labelled ground truth in `testdata/ground_truth/`.
