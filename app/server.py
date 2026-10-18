"""HTTP + WebSocket entry point.

  GET  /                  phone capture app (web/)
  WS   /ws/live           one live sweep session
  GET  /api/sweeps        recorded sweeps
  GET  /sweeps/<id>/...   saved frames, audio, claim_packet.json and report.html (every frame_ref opens)
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from app.capture.recorder import SweepRecorder
from app.config import ROOT, load_settings
from app.live.bridge import LiveBridge
from app.live.prompts import summary_message
from app.live.scanner import LiveScanner
from app.pipeline.run import process_sweep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

settings = load_settings()
settings.sweeps_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Library Contents Claim Agent")


@app.middleware("http")
async def no_stale_client(request, call_next):
    """Phones must never run a cached copy of the capture app: in testing Safari kept an old app.js
    that still sent 4K frames after a fix. Recorded sweep files can be cached normally."""
    response = await call_next(request)
    if not request.url.path.startswith("/sweeps/"):
        response.headers["Cache-Control"] = "no-store"
    return response


async def on_sweep_end(bridge: LiveBridge) -> None:
    """Runs the post-sweep pipeline; the agent reads the summary as soon as the packet exists."""
    rec = bridge.rec
    rec.log("pipeline_started")
    scanner: LiveScanner | None = getattr(bridge, "scanner", None)

    async def progress(msg: str) -> None:
        await bridge._send_json({"type": "status", "text": msg})

    async def on_packet(packet, info) -> None:
        rec.meta["packet_ready_t"] = rec.t()
        rec.write_meta()
        ttp = rec.meta["packet_ready_t"] - (rec.meta.get("sweep_ended_t") or rec.meta["packet_ready_t"])
        rec.log("pipeline_done", time_to_packet_s=ttp, defects=info["defects"])
        await bridge._send_json({"type": "packet", "report": f"/sweeps/{rec.id}/report.html",
                                 "json": f"/sweeps/{rec.id}/claim_packet.json", "summary": info["summary"],
                                 "time_to_packet_s": round(ttp, 1)})
        await bridge.say(summary_message(info["summary"]))

    try:
        await process_sweep(rec.dir, settings, live_detections=scanner.detections if scanner else None,
                            progress=progress, on_packet=on_packet)
    except Exception as e:
        log.exception("pipeline failed for %s", rec.id)
        rec.log("pipeline_failed", error=str(e))
        await bridge._send_json({"type": "status", "text": f"processing failed: {e}"})
        await bridge.say(summary_message("Processing failed. Apologise and say the team will follow up."))
        return
    # Fold the live scanner's cost and the live clock into the final metrics.
    metrics_path = rec.dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if rec.meta.get("packet_ready_t") is not None and rec.meta.get("sweep_ended_t") is not None:
        metrics["time_to_packet_s"] = round(rec.meta["packet_ready_t"] - rec.meta["sweep_ended_t"], 1)
    if scanner:
        metrics["stages"]["live_detect"] = scanner.ledger.to_dict()["stages"].get("detect", {})
        metrics["total_usd"] = round(metrics["total_usd"] + scanner.ledger.to_dict()["total_usd"], 4)
    metrics_path.write_text(json.dumps(metrics, indent=1))


@app.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    if not settings.gemini_api_key:
        await ws.send_text(json.dumps({"type": "status", "text": "GEMINI_API_KEY is not set on the server"}))
        await ws.close()
        return
    rec = SweepRecorder(settings.sweeps_dir)
    log.info("session %s opened", rec.id)
    await ws.send_text(json.dumps({"type": "session", "id": rec.id}))
    bridge = LiveBridge(ws, settings, rec, on_sweep_end=on_sweep_end)
    bridge.scanner = LiveScanner(bridge)  # type: ignore[attr-defined]
    bridge.keyframe_listeners.append(bridge.scanner.on_keyframe)  # type: ignore[attr-defined]
    try:
        await bridge.run()
    finally:
        rec.close()
        log.info("session %s closed (%d frames)", rec.id, rec.frame_count)


@app.get("/api/sweeps")
def list_sweeps() -> list[dict]:
    out = []
    for d in sorted(settings.sweeps_dir.iterdir(), reverse=True):
        meta = d / "meta.json"
        if meta.is_file():
            out.append(json.loads(meta.read_text()))
    return out


app.mount("/sweeps", StaticFiles(directory=settings.sweeps_dir), name="sweeps")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
