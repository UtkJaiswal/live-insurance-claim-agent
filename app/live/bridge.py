"""Bridges one phone WebSocket to one Gemini Live session.

Phone -> server
  binary, first byte = message type:
    0x01 audio     raw PCM16 LE, 16 kHz mono
    0x02 preview   JPEG, <= 1 fps, forwarded to the live model
    0x03 keyframe  [4-byte big-endian header length][header JSON][JPEG], full resolution, recorded
  text: JSON control messages {"type": "hello" | "end_sweep" | "stop", ...}

Server -> phone
  binary: agent audio, raw PCM16 LE, 24 kHz mono
  text: JSON events (transcript, item, region, nudge, quality, status, interrupted, ...)

Live sessions with video are capped at 2 minutes unless context-window compression is on, and a
connection lives about 10 minutes, so compression is always on and the bridge resumes the session
with the last resumption handle whenever the server sends GoAway or drops the connection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import logging
import struct
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import errors, types

from app.capture.quality import ISSUE_HINTS, CaptureMonitor, Thresholds, assess_jpeg
from app.capture.recorder import SweepRecorder
from app.config import Settings
from app.live.prompts import SYSTEM_PROMPT, monitor_message
from app.live.tools import TOOL_DECLARATIONS, LiveState, run_tool

log = logging.getLogger("live")

MSG_AUDIO, MSG_PREVIEW, MSG_KEYFRAME = 0x01, 0x02, 0x03
MAX_RESUMES = 5

KeyframeListener = Callable[[str, dict[str, Any]], Awaitable[None]]
SweepEndHandler = Callable[["LiveBridge"], Awaitable[None]]


class LiveBridge:
    def __init__(
        self,
        ws: WebSocket,
        settings: Settings,
        recorder: SweepRecorder,
        on_sweep_end: SweepEndHandler | None = None,
    ):
        self.ws = ws
        self.settings = settings
        self.rec = recorder
        self.state = LiveState(country=settings.default_country, currency=settings.default_currency)
        self.monitor = CaptureMonitor(
            Thresholds(
                blur=settings.blur_ratio_threshold,
                min_texture=settings.min_texture,
                glare=settings.glare_threshold,
                dark=settings.dark_threshold,
            ),
            cooldown_s=settings.nudge_cooldown_s,
        )
        self.keyframe_listeners: list[KeyframeListener] = []
        self.on_sweep_end = on_sweep_end
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.session: Any = None
        self.last_frame_ref: str | None = None
        self._resume_handle: str | None = None
        self._last_frame_digest: bytes | None = None
        self._sweep_end_fired = False
        self._gravity_y_sum = 0.0
        self._gravity_n = 0
        self.max_pitch_deg: float | None = None
        self._ceiling_reminded = False
        self._client_gone = asyncio.Event()
        self._send_lock = asyncio.Lock()

    # ---------- session config ----------

    def _config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in TOOL_DECLARATIONS])],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.settings.live_voice)
                )
            ),
            # Indian English was transcribed as Spanish/German in testing, and the agent answered in German.
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["en-IN", "en-US"]),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
            session_resumption=types.SessionResumptionConfig(handle=self._resume_handle),
        )

    # ---------- lifecycle ----------

    async def run(self) -> None:
        reader = asyncio.create_task(self._read_client())
        resumes = 0
        try:
            while not self._client_gone.is_set():
                try:
                    async with self.client.aio.live.connect(model=self.settings.live_model, config=self._config()) as session:
                        self.session = session
                        self.rec.log("live_connected", resumed=self._resume_handle is not None)
                        await self._status("connected")
                        if resumes == 0:
                            await session.send_realtime_input(text=(
                                "The user has opened the app. Greet them. Their account lists country "
                                f"{self.settings.default_country} and currency {self.settings.default_currency}; "
                                "ask them to confirm or correct it."
                            ))
                        await self._receive_until_goaway(session)
                except errors.APIError as e:
                    self.rec.log("live_error", code=getattr(e, "code", None), message=str(e))
                    log.warning("live connection ended: %s", e)
                finally:
                    self.session = None
                if self._client_gone.is_set():
                    break
                resumes += 1
                if not self._resume_handle or resumes > MAX_RESUMES:
                    await self._status("agent disconnected")
                    break
                await self._status("reconnecting")
        finally:
            reader.cancel()
            self._sync_meta()

    async def _receive_until_goaway(self, session: Any) -> None:
        while not self._client_gone.is_set():
            async for msg in session.receive():
                if msg.session_resumption_update and msg.session_resumption_update.resumable:
                    self._resume_handle = msg.session_resumption_update.new_handle or self._resume_handle
                if msg.go_away is not None:
                    self.rec.log("live_goaway", time_left=str(msg.go_away.time_left))
                    return
                if msg.usage_metadata is not None:
                    self.rec.log("usage", **msg.usage_metadata.model_dump(exclude_none=True, mode="json"))
                if msg.tool_call:
                    await self._handle_tool_call(session, msg.tool_call)
                sc = msg.server_content
                if sc is None:
                    continue
                if sc.interrupted:
                    await self._send_json({"type": "interrupted"})
                if sc.model_turn:
                    for part in sc.model_turn.parts or []:
                        if part.inline_data and part.inline_data.data:
                            self.rec.audio_out(part.inline_data.data)
                            await self._send_bytes(part.inline_data.data)
                if sc.input_transcription and sc.input_transcription.text:
                    await self._transcript("user", sc.input_transcription.text)
                if sc.output_transcription and sc.output_transcription.text:
                    await self._transcript("agent", sc.output_transcription.text)
                if sc.turn_complete:
                    await self._send_json({"type": "turn_complete"})

    # ---------- phone -> server ----------

    async def _read_client(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await self._on_binary(msg["bytes"])
                elif msg.get("text") is not None:
                    await self._on_control(json.loads(msg["text"]))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self._client_gone.set()
            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    pass

    async def _on_binary(self, data: bytes) -> None:
        kind, payload = data[0], data[1:]
        if kind == MSG_AUDIO:
            self.rec.audio_in(payload)
            if self.session is not None:
                await self._safe(self.session.send_realtime_input(audio=types.Blob(data=payload, mime_type="audio/pcm;rate=16000")))
        elif kind == MSG_PREVIEW:
            if self.session is not None:
                await self._safe(self.session.send_realtime_input(video=types.Blob(data=payload, mime_type="image/jpeg")))
        elif kind == MSG_KEYFRAME:
            (hlen,) = struct.unpack(">I", payload[:4])
            header = json.loads(payload[4 : 4 + hlen])
            jpeg = payload[4 + hlen :]
            await self._on_keyframe(jpeg, header)

    async def _on_keyframe(self, jpeg: bytes, header: dict[str, Any]) -> None:
        # Frames after the sweep are not evidence; in the first test a locked phone kept sending
        # identical dark frames for five minutes after end_sweep.
        if self.state.sweep_ended_t is not None:
            return
        digest = hashlib.blake2b(jpeg, digest_size=16).digest()
        if digest == self._last_frame_digest:
            self.rec.log("duplicate_frame", header=header)
            return
        self._last_frame_digest = digest
        try:
            q = await asyncio.to_thread(assess_jpeg, jpeg, header.get("rotation_dps"))
        except ValueError:
            self.rec.log("bad_frame", header=header)
            return
        header = {**header, "region": self.state.region, "sweeping": self.state.sweeping}
        if self.state.sweeping:
            self._track_pitch(header.get("gravity"))
        ref = self.rec.save_keyframe(jpeg, header, q.as_dict())
        self.last_frame_ref = ref
        await self._send_json({"type": "quality", "ref": ref, **q.as_dict(), "issues": self.monitor.issues(q)})
        if self.state.sweeping:
            issue = self.monitor.observe(q, self.rec.t())
            if issue:
                text = ISSUE_HINTS[issue]
                n = self.monitor.repeats.get(issue, 0)
                if n:
                    text += f" You have said this {n} time(s) already: use different, shorter words."
                await self.nudge(text, issue)
        for listener in self.keyframe_listeners:
            asyncio.create_task(listener(ref, {**header, "_quality": q.as_dict()}))

    async def _on_control(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self.rec.meta["device"] = msg.get("device", "")
            self.rec.meta["camera"] = msg.get("camera", {})
            self.rec.write_meta()
            self.rec.log("hello", **msg)
        elif kind == "end_sweep":
            # Manual button: same effect as the agent calling end_sweep.
            if self.state.sweep_ended_t is not None:
                return
            if self.state.sweeping and self.ceiling_missing():
                self._ceiling_reminded = True  # first press: ask for the ceiling; the next press ends
                await self.nudge(self.CEILING_HINT + " Tell them to press End sweep again when done.", "ceiling")
                return
            _, event = run_tool(self.state, "end_sweep", {}, self.rec.t(), self.last_frame_ref)
            self.rec.log("end_sweep_button")
            await self._after_tool("end_sweep", event)
            if self.session is not None:
                await self._safe(self.session.send_realtime_input(text="[monitor] The user pressed End sweep. Tell them processing has started."))
        elif kind == "stop":
            self._client_gone.set()

    # ---------- ceiling coverage ----------

    CEILING_PITCH_DEG = 28.0

    def _track_pitch(self, g: list | None) -> None:
        """Camera pitch from the accelerometer. Wall area needs the ceiling line, and a plain ceiling is
        only reconstructed if the camera actually looks up (a real sweep never went above 18 degrees)."""
        if not g or len(g) != 3:
            return
        n = math.sqrt(sum(x * x for x in g))
        if n < 5:
            return
        self._gravity_y_sum += g[1]
        self._gravity_n += 1
        sign = 1.0 if self._gravity_y_sum > 0 else -1.0  # phone mostly upright: up is along device +y
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -sign * g[2] / n))))  # camera looks along device -z
        self.max_pitch_deg = pitch if self.max_pitch_deg is None else max(self.max_pitch_deg, pitch)

    def ceiling_missing(self) -> bool:
        return (self._gravity_n >= 5 and self.max_pitch_deg is not None
                and self.max_pitch_deg < self.CEILING_PITCH_DEG and not self._ceiling_reminded)

    CEILING_HINT = ("The camera has not looked up at the ceiling yet, so the room height and wall area cannot be "
                    "measured. Before ending, ask the user to stand back and slowly tilt the phone up at two corners "
                    "until the line where the walls meet the ceiling is in view.")

    # ---------- tools ----------

    async def _handle_tool_call(self, session: Any, tool_call: types.LiveServerToolCall) -> None:
        responses = []
        for fc in tool_call.function_calls or []:
            if fc.name == "end_sweep" and self.state.sweeping and self.ceiling_missing():
                self._ceiling_reminded = True  # once: a second end_sweep always ends
                result = {"ok": False, "error": self.CEILING_HINT + " Then call end_sweep again."}
                self.rec.log("tool_call", name=fc.name, args={}, result=result)
                await self._send_json({"type": "nudge", "issue": "ceiling", "text": "Tilt the phone up to show the ceiling corners."})
                responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
                continue
            result, event = run_tool(self.state, fc.name, dict(fc.args or {}), self.rec.t(), self.last_frame_ref)
            self.rec.log("tool_call", name=fc.name, args=dict(fc.args or {}), result=result)
            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
            await self._after_tool(fc.name, event)
        await self._safe(session.send_tool_response(function_responses=responses))

    async def _after_tool(self, name: str, event: dict | None) -> None:
        if name == "end_sweep":
            # The button and the agent can both end the sweep; only the first one counts.
            if self._sweep_end_fired:
                return
            self._sweep_end_fired = True
        if event:
            await self._send_json(event)
        self._sync_meta()
        if name == "end_sweep" and self.on_sweep_end is not None:
            asyncio.create_task(self.on_sweep_end(self))

    # ---------- server -> phone ----------

    async def nudge(self, text: str, issue: str) -> None:
        self.rec.log("nudge", issue=issue, text=text, frame_ref=self.last_frame_ref)
        await self._send_json({"type": "nudge", "issue": issue, "text": text})
        if self.session is not None:
            await self._safe(self.session.send_realtime_input(text=monitor_message(text)))

    async def say(self, text: str) -> None:
        """Injects a system message ([summary] ...) for the agent to speak."""
        if self.session is not None:
            await self._safe(self.session.send_realtime_input(text=text))

    async def _transcript(self, role: str, text: str) -> None:
        if role == "user" and any(c.isalnum() for c in text):
            self.state.last_user_speech_t = self.rec.t()
        self.rec.log("transcript", role=role, text=text)
        await self._send_json({"type": "transcript", "role": role, "text": text})

    async def _status(self, text: str) -> None:
        await self._send_json({"type": "status", "text": text})

    async def _send_json(self, obj: dict[str, Any]) -> None:
        if self._client_gone.is_set():
            return
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(obj, default=str))
            except Exception:
                self._client_gone.set()

    async def _send_bytes(self, data: bytes) -> None:
        if self._client_gone.is_set():
            return
        async with self._send_lock:
            try:
                await self.ws.send_bytes(data)
            except Exception:
                self._client_gone.set()

    async def _safe(self, coro: Awaitable[Any]) -> None:
        """Sends to the live model, swallowing errors from a connection that is mid-resume."""
        try:
            await coro
        except Exception as e:
            self.rec.log("live_send_error", message=str(e))

    def _sync_meta(self) -> None:
        self.rec.meta.update(
            country=self.state.country,
            currency=self.state.currency,
            sweep_started_t=self.state.sweep_started_t,
            sweep_ended_t=self.state.sweep_ended_t,
            frame_count=self.rec.frame_count,
        )
        self.rec.write_meta()
