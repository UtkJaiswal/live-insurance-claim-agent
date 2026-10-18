// Phone capture client: mic + camera + motion -> server, agent voice + events <- server.
// Protocol is documented in app/live/bridge.py.

const MSG_AUDIO = 1, MSG_PREVIEW = 2, MSG_KEYFRAME = 3;
const params = new URLSearchParams(location.search);
const KEYFRAME_MS = Number(params.get("kf") || 1000);   // high-res frames for the pipeline
const PREVIEW_MS = 1000;                                 // live model accepts <= 1 fps
const PREVIEW_MAX_SIDE = 768;
const KEYFRAME_QUALITY = Number(params.get("q") || 0.8);
// Long side of high-res frames. 4K frames (~1.3 MB) saturated a ~3 Mbit/s library uplink: frames
// arrived 30 s late, the voice lagged, and the server dropped the socket at 40 s. 2560 px is ~3x
// smaller and still reads spines at arm's length. ?kmax=3840 restores full 4K on fast networks.
const KEYFRAME_MAX_SIDE = Number(params.get("kmax") || 2560);
// Voice must stay real-time: skip frames (and previews) whenever this much is still waiting to upload.
const MAX_BUFFERED = Number(params.get("buf") || 1.5) * 1024 * 1024;

const $ = (id) => document.getElementById(id);
const video = $("video");
const enc = new TextEncoder();

let ws, audioCtx, micStream, timers = [], captureTimers = [], playHead = 0, playing = [];
let t0 = 0, keyframeBusy = false, sentFrames = 0, droppedFrames = 0;
const motion = { rotation_dps: null, gravity: null, orientation: null };

$("start").onclick = start;
$("end").onclick = () => send(JSON.stringify({ type: "end_sweep" }));
$("stop").onclick = stop;

// A locked phone or a backgrounded tab sends frozen or black frames; end the session instead.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden" && ws && ws.readyState === WebSocket.OPEN && !processing) stop();
});

async function start() {
  $("start").disabled = true;
  try {
    // iOS needs these inside the tap handler.
    if (typeof DeviceMotionEvent !== "undefined" && DeviceMotionEvent.requestPermission) {
      await DeviceMotionEvent.requestPermission().catch(() => {});
    }
    if (typeof DeviceOrientationEvent !== "undefined" && DeviceOrientationEvent.requestPermission) {
      await DeviceOrientationEvent.requestPermission().catch(() => {});
    }
    audioCtx = new AudioContext();
    await audioCtx.resume();

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30 },
      },
    });
    video.srcObject = micStream;
    await video.play();

    await audioCtx.audioWorklet.addModule("pcm-worklet.js?v=3");
    const src = audioCtx.createMediaStreamSource(new MediaStream(micStream.getAudioTracks()));
    const worklet = new AudioWorkletNode(audioCtx, "pcm-capture");
    worklet.port.onmessage = (e) => sendTyped(MSG_AUDIO, new Uint8Array(e.data));
    src.connect(worklet);
    // Some browsers only pull a worklet that reaches the destination; route it through silence.
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    worklet.connect(mute).connect(audioCtx.destination);

    listenMotion();
    connect();
  } catch (err) {
    setStatus("error: " + err.message, "bad");
    $("start").disabled = false;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    t0 = performance.now();
    const vt = micStream.getVideoTracks()[0];
    send(JSON.stringify({ type: "hello", device: navigator.userAgent, camera: vt.getSettings() }));
    setStatus("connecting", "warn");
    captureTimers = [setInterval(sendPreview, PREVIEW_MS), setInterval(sendKeyframe, KEYFRAME_MS)];
    timers.push(...captureTimers, setInterval(updateStats, 1000));
    $("end").disabled = false;
    $("stop").disabled = false;
  };
  ws.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : playPcm(e.data));
  ws.onclose = () => { setStatus("disconnected", "bad"); cleanup(); };
}

let processing = false;

function stop() {
  if (processing && !confirm("The claim packet is still being built. Stop anyway?")) return;
  send(JSON.stringify({ type: "stop" }));
  ws && ws.close();
}

function cleanup() {
  timers.forEach(clearInterval);
  timers = [];
  micStream && micStream.getTracks().forEach((t) => t.stop());
  $("start").disabled = false;
  $("end").disabled = true;
  $("stop").disabled = true;
}

// ---------- sending ----------

function send(data) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(data);
}

function sendTyped(type, bytes) {
  const msg = new Uint8Array(1 + bytes.byteLength);
  msg[0] = type;
  msg.set(bytes, 1);
  send(msg);
}

const previewCanvas = document.createElement("canvas");
const keyCanvas = document.createElement("canvas");

function grab(canvas, maxSide) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return null;
  const s = maxSide ? Math.min(1, maxSide / Math.max(w, h)) : 1;
  canvas.width = Math.round(w * s);
  canvas.height = Math.round(h * s);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function sendPreview() {
  if (ws.bufferedAmount > MAX_BUFFERED) return;
  const c = grab(previewCanvas, PREVIEW_MAX_SIDE);
  if (!c) return;
  c.toBlob(async (b) => b && sendTyped(MSG_PREVIEW, new Uint8Array(await b.arrayBuffer())), "image/jpeg", 0.6);
}

function sendKeyframe() {
  if (keyframeBusy) return;
  if (ws.bufferedAmount > MAX_BUFFERED) { droppedFrames++; return; }
  const c = grab(keyCanvas, KEYFRAME_MAX_SIDE);
  if (!c) return;
  keyframeBusy = true;
  const header = {
    client_t: +((performance.now() - t0) / 1000).toFixed(3),
    w: c.width, h: c.height,
    rotation_dps: motion.rotation_dps,
    gravity: motion.gravity,
    orientation: motion.orientation,
    screen_angle: (screen.orientation && screen.orientation.angle) || 0,
  };
  c.toBlob(async (b) => {
    keyframeBusy = false;
    if (!b) return;
    const h = enc.encode(JSON.stringify(header));
    const jpeg = new Uint8Array(await b.arrayBuffer());
    const msg = new Uint8Array(1 + 4 + h.length + jpeg.length);
    msg[0] = MSG_KEYFRAME;
    new DataView(msg.buffer).setUint32(1, h.length);
    msg.set(h, 5);
    msg.set(jpeg, 5 + h.length);
    send(msg);
    sentFrames++;
  }, "image/jpeg", KEYFRAME_QUALITY);
}

function listenMotion() {
  addEventListener("devicemotion", (e) => {
    const r = e.rotationRate;
    if (r && r.alpha != null) motion.rotation_dps = +Math.hypot(r.alpha, r.beta, r.gamma).toFixed(1);
    const g = e.accelerationIncludingGravity;
    if (g && g.x != null) motion.gravity = [g.x, g.y, g.z].map((v) => +v.toFixed(3));
  });
  addEventListener("deviceorientation", (e) => {
    if (e.alpha != null) motion.orientation = [e.alpha, e.beta, e.gamma].map((v) => +v.toFixed(1));
  });
}

// ---------- agent audio ----------

function playPcm(buf) {
  const pcm = new Int16Array(buf);
  const audio = audioCtx.createBuffer(1, pcm.length, 24000);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 0x8000;
  const node = audioCtx.createBufferSource();
  node.buffer = audio;
  node.connect(audioCtx.destination);
  playHead = Math.max(playHead, audioCtx.currentTime + 0.03);
  node.start(playHead);
  playHead += audio.duration;
  playing.push(node);
  node.onended = () => (playing = playing.filter((n) => n !== node));
}

function stopPlayback() {
  playing.forEach((n) => { try { n.stop(); } catch {} });
  playing = [];
  playHead = 0;
}

// ---------- events ----------

let lastRole = null, lastP = null;

function onEvent(ev) {
  switch (ev.type) {
    case "session": setStatus("session " + ev.id, "warn"); break;
    case "status": setStatus(ev.text, ev.text === "connected" ? "ok" : "warn"); break;
    case "transcript": appendTranscript(ev.role, ev.text); break;
    case "turn_complete": lastRole = null; break;
    case "interrupted": stopPlayback(); lastRole = null; break;
    case "locale": $("locale").textContent = `${ev.country} · ${ev.currency}`; break;
    case "region": $("region").textContent = ev.label; addInventory("region", ev.label); break;
    case "item": addInventory(ev.category, ev.description + (ev.brand_model ? ` (${ev.brand_model})` : "")); break;
    case "statement": addInventory(ev.kind, `${ev.subject}: ${ev.statement}`); break;
    case "book": addInventory("book", ev.label); break;
    case "sweep":
      addInventory("sweep", ev.state);
      if (ev.state === "ended") {
        // Keep the mic open for the read-back, stop sending camera frames.
        captureTimers.forEach(clearInterval);
        captureTimers = [];
        $("end").disabled = true;
        processing = true;
        showNudge("Building your claim packet. Keep this page open: the agent will read the summary in 1-3 minutes.", 600000);
      }
      break;
    case "nudge": if (ev.issue !== "art_question") showNudge(ev.text); break;
    case "scan": $("scan").textContent = `${ev.books_in_view} books in view · ${ev.frames_scanned} frames scanned`; break;
    case "packet": processing = false; $("nudge").style.display = "none"; showPacket(ev); break;
    case "quality": showQuality(ev); break;
  }
}

function appendTranscript(role, text) {
  const box = $("transcript");
  if (role !== lastRole || !lastP) {
    lastP = document.createElement("p");
    lastP.className = role;
    lastP.textContent = role === "agent" ? "Agent: " : "You: ";
    box.appendChild(lastP);
    lastRole = role;
  }
  lastP.textContent += text;
  box.scrollTop = box.scrollHeight;
}

function addInventory(tag, text) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="tag"></span><span></span>`;
  li.children[0].textContent = tag;
  li.children[1].textContent = text;
  $("inventory").prepend(li);
}

function showPacket(ev) {
  const box = $("packet");
  box.style.display = "block";
  box.innerHTML = "";
  const h = document.createElement("h2");
  h.textContent = `Claim packet ready (${ev.time_to_packet_s} s after the sweep)`;
  const p = document.createElement("p");
  p.textContent = ev.summary;
  const links = document.createElement("p");
  for (const [label, href] of [["Open report", ev.report], ["claim_packet.json", ev.json]]) {
    const a = document.createElement("a");
    a.href = href;
    a.target = "_blank";
    a.textContent = label;
    a.style.marginRight = "16px";
    links.appendChild(a);
  }
  box.append(h, p, links);
}

let nudgeTimer;
function showNudge(text, ms = 4000) {
  const n = $("nudge");
  n.textContent = text;
  n.style.display = "block";
  clearTimeout(nudgeTimer);
  nudgeTimer = setTimeout(() => (n.style.display = "none"), ms);
}

function showQuality(ev) {
  const level = ev.issues.length === 0 ? "ok" : ev.issues.length === 1 ? "warn" : "bad";
  $("qdot").className = "dot " + level;
  const sharp = ev.sharpness == null ? "low texture" : `sharp ${ev.sharpness.toFixed(2)}`;
  $("qtext").textContent = ev.issues.length ? ev.issues.join(", ") : sharp;
}

function setStatus(text, level) {
  $("status").lastChild.textContent = text;
  $("conn").className = "dot " + (level || "");
}

function updateStats() {
  const vt = micStream && micStream.getVideoTracks()[0];
  const s = vt ? vt.getSettings() : {};
  const backlog = ws ? (ws.bufferedAmount / 1048576).toFixed(1) : "0";
  $("stats").textContent = `camera ${s.width}×${s.height} · frames sent ${sentFrames} · skipped ${droppedFrames} · upload backlog ${backlog} MB` +
    (droppedFrames > sentFrames ? " · SLOW NETWORK" : "");
}
