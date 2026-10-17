// Downsamples microphone audio to 16 kHz mono PCM16 and posts ~100 ms chunks to the main thread.
// A box filter over each output period is enough anti-aliasing for speech.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.chunk = new Int16Array(1600);
    this.n = 0;
    this.acc = 0;
    this.cnt = 0;
    this.phase = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.cnt++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        const s = Math.max(-1, Math.min(1, this.acc / this.cnt));
        this.chunk[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        this.acc = 0;
        this.cnt = 0;
        this.phase -= this.ratio;
        if (this.n === this.chunk.length) {
          this.port.postMessage(this.chunk.buffer, [this.chunk.buffer]);
          this.chunk = new Int16Array(1600);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
