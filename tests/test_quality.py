import cv2
import numpy as np

from app.capture.quality import CaptureMonitor, FrameQuality, Thresholds, assess_grey, assess_jpeg


def _shelf_like(seed=0):
    """Vertical stripes of random width and tone, like a row of spines."""
    rng = np.random.default_rng(seed)
    img = np.zeros((960, 640), np.uint8)
    x = 0
    while x < 640:
        w = int(rng.integers(8, 40))
        img[:, x : x + w] = rng.integers(20, 235)
        x += w
    return img


def test_blur_lowers_sharpness_ratio():
    sharp = assess_grey(_shelf_like())
    blurred = assess_grey(cv2.GaussianBlur(_shelf_like(), (0, 0), 4))
    assert sharp.sharpness > 0.10 > blurred.sharpness


def test_plain_wall_gets_no_blur_verdict():
    wall = np.full((960, 640), 200, np.uint8)
    wall += np.random.default_rng(1).integers(0, 3, wall.shape, dtype=np.uint8)  # sensor noise
    q = assess_grey(wall)
    assert q.edge_coverage < 0.04
    assert CaptureMonitor(Thresholds()).issues(q) == []


def test_glare_and_dark():
    img = np.full((480, 640), 30, np.uint8)
    assert assess_grey(img).brightness < 45
    img[:100, :] = 255
    assert assess_grey(img).glare_frac > 0.2


def test_assess_jpeg_roundtrip():
    ok, buf = cv2.imencode(".jpg", _shelf_like())
    assert ok
    assert assess_jpeg(buf.tobytes()).sharpness is not None


def _q(sharpness=0.15, rotation=None):
    return FrameQuality(sharpness=sharpness, edge_coverage=0.2, glare_frac=0, brightness=120, rotation_dps=rotation)


def test_monitor_needs_persistence():
    mon = CaptureMonitor(Thresholds(), window=4, cooldown_s=6)
    seq = [_q(), _q(0.05), _q(), _q()]
    assert [mon.observe(q, t) for t, q in enumerate(seq)] == [None] * 4


def test_repeated_issue_backs_off():
    mon = CaptureMonitor(Thresholds(), window=2, cooldown_s=6)
    fired = [t for t in range(0, 60) if mon.observe(_q(0.05), float(t)) == "blur"]
    # gaps grow 6, 12, 24... instead of a nudge every 6 s
    gaps = np.diff(fired)
    assert len(fired) <= 4 and all(b > a for a, b in zip(gaps, gaps[1:]))


def test_fast_pan_takes_priority_over_blur():
    assert CaptureMonitor(Thresholds()).issues(_q(0.05, rotation=120)) == ["fast_pan"]
