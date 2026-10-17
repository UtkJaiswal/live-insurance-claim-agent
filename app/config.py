"""Runtime settings, read once from the environment (.env is loaded if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    live_model: str
    vision_model: str
    live_voice: str
    serper_api_key: str
    google_books_key: str
    tinyfish_api_key: str
    price_extract_model: str
    tinyfish_search_rpm: int
    price_deadline_s: float
    room_ensemble: int
    default_country: str
    default_currency: str
    sweeps_dir: Path
    # Capture tuning. Thresholds are logged with every frame so they can be recalibrated from recordings.
    blur_ratio_threshold: float
    min_texture: float
    glare_threshold: float
    dark_threshold: float
    nudge_cooldown_s: float
    max_analysed_frames: int


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=_env("GEMINI_API_KEY"),
        live_model=_env("LIVE_MODEL", "gemini-3.8-live"),
        vision_model=_env("VISION_MODEL", "gemini-3.8-flash"),
        live_voice=_env("LIVE_VOICE", "Kore"),
        serper_api_key=_env("SERPER_API_KEY"),
        google_books_key=_env("GOOGLE_BOOKS_API_KEY"),
        tinyfish_api_key=_env("TINYFISH_API_KEY"),
        price_extract_model=_env("PRICE_EXTRACT_MODEL", "gemini-3.5-flash-lite"),
        tinyfish_search_rpm=int(_env("TINYFISH_SEARCH_RPM", "30")),
        price_deadline_s=float(_env("PRICE_DEADLINE_S", "210")),
        room_ensemble=int(_env("ROOM_ENSEMBLE", "5")),
        default_country=_env("DEFAULT_COUNTRY", "IN"),
        default_currency=_env("DEFAULT_CURRENCY", "INR"),
        sweeps_dir=Path(_env("SWEEPS_DIR", str(ROOT / "sweeps"))),
        blur_ratio_threshold=float(_env("BLUR_RATIO_THRESHOLD", "0.10")),
        min_texture=float(_env("MIN_TEXTURE", "0.04")),
        glare_threshold=float(_env("GLARE_THRESHOLD", "0.03")),
        dark_threshold=float(_env("DARK_THRESHOLD", "45")),
        nudge_cooldown_s=float(_env("NUDGE_COOLDOWN_S", "6")),
        max_analysed_frames=int(_env("MAX_ANALYSED_FRAMES", "90")),
    )
