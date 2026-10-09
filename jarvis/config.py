"""Central configuration.

Every setting can be overridden with an environment variable or a line in the
project's ``.env`` file (see ``.env.example``). Paths may be relative to the
project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # python-dotenv is a soft dependency so the test-suite never needs it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _int(name: str, default: Optional[int]) -> Optional[int]:
    raw = _str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = _str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    raw = _str(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class Settings:
    # --- identity -----------------------------------------------------------
    name: str = "Jarvis"
    wake_word: str = "jarvis"
    honorific: str = "sir"            # how Jarvis addresses you ("" to disable)
    creator: str = ""                 # who built this Jarvis (used in "who made you")

    # --- speech-to-text -----------------------------------------------------
    stt_engine: str = "google"        # google | vosk
    wake_engine: str = ""             # engine used for wake-word spotting; "" = same as stt_engine
    language: str = "en-US"
    vosk_model_path: Path = PROJECT_ROOT / "models" / "vosk-model-small-en-us-0.15"
    mic_device_index: Optional[int] = None
    calibration_seconds: float = 1.0  # ambient-noise calibration at start-up
    wake_listen_timeout: float = 2.0  # seconds to wait for speech to start while asleep
    wake_phrase_limit: float = 5.0    # max length of a wake-word utterance
    command_timeout: float = 6.0      # seconds to wait for a command after the wake word
    command_phrase_limit: float = 12.0
    follow_up_seconds: float = 6.0    # keep listening this long after a reply (0 = always need the wake word)

    # --- text-to-speech -----------------------------------------------------
    tts_engine: str = "auto"          # auto | say | pyttsx3 | gtts | print
    tts_voice: str = "Daniel"         # British English voice on macOS
    tts_rate: int = 185               # words per minute

    # --- NLP / models ---------------------------------------------------------
    spacy_model: str = "en_core_web_sm"
    intent_backend: str = "torch"     # torch | tf
    model_dir: Path = PROJECT_ROOT / "models"
    torch_device: str = "auto"        # auto | mps | cuda | cpu
    confidence_threshold: float = 0.4

    # --- generative chat ------------------------------------------------------
    chat_enabled: bool = True
    chat_model_dir: Path = PROJECT_ROOT / "models" / "chat"

    # --- action modules -------------------------------------------------------
    openweather_api_key: str = ""
    default_city: str = ""
    units: str = "imperial"           # imperial (F, mph) | metric (C, m/s)
    db_path: Path = PROJECT_ROOT / "data" / "jarvis.db"

    @property
    def effective_wake_engine(self) -> str:
        return (self.wake_engine or self.stt_engine).lower()

    @property
    def address(self) -> str:
        """', sir' style suffix for replies (empty when no honorific is set)."""
        return f", {self.honorific}" if self.honorific else ""

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        raw_honorific = os.getenv("HONORIFIC")  # set it to an empty value to switch it off
        raw_voice = os.getenv("TTS_VOICE")      # empty = the Mac's default voice
        return cls(
            name=_str("JARVIS_NAME", d.name),
            wake_word=_str("WAKE_WORD", d.wake_word).lower(),
            honorific=d.honorific if raw_honorific is None else raw_honorific.strip(),
            creator=_str("CREATOR_NAME", d.creator),
            stt_engine=_str("STT_ENGINE", d.stt_engine).lower(),
            wake_engine=_str("WAKE_ENGINE", d.wake_engine).lower(),
            language=_str("STT_LANGUAGE", d.language),  # not LANGUAGE: that's a locale variable
            vosk_model_path=_path("VOSK_MODEL_PATH", d.vosk_model_path),
            mic_device_index=_int("MIC_DEVICE_INDEX", d.mic_device_index),
            calibration_seconds=_float("CALIBRATION_SECONDS", d.calibration_seconds),
            wake_listen_timeout=_float("WAKE_LISTEN_TIMEOUT", d.wake_listen_timeout),
            wake_phrase_limit=_float("WAKE_PHRASE_LIMIT", d.wake_phrase_limit),
            command_timeout=_float("COMMAND_TIMEOUT", d.command_timeout),
            command_phrase_limit=_float("COMMAND_PHRASE_LIMIT", d.command_phrase_limit),
            follow_up_seconds=_float("FOLLOW_UP_SECONDS", d.follow_up_seconds),
            tts_engine=_str("TTS_ENGINE", d.tts_engine).lower(),
            tts_voice=d.tts_voice if raw_voice is None else raw_voice.strip(),
            tts_rate=_int("TTS_RATE", d.tts_rate) or d.tts_rate,
            spacy_model=_str("SPACY_MODEL", d.spacy_model),
            intent_backend=_str("INTENT_BACKEND", d.intent_backend).lower(),
            model_dir=_path("MODEL_DIR", d.model_dir),
            torch_device=_str("TORCH_DEVICE", d.torch_device).lower(),
            confidence_threshold=_float("CONFIDENCE_THRESHOLD", d.confidence_threshold),
            chat_enabled=_bool("CHAT_ENABLED", d.chat_enabled),
            chat_model_dir=_path("CHAT_MODEL_DIR", d.chat_model_dir),
            openweather_api_key=_str("OPENWEATHER_API_KEY", d.openweather_api_key),
            default_city=_str("DEFAULT_CITY", d.default_city),
            units=_str("UNITS", d.units).lower(),
            db_path=_path("DB_PATH", d.db_path),
        )
