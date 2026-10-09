
import pytest

from jarvis.config import PROJECT_ROOT, Settings


def test_defaults(monkeypatch):
    for key in ("HONORIFIC", "DB_PATH", "UNITS", "CONFIDENCE_THRESHOLD", "WAKE_ENGINE", "STT_ENGINE"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.address == ", sir"
    assert settings.db_path == PROJECT_ROOT / "data" / "jarvis.db"
    assert settings.effective_wake_engine == settings.stt_engine


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("HONORIFIC", "")
    monkeypatch.setenv("DB_PATH", "elsewhere/reminders.db")
    monkeypatch.setenv("MIC_DEVICE_INDEX", "2")
    monkeypatch.setenv("CHAT_ENABLED", "false")
    monkeypatch.setenv("STT_ENGINE", "Google")
    monkeypatch.setenv("WAKE_ENGINE", "vosk")
    monkeypatch.setenv("LANGUAGE", "en_US:en")      # the shell's locale variable must not leak in
    monkeypatch.setenv("STT_LANGUAGE", "en-GB")
    settings = Settings.from_env()
    assert settings.language == "en-GB"
    assert settings.address == ""
    assert settings.db_path == PROJECT_ROOT / "elsewhere" / "reminders.db"
    assert settings.mic_device_index == 2
    assert settings.chat_enabled is False
    assert (settings.stt_engine, settings.effective_wake_engine) == ("google", "vosk")


def test_empty_voice_means_default_voice(monkeypatch):
    monkeypatch.setenv("TTS_VOICE", "")
    assert Settings.from_env().tts_voice == ""
    monkeypatch.delenv("TTS_VOICE")
    assert Settings.from_env().tts_voice == "Daniel"


def test_bad_number(monkeypatch):
    monkeypatch.setenv("TTS_RATE", "fast")
    with pytest.raises(ValueError, match="TTS_RATE"):
        Settings.from_env()


def test_device_helper():
    from jarvis.device import get_torch_device

    assert str(get_torch_device("cpu")) == "cpu"
    assert str(get_torch_device("auto")) in {"cpu", "mps", "cuda"}
