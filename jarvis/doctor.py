"""`python main.py --doctor`: check every piece of the set-up and say how to fix what's missing."""

from __future__ import annotations

import platform
import shutil
import sys
from typing import List, Tuple

from .config import Settings

OK, WARN, FAIL = "ok", "warn", "FAIL"


def run_checks(settings: Settings) -> List[Tuple[str, str, str]]:
    results: List[Tuple[str, str, str]] = []

    def add(status: str, name: str, detail: str) -> None:
        results.append((status, name, detail))

    # Python ---------------------------------------------------------------------------------
    version = sys.version.split()[0]
    add(OK if sys.version_info >= (3, 10) else FAIL, "Python", version + ("" if sys.version_info >= (3, 10) else " (need 3.10+)"))

    # PyTorch / MPS ---------------------------------------------------------------------------
    from .device import is_apple_silicon, tensorflow_info, torch_info

    info = torch_info()
    if not info.get("installed"):
        add(FAIL, "PyTorch", "not installed: pip install torch")
    else:
        accel = "MPS (Apple GPU)" if info["mps_available"] else ("CUDA" if info["cuda_available"] else "CPU only")
        status = OK if info["mps_available"] or not is_apple_silicon() else WARN
        add(status, "PyTorch", f"{info['version']} - {accel}")

    # TensorFlow (optional) --------------------------------------------------------------------
    if settings.intent_backend == "tf":
        tf = tensorflow_info()
        if not tf.get("installed"):
            add(FAIL, "TensorFlow", "INTENT_BACKEND=tf but TensorFlow is missing: pip install -r requirements-tf.txt")
        elif tf.get("error"):
            add(FAIL, "TensorFlow", tf["error"])
        else:
            add(OK if tf["gpus"] else WARN, "TensorFlow", f"{tf['version']} - GPUs: {tf['gpus'] or 'none (CPU)'}")

    # NLTK -------------------------------------------------------------------------------------
    try:
        from .nlp.preprocess import missing_nltk_data

        missing = missing_nltk_data()
        add(OK if not missing else FAIL, "NLTK data",
            "all resources present" if not missing else f"missing {missing}: python scripts/setup_nlp.py")
    except ImportError:
        add(FAIL, "NLTK", "not installed: pip install nltk")

    # spaCy --------------------------------------------------------------------------------------
    try:
        import spacy

        spacy.load(settings.spacy_model)
        add(OK, "spaCy", f"{spacy.__version__} with {settings.spacy_model}")
    except ImportError:
        add(FAIL, "spaCy", "not installed: pip install spacy")
    except OSError:
        add(FAIL, "spaCy", f"model missing: python -m spacy download {settings.spacy_model}")

    # models -------------------------------------------------------------------------------------
    from .brain.train import meta_path, weights_path

    if meta_path(settings.model_dir, settings.intent_backend).exists() and weights_path(settings.model_dir, settings.intent_backend).exists():
        add(OK, "Intent model", f"trained ({settings.intent_backend})")
    else:
        add(WARN, "Intent model", "not trained yet: python -m jarvis.brain.train (or it trains on first run)")
    if (settings.chat_model_dir / "config.json").exists():
        add(OK, "Chat model", f"fine-tuned model in {settings.chat_model_dir}")
    else:
        add(WARN, "Chat model", "none - using rule-based small talk (optional: python scripts/finetune_chat.py)")

    # microphone / speech-to-text -----------------------------------------------------------------
    try:
        import speech_recognition  # noqa: F401

        from .speech.listener import Listener, ListenerError

        try:
            mics = Listener.list_microphones()
            add(OK if mics else FAIL, "Microphone",
                f"{len(mics)} input device(s): " + ", ".join(name for _, name in mics[:3]) if mics
                else "no input devices found")
        except ListenerError:
            add(FAIL, "PyAudio", "missing: brew install portaudio && pip install pyaudio")
    except ImportError:
        add(FAIL, "SpeechRecognition", "not installed: pip install SpeechRecognition")
    if settings.stt_engine == "google" and platform.system() == "Darwin" and platform.machine() == "arm64":
        add(OK if shutil.which("flac") else WARN, "FLAC encoder",
            "found" if shutil.which("flac") else "not found - Google STT needs it on Apple Silicon: brew install flac")
    if "vosk" in {settings.stt_engine, settings.effective_wake_engine}:
        complete = (settings.vosk_model_path / "conf" / "model.conf").exists()
        add(OK if complete else FAIL, "Vosk model",
            str(settings.vosk_model_path) if complete else "missing or incomplete: python scripts/download_vosk.py")

    # text-to-speech -------------------------------------------------------------------------------
    from .speech.speaker import build_engine

    engine = build_engine(settings.tts_engine, settings.tts_voice, settings.tts_rate)
    add(OK if engine.name != "print" else WARN, "Text-to-speech",
        f"using '{engine.name}'" + (" (no voice output)" if engine.name == "print" else ""))
    engine.close()

    # weather --------------------------------------------------------------------------------------
    add(OK if settings.openweather_api_key else WARN, "OpenWeatherMap key",
        "set" if settings.openweather_api_key else "missing: add OPENWEATHER_API_KEY to .env")
    add(OK if settings.default_city else WARN, "Default city",
        settings.default_city or "not set (Jarvis will ask which city): DEFAULT_CITY in .env")
    return results


def main(settings: Settings) -> int:
    print("J.A.R.V.I.S. set-up check\n")
    results = run_checks(settings)
    width = max(len(name) for _, name, _ in results)
    symbols = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
    for status, name, detail in results:
        print(f"  {symbols[status]} {name.ljust(width)}  {detail}")
    failures = sum(1 for status, _, _ in results if status == FAIL)
    print("\nAll good!" if not failures else f"\n{failures} problem(s) to fix before voice mode will work.")
    return 1 if failures else 0
