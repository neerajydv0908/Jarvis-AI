"""Text-to-speech (roadmap step 6).

Engines:
  * ``say``     - macOS's built-in speech command. The default on a Mac: it runs in its
                  own process, so it can never hang the assistant.
  * ``pyttsx3`` - offline, cross-platform (NSSpeechSynthesizer on macOS, SAPI5 on
                  Windows, eSpeak on Linux). The default elsewhere.
  * ``gtts``    - Google Text-to-Speech: more natural, needs internet (British accent
                  via the co.uk endpoint, for the full J.A.R.V.I.S. effect).
  * ``print``   - silent; text only.

All speech goes through one worker thread, so engines that are not thread-safe
(pyttsx3) are only ever touched from a single thread, and ``is_speaking`` tells
the microphone listener when Jarvis is talking.
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


class SpeechEngine:
    name = "base"

    def speak(self, text: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class PrintEngine(SpeechEngine):
    name = "print"

    def speak(self, text: str) -> None:
        pass  # the interface already prints every reply


_DEFAULT_VOICE_WORDS = {"", "default", "system"}


def pick_voice(requested: str, available: List[str]) -> Optional[str]:
    """Match a configured voice name against the installed ones.

    An exact (case-insensitive) match wins; otherwise a downloaded quality variant
    such as "Daniel (Premium)" or "Daniel (Enhanced)" is used. None means the Mac's
    default voice.
    """
    wanted = (requested or "").strip().lower()
    if wanted in _DEFAULT_VOICE_WORDS:
        return None
    for name in available:
        if name.lower() == wanted:
            return name
    variants = [name for name in available if name.lower().startswith(wanted + " (")]
    for quality in ("premium", "enhanced"):
        for name in variants:
            if quality in name.lower():
                return name
    return variants[0] if variants else None


class SayEngine(SpeechEngine):
    """macOS ``say``."""

    name = "say"

    def __init__(self, voice: str = "Daniel", rate: int = 185):
        if not shutil.which("say"):
            raise RuntimeError("the 'say' command is only available on macOS")
        self.voice = pick_voice(voice, self.available_voices())
        if self.voice is None and (voice or "").strip().lower() not in _DEFAULT_VOICE_WORDS:
            log.warning(
                "The '%s' voice isn't installed, so Jarvis is using your Mac's default voice. To add it: "
                "System Settings > Accessibility > Read & Speak (Spoken Content on older macOS) > the (i) "
                "button next to System voice > English > %s. Or set TTS_VOICE in .env to a voice listed by "
                "`say -v '?'`, or leave it empty for the default voice.", voice, voice)
        self.rate = rate

    @staticmethod
    def available_voices() -> List[str]:
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        voices = []
        for line in out.splitlines():
            # "Daniel              en_GB    # Hello! My name is Daniel."
            name = line.split("#")[0].rsplit(None, 1)[0].strip() if "#" in line else ""
            if name:
                voices.append(name)
        return voices

    def speak(self, text: str) -> None:
        cmd = ["say", "-r", str(self.rate)]
        if self.voice:
            cmd += ["-v", self.voice]
        subprocess.run(cmd + [text.lstrip("-")], check=False)


class Pyttsx3Engine(SpeechEngine):
    name = "pyttsx3"

    def __init__(self, voice: str = "", rate: int = 185):
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        if voice:
            for candidate in self.engine.getProperty("voices"):
                if voice.lower() in (candidate.name or "").lower() or voice.lower() in (candidate.id or "").lower():
                    self.engine.setProperty("voice", candidate.id)
                    break
            else:
                log.warning("pyttsx3 voice %r not found; using the default voice.", voice)

    def speak(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()

    def close(self) -> None:
        try:
            self.engine.stop()
        except Exception:  # pragma: no cover
            pass


class GTTSEngine(SpeechEngine):
    name = "gtts"

    def __init__(self, tld: str = "co.uk"):
        from gtts import gTTS  # noqa: F401  (fail early if missing)

        self.tld = tld
        self.player = self._find_player()
        if not self.player:
            raise RuntimeError("no audio player found for gTTS (need afplay, mpg123, ffplay or mpv)")

    @staticmethod
    def _find_player() -> Optional[List[str]]:
        for cmd in (["afplay"], ["mpg123", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
                    ["mpv", "--really-quiet"]):
            if shutil.which(cmd[0]):
                return cmd
        return None

    def speak(self, text: str) -> None:
        from gtts import gTTS

        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            gTTS(text=text, lang="en", tld=self.tld).save(path)
            subprocess.run(self.player + [path], check=False)
        finally:
            os.unlink(path)


def build_engine(name: str = "auto", voice: str = "Daniel", rate: int = 185) -> SpeechEngine:
    """Create the requested engine, falling back to the next one that works."""
    name = (name or "auto").lower()
    if name == "auto":
        order = ["say", "pyttsx3", "gtts"] if platform.system() == "Darwin" else ["pyttsx3", "gtts"]
    else:
        order = [name]
    for candidate in order + ["print"]:
        try:
            if candidate == "say":
                return SayEngine(voice, rate)
            if candidate == "pyttsx3":
                return Pyttsx3Engine(voice, rate)
            if candidate == "gtts":
                return GTTSEngine()
            if candidate == "print":
                return PrintEngine()
            log.warning("Unknown TTS engine %r", candidate)
        except Exception as exc:
            log.warning("TTS engine %s unavailable: %s", candidate, exc)
    return PrintEngine()


class Speaker:
    """Thread-safe front end: ``say()`` queues text for a single worker thread."""

    timeout_base = 20.0  # seconds; plus ~0.15 s per character before an engine counts as stuck

    def __init__(self, engine: str = "auto", voice: str = "Daniel", rate: int = 185,
                 engine_instance: Optional[SpeechEngine] = None):
        self._queue: "queue.Queue[Optional[Tuple[str, threading.Event]]]" = queue.Queue()
        self._speaking = threading.Event()
        self._ready = threading.Event()
        self._engine: Optional[SpeechEngine] = engine_instance
        self._config = (engine, voice, rate)
        self._stuck = False
        self._worker = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._worker.start()
        self._ready.wait()

    @property
    def engine_name(self) -> str:
        return self._engine.name if self._engine else "none"

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set() or not self._queue.empty()

    def say(self, text: str, block: bool = True) -> None:
        if not text or self._stuck:
            return
        done = threading.Event()
        self._queue.put((text, done))
        if block and not done.wait(timeout=self.timeout_base + 0.15 * len(text) * (1 + self._queue.qsize())):
            # e.g. an engine whose "finished speaking" callback never arrives
            self._stuck = True
            log.error("Text-to-speech engine '%s' stopped responding; continuing with text only. "
                      "Try TTS_ENGINE=say (macOS) or TTS_ENGINE=gtts in .env.", self.engine_name)

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5)

    def _run(self) -> None:
        if self._engine is None:  # create the engine on the thread that will use it
            self._engine = build_engine(*self._config)
        log.info("Text-to-speech engine: %s", self._engine.name)
        self._ready.set()
        while True:
            item = self._queue.get()
            if item is None:
                self._engine.close()
                return
            text, done = item
            self._speaking.set()
            try:
                self._engine.speak(text)
            except Exception:
                log.exception("Text-to-speech failed")
            finally:
                self._speaking.clear()
                done.set()
