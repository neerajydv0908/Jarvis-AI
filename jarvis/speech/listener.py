"""Speech-to-text listener with wake-word detection (roadmap step 2).

* ``SpeechRecognition`` + ``PyAudio`` capture the microphone.
* ``adjust_for_ambient_noise`` calibrates the energy threshold at start-up so
  background hum doesn't trigger transcription.
* Transcription uses Google's Web Speech API (online, accurate) or Vosk
  (offline, private). Vosk is called directly so its model loads only once.
* The microphone is opened only while listening, so Jarvis never hears itself
  while it is talking.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

log = logging.getLogger(__name__)

PYAUDIO_HINT = ("PyAudio is missing. On macOS run `brew install portaudio` and then `pip install pyaudio`.")
MIC_HINT = ("Could not open the microphone. Check it is connected, and allow microphone access for your "
            "terminal app in System Settings > Privacy & Security > Microphone.")


class ListenerError(RuntimeError):
    """Problems the user must fix (no microphone, missing model, no internet...)."""


class WakeWordDetector:
    """Finds the wake word in a transcript, tolerating near-misses like 'jervis'."""

    def __init__(self, wake_word: str = "jarvis", threshold: float = 0.8, aliases: Tuple[str, ...] = ()):
        self.wake_word = wake_word.lower()
        self.threshold = threshold
        self.aliases = {a.lower() for a in aliases} | {self.wake_word}

    def _matches(self, word: str) -> bool:
        word = re.sub(r"'s$", "", word.lower())
        if word in self.aliases:
            return True
        return len(word) >= 4 and difflib.SequenceMatcher(None, word, self.wake_word).ratio() >= self.threshold

    def detect(self, transcript: str) -> Tuple[bool, str]:
        """Return (heard_wake_word, command_spoken_with_it).

        "jarvis what time is it" -> (True, "what time is it")
        "what time is it jarvis" -> (True, "what time is it")
        "hey jarvis"             -> (True, "")
        """
        words = list(re.finditer(r"[A-Za-z']+", transcript))
        for i, match in enumerate(words):
            if self._matches(match.group()):
                after = transcript[match.end():].strip(" ,.!?")
                before = transcript[:match.start()].strip(" ,.!?")
                before = re.sub(r"^(?:hey|hi|hello|ok|okay|yo|oh)\b[\s,]*", "", before, flags=re.I).strip()
                command = after or (before if len(before.split()) >= 2 else "")
                return True, command
        return False, ""


class Listener:
    def __init__(self, engine: str = "google", wake_engine: Optional[str] = None, language: str = "en-US",
                 vosk_model_path: Optional[Path] = None, device_index: Optional[int] = None,
                 wake_word: str = "jarvis"):
        try:
            import speech_recognition as sr
        except ImportError as exc:  # pragma: no cover
            raise ListenerError("SpeechRecognition is not installed: pip install SpeechRecognition") from exc
        self.sr = sr
        self.engine = engine.lower()
        self.wake_engine = (wake_engine or engine).lower()
        self.language = language
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8       # seconds of silence that end a phrase
        self.recognizer.non_speaking_duration = 0.4
        self.recognizer.operation_timeout = 10      # don't hang forever on a slow network
        try:
            self.microphone = sr.Microphone(device_index=device_index)
        except AttributeError as exc:  # SpeechRecognition's way of saying PyAudio is missing
            raise ListenerError(PYAUDIO_HINT) from exc
        except (AssertionError, OSError) as exc:  # bad MIC_DEVICE_INDEX, or no input device at all
            raise ListenerError(f"Microphone {device_index if device_index is not None else '(default)'} "
                                f"is not usable: {exc}. See `python main.py --list-mics`.") from exc
        if device_index is not None:
            self._check_input_device(device_index)
        self._vosk_model = None
        self._wake_grammar: Optional[List[str]] = None
        if "vosk" in {self.engine, self.wake_engine}:
            self._vosk_model = self._load_vosk(vosk_model_path)
            self._wake_grammar = [self.wake_word, "[unk]"]
            find_word = getattr(self._vosk_model, "vosk_model_find_word", None)
            if find_word is not None and find_word(self.wake_word) < 0:
                # Vosk can never output a word it doesn't know, so it can't spot this wake word
                self._wake_grammar = None
                if self.engine != "vosk":
                    log.warning("'%s' is not in the Vosk vocabulary; using %s for the wake word instead.",
                                self.wake_word, self.engine)
                    self.wake_engine = self.engine
                else:
                    log.warning("'%s' is not in the Vosk vocabulary, so offline mode may never wake up. "
                                "Choose another WAKE_WORD or use STT_ENGINE=google.", self.wake_word)

    # -- set-up ---------------------------------------------------------------------------
    @staticmethod
    def _load_vosk(path: Optional[Path]):
        try:
            from vosk import Model, SetLogLevel
        except ImportError as exc:
            raise ListenerError("Vosk is not installed: pip install vosk") from exc
        if not path or not Path(path).exists():
            raise ListenerError(f"Vosk model not found at {path}. Run: python scripts/download_vosk.py")
        SetLogLevel(-1)
        try:
            return Model(str(path))
        except Exception as exc:  # vosk raises a bare Exception for broken/partial model folders
            raise ListenerError(f"The Vosk model in {path} could not be loaded (incomplete download?). "
                                "Delete that folder and run: python scripts/download_vosk.py") from exc

    @staticmethod
    def _check_input_device(index: int) -> None:
        import pyaudio

        audio = pyaudio.PyAudio()
        try:
            info = audio.get_device_info_by_index(index)
        finally:
            audio.terminate()
        if int(info.get("maxInputChannels", 0)) < 1:
            raise ListenerError(f"Device {index} ({info.get('name')}) has no microphone input. "
                                "Pick one from `python main.py --list-mics`.")

    @staticmethod
    def list_microphones() -> List[Tuple[int, str]]:
        """(index, name) of every input device - use the index as MIC_DEVICE_INDEX."""
        try:
            import pyaudio
        except ImportError as exc:
            raise ListenerError(PYAUDIO_HINT) from exc
        audio = pyaudio.PyAudio()
        try:
            devices = []
            for index in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(index)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append((index, str(info.get("name", f"device {index}"))))
            return devices
        finally:
            audio.terminate()

    @contextlib.contextmanager
    def _open(self) -> Iterator:
        """Open the microphone for one listen; turn every device failure into a ListenerError."""
        mic = self.microphone
        try:
            source = mic.__enter__()
        except Exception as exc:
            raise ListenerError(f"{MIC_HINT} ({exc})") from exc
        if getattr(mic, "stream", None) is None:  # SpeechRecognition swallows PyAudio's open() error
            raise ListenerError(MIC_HINT)
        try:
            yield source
        finally:
            try:
                mic.__exit__(None, None, None)
            except Exception:
                log.debug("Closing the microphone failed", exc_info=True)

    def calibrate(self, seconds: float = 1.0) -> float:
        """Measure background noise and set the energy threshold above it."""
        with self._open() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=seconds)
        log.info("Calibrated energy threshold: %.0f", self.recognizer.energy_threshold)
        return self.recognizer.energy_threshold

    # -- listening ---------------------------------------------------------------------------
    def capture(self, timeout: Optional[float], phrase_time_limit: Optional[float]):
        """Record one phrase. Returns AudioData, or None if nobody started speaking in time."""
        with self._open() as source:
            try:
                return self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except self.sr.WaitTimeoutError:
                return None
            except (OSError, AssertionError, AttributeError) as exc:  # e.g. unplugged mid-listen
                raise ListenerError(f"Microphone error: {exc}") from exc

    def listen(self, timeout: Optional[float] = 5, phrase_time_limit: Optional[float] = 10) -> Optional[str]:
        """Record and transcribe a command. Returns None on silence or unintelligible audio."""
        audio = self.capture(timeout, phrase_time_limit)
        return self.transcribe(audio, self.engine) if audio is not None else None

    def listen_for_wake_word(self, timeout: Optional[float] = 2, phrase_time_limit: Optional[float] = 5) -> Optional[str]:
        """Record a phrase and return its transcript only if it may contain the wake word."""
        audio = self.capture(timeout, phrase_time_limit)
        if audio is None:
            return None
        if self.wake_engine == "vosk" and self.engine != "vosk":
            # Cheap offline keyword spotting first; only send audio to Google once
            # Vosk heard the wake word (to get the rest of the sentence).
            spotted = self._vosk(audio, grammar=self._wake_grammar)
            if not spotted or self.wake_word not in spotted.split():
                return None
            full = self.transcribe(audio, self.engine)
            return full if full else self.wake_word
        return self.transcribe(audio, self.wake_engine)

    # -- speech-to-text ----------------------------------------------------------------------------
    def transcribe(self, audio, engine: str) -> Optional[str]:
        if engine == "vosk":
            return self._vosk(audio) or None
        try:
            return self.recognizer.recognize_google(audio, language=self.language)
        except self.sr.UnknownValueError:
            return None
        except (self.sr.RequestError, TimeoutError, ConnectionError) as exc:
            if self._vosk_model is not None:
                log.warning("Google speech API unavailable (%s); falling back to Vosk.", exc)
                return self._vosk(audio) or None
            raise ListenerError(f"Google's speech service is unreachable - check your internet connection ({exc}).") from exc
        except OSError as exc:
            # SpeechRecognition ships an Intel-only 'flac' encoder: "Bad CPU type in executable" (errno 86)
            if "flac" in str(exc).lower() or getattr(exc, "errno", None) == 86:
                raise ListenerError("Could not encode audio for Google. On Apple Silicon run `brew install flac`.") from exc
            raise ListenerError(f"Speech recognition failed: {exc}") from exc

    def _vosk(self, audio, grammar: Optional[List[str]] = None) -> str:
        from vosk import KaldiRecognizer

        if self._vosk_model is None:
            raise ListenerError("Vosk model not loaded")
        rate = 16000
        recognizer = (KaldiRecognizer(self._vosk_model, rate, json.dumps(grammar)) if grammar
                      else KaldiRecognizer(self._vosk_model, rate))
        recognizer.AcceptWaveform(audio.get_raw_data(convert_rate=rate, convert_width=2))
        text = json.loads(recognizer.FinalResult()).get("text", "")
        return " ".join(w for w in text.split() if w != "[unk]")
