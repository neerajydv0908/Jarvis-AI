"""Wake word, speaker and the voice main loop - with fake audio devices."""

import threading
import time
from collections import deque

import pytest

from jarvis.app import VoiceApp
from jarvis.config import Settings
from jarvis.speech.listener import ListenerError, WakeWordDetector
from jarvis.speech.speaker import PrintEngine, SpeechEngine, Speaker, build_engine


# -- wake word -------------------------------------------------------------------------------------
@pytest.mark.parametrize("transcript, expected", [
    ("jarvis what time is it", (True, "what time is it")),
    ("Jarvis, what's the weather", (True, "what's the weather")),
    ("what time is it jarvis", (True, "what time is it")),
    ("hey jarvis", (True, "")),
    ("jervis are you there", (True, "are you there")),   # near-miss from the recogniser
    ("Jarvis's turn", (True, "turn")),
    ("I need some customer service", (False, "")),
    ("let's play some java", (False, "")),
    ("", (False, "")),
])
def test_wake_word(transcript, expected):
    assert WakeWordDetector("jarvis").detect(transcript) == expected


# -- speaker ---------------------------------------------------------------------------------------------
class RecordingEngine(SpeechEngine):
    name = "recording"

    def __init__(self, delay=0.0):
        self.spoken = []
        self.threads = set()
        self.delay = delay

    def speak(self, text):
        self.threads.add(threading.get_ident())
        time.sleep(self.delay)
        self.spoken.append(text)


def test_speaker_uses_one_thread_and_blocks():
    engine = RecordingEngine(delay=0.05)
    speaker = Speaker(engine_instance=engine)
    workers = [threading.Thread(target=speaker.say, args=(f"line {i}",)) for i in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert sorted(engine.spoken) == [f"line {i}" for i in range(4)]
    assert len(engine.threads) == 1          # pyttsx3-safe: always the same thread
    assert not speaker.is_speaking
    speaker.close()


def test_is_speaking_while_talking():
    engine = RecordingEngine(delay=0.3)
    speaker = Speaker(engine_instance=engine)
    speaker.say("hello", block=False)
    time.sleep(0.05)
    assert speaker.is_speaking
    speaker.close()


def test_stuck_engine_does_not_hang_the_assistant():
    class HangingEngine(SpeechEngine):
        name = "hanging"

        def speak(self, text):
            threading.Event().wait(5)

    speaker = Speaker(engine_instance=HangingEngine())
    speaker.timeout_base = 0.1
    started = time.time()
    speaker.say("hi")
    speaker.say("are you there?")  # now text-only, returns at once
    assert time.time() - started < 2


@pytest.mark.parametrize("requested, available, expected", [
    ("Daniel", ["Alex", "Daniel", "Samantha"], "Daniel"),
    ("daniel", ["Daniel"], "Daniel"),
    ("Daniel", ["Daniel (Enhanced)", "Samantha"], "Daniel (Enhanced)"),
    ("Daniel", ["Daniel (Enhanced)", "Daniel (Premium)"], "Daniel (Premium)"),
    ("Daniel", ["Samantha", "Danielle"], None),       # not a variant of Daniel
    ("", ["Daniel"], None),                           # empty = Mac's default voice
    ("system", ["Daniel"], None),
])
def test_pick_voice(requested, available, expected):
    from jarvis.speech.speaker import pick_voice

    assert pick_voice(requested, available) == expected


def test_unknown_engine_falls_back_to_print():
    assert isinstance(build_engine("does-not-exist"), PrintEngine)


# -- voice loop with a scripted microphone --------------------------------------------------------------
class FakeListener:
    def __init__(self, wake_phrases, commands):
        self.wake_phrases = deque(wake_phrases)
        self.commands = deque(commands)
        self.calibrated = False

    def calibrate(self, seconds):
        self.calibrated = True

    def listen_for_wake_word(self, timeout, phrase_time_limit):
        if not self.wake_phrases:
            raise KeyboardInterrupt  # end of script
        phrase = self.wake_phrases.popleft()
        if isinstance(phrase, Exception):
            raise phrase
        return phrase

    def listen(self, timeout, phrase_time_limit):
        return self.commands.popleft() if self.commands else None


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text, block=True):
        self.said.append(text)


def _voice_app(assistant, wake_phrases, commands, follow_up=5):
    settings = Settings(follow_up_seconds=follow_up, calibration_seconds=0)
    listener, speaker = FakeListener(wake_phrases, commands), FakeSpeaker()
    app = VoiceApp(assistant, listener, speaker, settings, error_backoff=0)
    return app, listener, speaker


def test_voice_loop_full_session(make_assistant):
    assistant = make_assistant()
    app, listener, speaker = _voice_app(
        assistant,
        wake_phrases=[None, "some background chatter", "jarvis what time is it", "hey jarvis"],
        commands=[None,                       # no follow-up after the time -> back to sleep
                  "tell me a joke", "thanks", None])
    app.run()
    assert listener.calibrated
    assert speaker.said[0] == "Good afternoon, sir. Jarvis is online."
    assert "It's 2 PM." in speaker.said
    assert speaker.said[-1] == "Goodbye, sir."  # Ctrl+C
    joke_index = speaker.said.index("It's 2 PM.") + 2   # acknowledgement, then the joke
    assert speaker.said[joke_index] in assistant.jokes._jokes


def test_voice_loop_shutdown_command(make_assistant):
    app, _, speaker = _voice_app(make_assistant(), ["jarvis shut down", "jarvis never reached"], [])
    app.run()
    assert speaker.said[-1].startswith(("Shutting down", "Powering down", "Going offline"))


def test_voice_loop_announces_reminders_between_listens(make_assistant):
    assistant = make_assistant()
    assistant.notifications.put("Excuse me, sir, it's time to stretch.")
    app, _, speaker = _voice_app(assistant, [None], [])
    app.run()
    assert "Excuse me, sir, it's time to stretch." in speaker.said


def test_voice_loop_reports_listener_errors_once(make_assistant):
    error = ListenerError("Speech service unavailable")
    app, _, speaker = _voice_app(make_assistant(), [error, error, error], [])
    app.run()
    complaints = [s for s in speaker.said if "trouble with speech recognition" in s]
    assert len(complaints) == 1


def test_voice_loop_error_throttle_survives_silence(make_assistant):
    error = ListenerError("offline")
    app, _, speaker = _voice_app(make_assistant(), [error, None, error, None, error], [])
    app.run()
    assert len([s for s in speaker.said if "trouble with speech recognition" in s]) == 1


def test_unanswered_question_expires_when_conversation_ends(make_assistant):
    assistant = make_assistant()
    app, _, speaker = _voice_app(assistant, ["jarvis remind me to call mom",
                                             "jarvis will it rain tomorrow in chicago"], [None, None])
    app.run()
    assert "When should I remind you to call mom?" in speaker.said
    assert "Tomorrow's forecast for Chicago." in speaker.said
    assert assistant.reminders.pending() == []


def test_questions_get_an_answer_window_even_without_follow_ups(make_assistant):
    app, _, speaker = _voice_app(make_assistant(), ["jarvis what's the weather"], ["denver"], follow_up=0)
    app.run()
    assert "Which city should I check?" in speaker.said
    assert "Current weather in Denver." in speaker.said


def test_queued_reminder_is_spoken_before_shutdown(make_assistant):
    assistant = make_assistant()
    app, _, speaker = _voice_app(assistant, ["jarvis shut down"], [])
    original = assistant.handle

    def handle_and_fire(text):
        assistant.notifications.put("Excuse me, sir, it's time to stretch.")  # comes due mid-exchange
        return original(text)

    assistant.handle = handle_and_fire
    app.run()
    assert "Excuse me, sir, it's time to stretch." in speaker.said


# -- listener error handling (no audio hardware needed) ------------------------------------------------
def _bare_listener():
    import speech_recognition as sr

    from jarvis.speech.listener import Listener

    listener = Listener.__new__(Listener)
    listener.sr = sr
    listener.recognizer = sr.Recognizer()
    listener.language = "en-US"
    listener._vosk_model = None
    return listener


def test_microphone_that_fails_to_open_raises_listener_error():
    class BrokenMic:
        stream = None

        def __enter__(self):
            return self  # SpeechRecognition hides PyAudio's error and leaves stream=None

        def __exit__(self, *exc):
            raise AttributeError("'NoneType' object has no attribute 'close'")

    listener = _bare_listener()
    listener.microphone = BrokenMic()
    with pytest.raises(ListenerError, match="microphone"):
        listener.capture(1, 1)


@pytest.mark.parametrize("error, message", [
    (TimeoutError("timed out"), "internet"),
    (ConnectionResetError("reset"), "internet"),
    (OSError(86, "Bad CPU type in executable"), "brew install flac"),
])
def test_google_errors_are_explained(error, message):
    listener = _bare_listener()

    def boom(*args, **kwargs):
        raise error

    listener.recognizer.recognize_google = boom
    with pytest.raises(ListenerError, match=message):
        listener.transcribe(object(), "google")
