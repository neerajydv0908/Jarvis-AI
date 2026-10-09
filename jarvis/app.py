"""The main event loops (roadmap step 7).

Voice mode::

    while True:
        speak any reminders that came due
        1. wait for the wake word          (short listens, so reminders are never delayed long)
        2. listen and transcribe the command
        3-4. NLP -> intent -> action module   (Assistant.handle)
        5. speak the reply                  (the microphone is closed while speaking)
        6. keep listening a few seconds for a follow-up, then go back to waiting

Reminders are checked by a background thread (ReminderScheduler) and handed
to this loop through a queue, so only one thread ever uses the speaker and
the microphone at a time.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from typing import Optional

from .assistant import Assistant, Reply
from .config import Settings
from .skills.reminders import ReminderScheduler
from .speech.listener import Listener, ListenerError, WakeWordDetector
from .speech.speaker import Speaker

log = logging.getLogger(__name__)

ACKNOWLEDGEMENTS = ["Yes{address}?", "At your service{address}.", "How can I help?", "Listening."]


def _print_reply(reply: Reply, debug: bool) -> None:
    suffix = f"   [{reply.intent} {reply.confidence:.2f}]" if debug and reply.intent else ""
    print(f"JARVIS: {reply.text}{suffix}")


class TextApp:
    """Keyboard chat with the same brain; optional speech output. Great for testing."""

    def __init__(self, assistant: Assistant, speaker: Optional[Speaker] = None, debug: bool = False):
        self.assistant = assistant
        self.speaker = speaker
        self.debug = debug
        self._stop = threading.Event()

    def _notify_loop(self) -> None:
        while not self._stop.is_set():
            try:
                message = self.assistant.notifications.get(timeout=0.5)
            except queue.Empty:
                continue
            print(f"\nJARVIS: {message}\nYou: ", end="", flush=True)
            if self.speaker:
                self.speaker.say(message)

    def run(self) -> None:
        notifier = threading.Thread(target=self._notify_loop, name="notifier", daemon=True)
        notifier.start()
        greeting = self.assistant.greeting()
        print(f"JARVIS: {greeting}  (type 'exit' to quit)")
        if self.speaker:
            self.speaker.say(greeting)
        try:
            while True:
                try:
                    text = input("You: ").strip()
                except EOFError:
                    break
                if not text:
                    continue
                reply = self.assistant.handle(text)
                _print_reply(reply, self.debug)
                if self.speaker:
                    self.speaker.say(reply.text)
                if reply.shutdown:
                    break
        except KeyboardInterrupt:
            print()
        finally:
            self._stop.set()


class VoiceApp:
    def __init__(self, assistant: Assistant, listener: Listener, speaker: Speaker, settings: Settings,
                 debug: bool = False, rng: Optional[random.Random] = None, error_backoff: float = 1.0):
        self.assistant = assistant
        self.listener = listener
        self.speaker = speaker
        self.settings = settings
        self.debug = debug
        self.wake = WakeWordDetector(settings.wake_word)
        self._rng = rng or random.Random()
        self._speech_errors = 0
        self.error_backoff = error_backoff  # seconds; grows with repeated failures (no busy loop)

    # -- output --------------------------------------------------------------------------
    def say(self, text: str) -> None:
        print(f"JARVIS: {text}")
        self.speaker.say(text)  # blocks until finished; the microphone is closed meanwhile

    def announce_reminders(self) -> None:
        while True:
            try:
                message = self.assistant.notifications.get_nowait()
            except queue.Empty:
                return
            self.say(message)

    # -- input ----------------------------------------------------------------------------
    def _safe(self, func, *args) -> Optional[str]:
        try:
            text = func(*args)
        except ListenerError as exc:
            self._speech_errors += 1
            print(f"  [speech problem] {exc}")
            if self._speech_errors in (1, 20):  # don't nag on every failed attempt
                self.say("I'm having trouble with speech recognition. The details are in the terminal.")
            time.sleep(min(10.0, self.error_backoff * self._speech_errors))
            return None
        if text:  # only a real transcript proves the problem is gone
            self._speech_errors = 0
        return text

    def wait_for_wake_word(self) -> Optional[str]:
        """Returns the command spoken with the wake word ('' if none), or None if not woken."""
        heard = self._safe(self.listener.listen_for_wake_word, self.settings.wake_listen_timeout,
                           self.settings.wake_phrase_limit)
        if not heard:
            return None
        if self.debug:
            print(f"  (heard: {heard!r})")
        woken, command = self.wake.detect(heard)
        return command if woken else None

    def listen(self, timeout: float) -> Optional[str]:
        print("  (listening...)")
        text = self._safe(self.listener.listen, timeout, self.settings.command_phrase_limit)
        if text:
            print(f"You: {text}")
        return text

    # -- loop -------------------------------------------------------------------------------
    def converse(self, command: str) -> bool:
        """Handle one command plus follow-ups. Returns False when Jarvis should shut down."""
        try:
            return self._converse(command)
        finally:
            self.assistant.cancel_pending()  # an unanswered question must not capture a later command

    def _converse(self, command: str) -> bool:
        if command:  # spoken in the same breath as the wake word: "Jarvis, what time is it?"
            print(f"You: {command}")
        else:
            self.say(self._rng.choice(ACKNOWLEDGEMENTS).format(address=self.settings.address))
            command = self.listen(self.settings.command_timeout)
            if not command:
                return True
        while True:
            reply = self.assistant.handle(command)
            if self.debug:
                print(f"  [{reply.intent} {reply.confidence:.2f}]")
            self.say(reply.text)
            if reply.shutdown:
                return False
            self.announce_reminders()
            if self.assistant.awaiting_answer:  # Jarvis asked a question: always wait for the answer
                command = self.listen(max(self.settings.follow_up_seconds, self.settings.command_timeout))
                if not command:
                    return True
                continue
            if reply.end_conversation or self.settings.follow_up_seconds <= 0:
                return True
            command = self.listen(self.settings.follow_up_seconds)  # follow-up without the wake word
            if not command:
                return True

    def run(self) -> None:
        print(f"Calibrating for background noise ({self.settings.calibration_seconds:.0f}s) - please be quiet...")
        try:
            self.listener.calibrate(self.settings.calibration_seconds)
        except Exception as exc:  # pragma: no cover - hardware problems
            log.warning("Calibration failed: %s", exc)
        self.say(self.assistant.greeting())
        print(f"Say '{self.settings.wake_word.title()}' to wake me up. Ctrl+C to quit.")
        try:
            while True:
                self.announce_reminders()
                command = self.wait_for_wake_word()
                if command is None:
                    continue
                if not self.converse(command):
                    break
                print(f"(waiting for '{self.settings.wake_word.title()}')")
            self.announce_reminders()  # never drop a reminder that came due during the last exchange
        except KeyboardInterrupt:
            print()
            self.announce_reminders()
            self.say(f"Goodbye{self.settings.address}.")


def start_scheduler(assistant: Assistant) -> ReminderScheduler:
    scheduler = ReminderScheduler(assistant.reminders, assistant.reminder_due)
    scheduler.start()
    return scheduler
