"""The assistant itself: command text in, reply text out.

This is steps 3-4 of the main loop (NLP -> intent -> action module). It owns no
audio, so the voice loop, the text loop and the tests all drive it the same way.
Multi-turn follow-ups ("Which city?", "What time?", "Are you sure?") are handled
with a single pending callback that gets the next utterance first.
"""

from __future__ import annotations

import logging
import queue
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .config import Settings
from .nlp.entities import EntityExtractor, ReminderTask, normalize_city, second_person
from .nlp.preprocess import normalize_wake_word, strip_wake_word
from .nlp.timeparse import describe_when, format_clock, parse_time
from .skills.clock import date_answer, part_of_day, time_answer
from .skills.jokes import JokeTeller
from .skills.knowledge import KnowledgeError
from .skills.reminders import Reminder, ReminderStore
from .skills.weather import WeatherError

log = logging.getLogger(__name__)

_YES_RE = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok(?:ay)?|do it|go ahead|please do|confirm|absolutely|"
                     r"correct|definitely|of course|affirmative)\b", re.I)
_NO_RE = re.compile(r"^(?:no|nope|nah|don't|do not|cancel|stop|never ?mind|keep (?:them|it)|negative)\b", re.I)
# "never mind", "cancel that", "nah, forget it", "no thanks" ... (but not "stop by the store")
_CANCEL_RE = re.compile(r"^(?:no|nope|nah|cancel|never ?mind|forget (?:it|that|about it)|skip (?:it|that)|abort)\b"
                        r"|^(?:stop|nothing|don't|do not)(?: it| that)?(?: thanks| thank you)?$", re.I)
# words that can surround a time answer without changing it ("um, make it 5 pm please")
_FILLER = {"at", "on", "by", "around", "um", "uh", "ok", "okay", "please", "maybe", "make", "it", "how", "about",
           "lets", "let's", "say", "then", "so", "oh", "hmm", "well", "just", "yeah", "yes", "sure", "remind", "me",
           "and", "the", "for", "that", "would", "be", "great", "good", "thanks", "thank", "you"}
PENDING_TTL = timedelta(minutes=2)  # an unanswered question expires

FollowUp = Callable[[str], Optional["Reply"]]


@dataclass
class Reply:
    text: str
    intent: str = ""
    confidence: float = 0.0
    end_conversation: bool = False   # go back to waiting for the wake word
    shutdown: bool = False           # exit the program


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class Assistant:
    def __init__(self, settings: Settings, classifier, extractor: EntityExtractor, chat, weather,
                 reminders: ReminderStore, knowledge, jokes: Optional[JokeTeller] = None,
                 responses: Optional[Dict[str, List[str]]] = None, clock: Callable[[], datetime] = datetime.now,
                 seed: Optional[int] = None):
        self.settings = settings
        self.classifier = classifier
        self.extractor = extractor
        self.chat = chat
        self.weather = weather
        self.reminders = reminders
        self.knowledge = knowledge
        self.jokes = jokes or JokeTeller(seed=seed)
        self.responses = responses or {}
        self.clock = clock
        self._rng = random.Random(seed)
        self.notifications: "queue.Queue[str]" = queue.Queue()
        self.history: List[Tuple[str, str]] = []
        self._pending: Optional[FollowUp] = None
        self._pending_since: Optional[datetime] = None
        self._handlers: Dict[str, Callable[[str], Reply]] = {
            "greeting": self._greeting,
            "goodbye": self._goodbye,
            "shutdown": self._shutdown,
            "thanks": partial(self._canned, "thanks"),
            "identity": partial(self._canned, "identity"),
            "capabilities": partial(self._canned, "capabilities"),
            "how_are_you": partial(self._canned, "how_are_you"),
            "weather": self._weather,
            "set_reminder": self._set_reminder,
            "list_reminders": self._list_reminders,
            "clear_reminders": self._clear_reminders,
            "time": lambda _: Reply(time_answer(self.clock())),
            "date": lambda command: Reply(date_answer(self.clock(), command)),
            "joke": lambda _: Reply(self.jokes.tell()),
            "knowledge": self._knowledge,
            "general_chat": self._chat,
        }

    # ------------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------------
    @classmethod
    def build(cls, settings: Settings, train_if_missing: bool = True) -> "Assistant":
        """Create every component from the settings (trains the intent model on first run)."""
        from .brain.chat import ChatEngine
        from .brain.classifier import IntentClassifier, ModelNotTrainedError
        from .brain.dataset import load_intents, responses_by_tag
        from .nlp.preprocess import TextPreprocessor
        from .skills.knowledge import Wikipedia
        from .skills.weather import WeatherService

        preprocessor = TextPreprocessor()
        try:
            classifier = IntentClassifier(settings.model_dir, settings.intent_backend, settings.torch_device,
                                          preprocessor)
        except ModelNotTrainedError:
            if not train_if_missing:
                raise
            from .brain.train import train

            print("First run: training the intent classifier (takes a few seconds)...")
            train(backend=settings.intent_backend, model_dir=settings.model_dir, device=settings.torch_device,
                  verbose=False, preprocessor=preprocessor)
            classifier = IntentClassifier(settings.model_dir, settings.intent_backend, settings.torch_device,
                                          preprocessor)
        responses = responses_by_tag(load_intents())
        return cls(
            settings=settings,
            classifier=classifier,
            extractor=EntityExtractor(settings.spacy_model),
            chat=ChatEngine(settings.chat_model_dir, settings.torch_device, settings.chat_enabled,
                            responses.get("general_chat")),
            weather=WeatherService(settings.openweather_api_key, settings.units),
            reminders=ReminderStore(settings.db_path),
            knowledge=Wikipedia(),
            responses=responses,
        )

    # ------------------------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------------------------
    def handle(self, text: str) -> Reply:
        command = strip_wake_word(text, self.settings.wake_word).strip(" ,.!?")
        if not command:
            return Reply(f"Yes{self.settings.address}?", intent="wake")

        if self._pending is not None:  # answer to a question Jarvis just asked?
            follow_up, asked_at = self._pending, self._pending_since
            self.cancel_pending()
            if asked_at is not None and self.clock() - asked_at <= PENDING_TTL:
                try:
                    reply = follow_up(command)
                except Exception:
                    log.exception("Follow-up handler failed")
                    reply = Reply("Sorry, something went wrong while I was working on that.")
                if reply is not None:
                    return self._remember(command, reply)

        prediction = self.classifier.predict(normalize_wake_word(command, self.settings.wake_word))
        intent = prediction.intent
        if prediction.known_tokens == 0 or prediction.confidence < self.settings.confidence_threshold:
            intent = "general_chat"
        log.debug("intent=%s confidence=%.2f tokens=%s ranking=%s", prediction.intent, prediction.confidence,
                  prediction.tokens, prediction.ranking)
        try:
            reply = self._handler_for(intent)(command)
        except Exception:
            log.exception("The %s handler failed", intent)
            reply = Reply("Sorry, something went wrong while I was working on that.")
        reply.intent = reply.intent or intent
        reply.confidence = prediction.confidence
        return self._remember(command, reply)

    @property
    def awaiting_answer(self) -> bool:
        """True right after Jarvis asked a question ("Which city should I check?")."""
        return self._pending is not None

    def cancel_pending(self) -> None:
        """Forget any question Jarvis is waiting on (called when a voice conversation ends)."""
        self._pending = None
        self._pending_since = None

    def _ask(self, follow_up: FollowUp, question: str, intent: str = "") -> Reply:
        self._pending = follow_up
        self._pending_since = self.clock()
        return Reply(question, intent=intent)

    def greeting(self) -> str:
        return f"Good {part_of_day(self.clock())}{self.settings.address}. {self.settings.name} is online."

    def reminder_due(self, reminder: Reminder) -> None:
        """Scheduler callback (runs on the scheduler thread): queue the announcement."""
        self.notifications.put(self.announcement(reminder))

    def announcement(self, reminder: Reminder) -> str:
        task = second_person(reminder.task)
        address = self.settings.address
        if reminder.connector == "about":
            text = f"Excuse me{address}, here's your reminder about {task}."
        elif reminder.connector == "that":
            text = f"Excuse me{address}, a reminder that {task}."
        else:
            text = f"Excuse me{address}, it's time to {task}."
        now = self.clock()
        if now - reminder.due > timedelta(minutes=2):  # e.g. it came due while Jarvis was off
            if reminder.due.date() == now.date():
                day = ""
            elif reminder.due.date() == now.date() - timedelta(days=1):
                day = " yesterday"
            else:
                day = f" on {reminder.due.strftime('%A')}"
            text += f" It was due at {format_clock(reminder.due)}{day}."
        return text

    def close(self) -> None:
        self.reminders.close()

    # ------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------
    def _remember(self, command: str, reply: Reply) -> Reply:
        self.history = (self.history + [(command, reply.text)])[-10:]
        return reply

    def _format(self, template: str) -> str:
        creator_line = (f"I was built by {self.settings.creator}." if self.settings.creator
                        else "I was built as a personal project in Python.")
        values = {"address": self.settings.address, "name": self.settings.name, "creator_line": creator_line,
                  "creator": self.settings.creator or "my developer"}
        try:
            return template.format(**values).strip()
        except (KeyError, IndexError, ValueError):
            return template

    def _handler_for(self, intent: str) -> Callable[[str], Reply]:
        """Code handler if there is one; otherwise the intent's responses from intents.json.

        So a simple question-and-answer command needs no Python at all: add the intent
        with patterns and responses to intents.json and retrain.
        """
        if intent in self._handlers:
            return self._handlers[intent]
        if self.responses.get(intent):
            return partial(self._canned, intent)
        return self._chat

    def _canned(self, intent: str, _command: str = "") -> Reply:
        options = self.responses.get(intent) or ["Okay."]
        return Reply(self._format(self._rng.choice(options)))

    def _looks_like_new_command(self, text: str, allowed: Set[str]) -> bool:
        """True when a follow-up answer is clearly a different request ("what time is it").

        Deliberately strict: answers to "what should I remind you about?" are free text
        ("submit the lab report") and must not be mistaken for commands because one word
        ("report") also appears in another intent.
        """
        prediction = self.classifier.predict(normalize_wake_word(text, self.settings.wake_word))
        if prediction.intent in allowed | {"general_chat"} or prediction.confidence < 0.85:
            return False
        if prediction.known_tokens >= 2:
            return True
        # one-word control commands still count: "exit", "thanks", "goodbye"
        return prediction.known_tokens == 1 and prediction.intent in {"shutdown", "goodbye", "thanks"}

    @staticmethod
    def _is_cancel(text: str) -> bool:
        return bool(_CANCEL_RE.match(re.sub(r"[^\w\s']", "", text).strip()))

    # ------------------------------------------------------------------------------------
    # intent handlers
    # ------------------------------------------------------------------------------------
    def _greeting(self, command: str) -> Reply:
        if re.search(r"\bgood (?:morning|afternoon|evening)\b", command, re.I):
            return Reply(f"Good {part_of_day(self.clock())}{self.settings.address}. How can I help?")
        return self._canned("greeting")

    def _goodbye(self, _command: str) -> Reply:
        reply = self._canned("goodbye")
        reply.end_conversation = True
        return reply

    def _shutdown(self, _command: str) -> Reply:
        reply = self._canned("shutdown")
        reply.end_conversation = reply.shutdown = True
        return reply

    def _chat(self, command: str) -> Reply:
        return Reply(self.chat.reply(command, self.history[-3:]), intent="general_chat")

    # -- weather -------------------------------------------------------------------------------
    def _weather(self, command: str) -> Reply:
        if not getattr(self.weather, "api_key", True):
            return Reply("I need an OpenWeatherMap API key for weather updates. "
                         "Add OPENWEATHER_API_KEY to your .env file.")
        city = self.extractor.extract_city(command) or self.settings.default_city
        if not city:
            return self._ask(partial(self._weather_city_answer, command), "Which city should I check?", "weather")
        return self._weather_for(city, command)

    def _weather_city_answer(self, original: str, answer: str) -> Optional[Reply]:
        if self._is_cancel(answer):
            return Reply("Okay, never mind.", intent="weather")
        if self._looks_like_new_command(answer, allowed={"weather", "knowledge"}):
            return None
        city = self.extractor.extract_city(answer) or normalize_city(answer)
        return self._weather_for(city, original)

    def _weather_for(self, city: str, command: str) -> Reply:
        try:
            if re.search(r"\btomorrow\b", command, re.I):
                return Reply(self.weather.forecast(city, days_ahead=1).speak(), intent="weather")
            return Reply(self.weather.current(city).speak(), intent="weather")
        except WeatherError as exc:
            return Reply(str(exc), intent="weather")

    # -- reminders -------------------------------------------------------------------------------
    def _set_reminder(self, command: str) -> Reply:
        now = self.clock()
        parsed = parse_time(command, now)
        rest = parsed.strip_from(command) if parsed else command
        task = self.extractor.extract_reminder_task(rest)
        if parsed and task:
            return self._create_reminder(task, parsed.when)
        if task:
            return self._ask(partial(self._reminder_time_answer, task),
                             f"When should I remind you {task.spoken()}?", "set_reminder")
        if parsed:
            return self._ask(partial(self._reminder_task_answer, parsed.when),
                             f"What should I remind you about {describe_when(parsed.when, now)}?", "set_reminder")
        return self._ask(self._reminder_task_then_time, "Sure. What should I remind you about?", "set_reminder")

    def _create_reminder(self, task: ReminderTask, when: datetime) -> Reply:
        self.reminders.add(task.task, when, task.connector)
        return Reply(f"Okay, I'll remind you {task.spoken()} {describe_when(when, self.clock())}.",
                     intent="set_reminder")

    def _reminder_time_answer(self, task: ReminderTask, answer: str) -> Optional[Reply]:
        if self._is_cancel(answer):
            return Reply("Okay, no reminder.", intent="set_reminder")
        now = self.clock()
        parsed = parse_time(answer, now)
        if parsed is None and re.fullmatch(r"\d{1,2}(?::\d{2})?", answer.strip()):
            parsed = parse_time("at " + answer.strip(), now)  # a bare "5" means 5 o'clock
        if parsed:
            leftover = re.findall(r"[\w']+", parsed.strip_from(answer).lower())
            if all(word in _FILLER for word in leftover):
                return self._create_reminder(task, parsed.when)
            return None  # "what's the weather tonight": a new request that happens to contain a time
        if self._looks_like_new_command(answer, allowed={"set_reminder"}):
            return None
        return Reply("Sorry, I didn't catch a time, so I haven't set that reminder.", intent="set_reminder")

    def _reminder_task_answer(self, when: datetime, answer: str) -> Optional[Reply]:
        if self._is_cancel(answer):
            return Reply("Okay, no reminder.", intent="set_reminder")
        if self._looks_like_new_command(answer, allowed={"set_reminder"}):
            return None
        task = self.extractor.extract_reminder_task(self._as_reminder_request(answer))
        if task is None:
            return Reply("Sorry, I didn't catch that, so I haven't set a reminder.", intent="set_reminder")
        return self._create_reminder(task, when)

    def _reminder_task_then_time(self, answer: str) -> Optional[Reply]:
        if self._is_cancel(answer):
            return Reply("Okay, no reminder.", intent="set_reminder")
        if self._looks_like_new_command(answer, allowed={"set_reminder"}):
            return None
        return self._set_reminder(self._as_reminder_request(answer))

    def _as_reminder_request(self, answer: str) -> str:
        answer = answer.strip()
        if re.match(r"^(?:remind|set|add|create)\b", answer, re.I):
            return answer
        if re.match(r"^(?:to|about|that|of)\b", answer, re.I):
            return f"remind me {answer}"
        # "call mom" -> "remind me to call mom"; "the team meeting" -> "remind me about the team meeting"
        connector = "to" if self.extractor.starts_with_verb(answer) else "about"
        return f"remind me {connector} {answer}"

    def _list_reminders(self, _command: str) -> Reply:
        pending = self.reminders.pending()
        if not pending:
            return Reply("You don't have any reminders.")
        now = self.clock()
        items = [f"{ReminderTask(r.task, r.connector).spoken()} {describe_when(r.due, now)}" for r in pending[:5]]
        intro = "You have one reminder: " if len(pending) == 1 else f"You have {len(pending)} reminders: "
        more = f", and {len(pending) - 5} more" if len(pending) > 5 else ""
        return Reply(intro + _join(items) + more + ".")

    def _clear_reminders(self, _command: str) -> Reply:
        count = len(self.reminders.pending())
        if count == 0:
            return Reply("You don't have any reminders to delete.")
        noun = "reminder" if count == 1 else "reminders"
        return self._ask(self._confirm_clear, f"That will delete {count} {noun}. Should I go ahead?")

    def _confirm_clear(self, answer: str) -> Optional[Reply]:
        if _YES_RE.match(answer.strip()):
            deleted = self.reminders.clear_pending()
            return Reply(f"Done. I deleted {deleted} reminder{'s' if deleted != 1 else ''}.", intent="clear_reminders")
        if _NO_RE.match(answer.strip()):
            return Reply("Okay, I'll keep them.", intent="clear_reminders")
        return None  # something else entirely: treat it as a new command

    # -- questions ----------------------------------------------------------------------------------
    def _knowledge(self, command: str) -> Reply:
        topic = self.extractor.extract_topic(command)
        try:
            return Reply(self.knowledge.answer(topic))
        except KnowledgeError as exc:
            return Reply(str(exc))
