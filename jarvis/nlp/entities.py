"""Entity extraction with spaCy (roadmap step 3).

* Named Entity Recognition finds places (GPE / LOC / FAC) for weather requests.
* Dependency parsing finds *what* to remind you about: the clause hanging off
  "remind" ("remind me [to call mom]"), the clause attached to the noun
  "reminder" ("set a reminder [to water the plants]") or the object of
  "about/of/for" ("remind me about [the team meeting]").

Speech transcripts are often lower-case, which hurts NER, so every extractor
also has a rule-based fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
_STATE_CODES = set(US_STATES.values())

# Words that end a place name ("in Chicago this weekend") or are not places at all.
_TIME_WORDS = (
    r"today|tomorrow|tonight|now|right\s+now|currently|later|this\s+\w+|next\s+\w+|on\s+\w+day"
    r"|at\s+the\s+moment|please|outside|like|in\s+the\s+\w+"
)
_NOT_PLACES = {
    "the morning", "morning", "afternoon", "evening", "night", "the weekend", "weekend", "week",
    "today", "tomorrow", "tonight", "here", "there", "outside", "celsius", "fahrenheit", "general",
    "my area", "my city", "my location", "this area", "town", "the city", "the area", "the day",
    "home", "the house", "the moment",
}
_TIME_ONLY_RE = re.compile(
    r"^(?:the\s+|this\s+|next\s+|the\s+next\s+|the\s+rest\s+of\s+the\s+|the\s+coming\s+|coming\s+|a\s+few\s+|few\s+)?"
    r"(?:\w+\s+)?(?:morning|afternoon|evening|night|weekend|week|day|days|hours?|moment|now|today|tomorrow|tonight)$")
_NOT_PLACE_WORDS = {"me", "us", "here", "there", "home", "outside", "inside", "the", "a", "an", "it", "general"}
_PLACE_PREPOSITION_RE = re.compile(r"\b(?:in|at|for|near|around)\s+", re.I)
_PLACE_STOP_RE = re.compile(rf"\s+(?:in|at|for|on|near|around|and|with|{_TIME_WORDS})\b.*$", re.I)

_REMINDER_ANCHORS = {"remind", "reminder", "reminders", "forget", "remember", "ping", "alert", "nudge"}
_CLAUSE_DEPS = {"xcomp", "ccomp", "acl", "relcl", "advcl"}
_ABOUT_PREPS = {"about", "of", "for", "regarding", "on"}
_REMINDER_REGEX = re.compile(
    r"\b(?:remind(?:er)?(?:\s+me)?|remember|forget|ping\s+me)\s+(?P<connector>to|that|about|of|for)\s+(?P<task>.+)$",
    re.I,
)
_LEADING_FILLER_RE = re.compile(r"^(?:(?:hey|ok|okay|please|so|um|uh|jarvis)[\s,]+)+", re.I)

# "who is X", "tell me about X", "what does X mean" ...
_TOPIC_PREFIX_RE = re.compile(
    r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:tell\s+me\s+|do\s+you\s+know\s+|i\s+want\s+to\s+know\s+|i'd\s+like\s+to\s+know\s+)?"
    r"(?:(?:who|what)\s+(?:is|are|was|were)\s+|(?:what|who)'s\s+|whats\s+"
    r"|(?:something\s+|more\s+|everything\s+)?about\s+|(?:the\s+)?(?:definition|meaning)\s+of\s+"
    r"|define\s+|explain\s+|search\s+(?:wikipedia\s+|the\s+web\s+)?for\s+|look\s+up\s+|google\s+"
    r"|(?:some\s+)?(?:facts|info|information|details)\s+(?:about|on)\s+|wikipedia\s+)",
    re.I,
)
_TOPIC_SUFFIX_RE = re.compile(r"\s+(?:on\s+wikipedia|for\s+me|please|mean)$", re.I)


def second_person(text: str) -> str:
    """Rewrite the user's words for Jarvis to say back: 'call my mom' -> 'call your mom'."""
    swaps = {"my": "your", "me": "you", "mine": "yours", "myself": "yourself", "i": "you",
             "i'm": "you're", "i've": "you've", "i'll": "you'll", "am": "are", "our": "your", "us": "you"}

    def repl(m: re.Match) -> str:
        word = m.group(0)
        new = swaps[word.lower()]
        return new.capitalize() if word[0].isupper() and word.lower() != "i" else new

    return re.sub(r"\b(?:my|me|mine|myself|i'm|i've|i'll|i|our|us)\b|(?<=\bi )am\b", repl, text, flags=re.I)


@dataclass
class ReminderTask:
    task: str
    connector: str = "to"   # "to" call mom | "about" the meeting | "that" the oven is on

    def spoken(self) -> str:
        return f"{self.connector} {second_person(self.task)}"


class EntityExtractor:
    """Wraps a spaCy pipeline; loads it once and reuses it."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("spaCy is not installed: pip install spacy") from exc
        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}"
            ) from exc

    # -- generic -----------------------------------------------------------------
    def entities(self, text: str) -> List[Tuple[str, str]]:
        return [(ent.text, ent.label_) for ent in self.nlp(text).ents]

    # -- weather: which city? ----------------------------------------------------------
    def extract_city(self, text: str) -> Optional[str]:
        """'what's the weather in lawrence kansas tomorrow' -> 'Lawrence,KS,US'."""
        candidate = self._city_after_preposition(text) or self._city_from_ner(text)
        return normalize_city(candidate) if candidate else None

    def _city_after_preposition(self, text: str) -> Optional[str]:
        cleaned = re.sub(r"[?!.]+$", "", text.strip())
        for m in _PLACE_PREPOSITION_RE.finditer(cleaned):
            tail = re.split(r"[?!;]", cleaned[m.end():])[0]
            if "," in tail:  # keep "Lawrence, Kansas" / "Lawrence, KS"; drop other trailing clauses
                head, _, after = tail.partition(",")
                after = _PLACE_STOP_RE.sub("", " " + after).strip()
                is_state = after.lower() in US_STATES or (len(after) == 2 and after.upper() in _STATE_CODES)
                tail = f"{head}, {after}" if is_state else head
            tail = _PLACE_STOP_RE.sub("", tail).strip()
            tail = re.sub(r"^(?:the\s+)?(?:city|town)\s+of\s+", "", tail, flags=re.I).strip()
            if _is_not_place(tail):
                continue
            if len(tail.split()) > 5:  # a whole sentence, not a place name
                continue
            return tail
        return None

    def _city_from_ner(self, text: str) -> Optional[str]:
        doc = self.nlp(text)
        places = [ent for ent in doc.ents if ent.label_ in {"GPE", "LOC", "FAC"}]
        if not places:
            return None
        # merge neighbours: "Kansas City" + "Missouri" -> "Kansas City Missouri"
        first = places[0]
        merged = [first.text]
        end = first.end
        for ent in places[1:]:
            between = doc[end:ent.start].text.strip()
            if between in {"", ","}:
                merged.append(ent.text)
                end = ent.end
            else:
                break
        return " ".join(merged)

    # -- reminders: what to remind about? ---------------------------------------
    def starts_with_verb(self, text: str) -> bool:
        """'call mom' -> True, 'the team meeting' -> False (tagged in context for accuracy)."""
        doc = self.nlp(f"remind me to {text.strip()}")
        return len(doc) > 3 and doc[3].pos_ in {"VERB", "AUX"}

    def extract_reminder_task(self, text: str) -> Optional[ReminderTask]:
        """Pull the task out of a reminder request whose time phrase was already removed."""
        text = _LEADING_FILLER_RE.sub("", text.strip()).strip(" ,.?!")
        if not text:
            return None
        doc = self.nlp(text)
        found = self._task_from_dependencies(doc)
        if found is None:
            m = _REMINDER_REGEX.search(text)
            if m:
                connector = m.group("connector").lower()
                found = ReminderTask(m.group("task"), "about" if connector in {"of", "for"} else connector)
        if found is None:
            return None
        task = re.sub(r"\s+", " ", found.task).strip(" ,.?!")
        task = re.sub(r"\s+(?:please|for me|ok|okay)$", "", task, flags=re.I)
        return ReminderTask(task, found.connector) if task else None

    def _task_from_dependencies(self, doc) -> Optional[ReminderTask]:
        for tok in doc:
            if tok.lower_ not in _REMINDER_ANCHORS and tok.lemma_.lower() not in _REMINDER_ANCHORS:
                continue
            # 1) a clause: "remind me [to call mom]", "a reminder [to water the plants]"
            for child in tok.children:
                if child.dep_ in _CLAUSE_DEPS and child.i > tok.i and child.pos_ in {"VERB", "AUX"}:
                    span = doc[child.left_edge.i: child.right_edge.i + 1]
                    words = [t for t in span if not (t.i < child.i and t.dep_ in {"aux", "mark"} and t.lower_ in {"to", "that"})]
                    connector = "that" if any(t.dep_ == "mark" and t.lower_ == "that" for t in span) else "to"
                    if child.pos_ == "AUX" or any(t.dep_ == "nsubj" for t in child.children):
                        connector = "that"
                    task = " ".join(t.text for t in words)
                    if task:
                        return ReminderTask(task, connector)
            # 2) "remind me [about the meeting]" / "reminder [for my appointment]"
            heads = [tok] + ([tok.head] if tok.dep_ in {"dobj", "nsubj", "pobj"} else [])
            for head in heads:
                for child in head.children:
                    if child.dep_ == "prep" and child.lower_ in _ABOUT_PREPS and child.i > tok.i:
                        objects = [c for c in child.children if c.dep_ == "pobj"]
                        if objects:
                            span = doc[child.i + 1: child.right_edge.i + 1]
                            return ReminderTask(span.text, "about")
        return None

    # -- questions: what topic? -------------------------------------------------------
    def extract_topic(self, text: str) -> str:
        """'who is alan turing' -> 'alan turing'; falls back to the whole question."""
        question = _LEADING_FILLER_RE.sub("", text.strip()).strip(" ?.!")
        m = re.match(r"^what\s+does\s+(?P<t>.+?)\s+(?:mean|stand\s+for)$", question, re.I)
        if m:
            return m.group("t")
        m = _TOPIC_PREFIX_RE.match(question)
        if m:
            topic = _TOPIC_SUFFIX_RE.sub("", question[m.end():])
            topic = re.sub(r"^(?:a|an|the)\s+", "", topic, flags=re.I).strip()
            if topic:
                return topic
        # no recognisable prefix ("can you tell me who Alan Turing was"): use the single
        # named entity if there is one, otherwise let Wikipedia search the whole question
        doc = self.nlp(question)
        named = [e for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
                                                     "EVENT", "WORK_OF_ART", "PRODUCT", "LAW", "LANGUAGE"}]
        if len(named) == 1:
            return named[0].text
        return question


def _is_not_place(candidate: str) -> bool:
    """'the afternoon', 'the next few days', 'me', 'my city' ... are not places."""
    lowered = re.sub(r"\s+", " ", candidate.lower()).strip()
    bare = re.sub(r"^(?:the|a|an)\s+", "", lowered)
    return (not bare or lowered in _NOT_PLACES or bare in _NOT_PLACES or bare in _NOT_PLACE_WORDS
            or lowered.startswith(("my ", "this ", "next ", "the next ", "your "))
            or bool(_TIME_ONLY_RE.match(lowered)))


def normalize_city(raw: str) -> str:
    """Format a spoken place for OpenWeatherMap: 'lawrence kansas' -> 'Lawrence,KS,US'."""
    raw = re.sub(r"\s+", " ", raw.strip(" ,.?!"))
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        city = parts[0]
        rest = parts[1:]
    else:
        words = raw.split()
        city, rest = raw, []
        for n in (2, 1):  # "new york" / "kansas" as a trailing state name
            if len(words) > n:
                tail = " ".join(words[-n:]).lower()
                if tail in US_STATES or (n == 1 and tail.upper() in _STATE_CODES and len(tail) == 2):
                    city, rest = " ".join(words[:-n]), [tail]
                    break
    city = _title(city)
    if rest:
        first = rest[0].lower()
        state = US_STATES.get(first) or (first.upper() if first.upper() in _STATE_CODES else None)
        if state:
            return f"{city},{state},US"
        return ",".join([city] + [r if len(r) <= 3 else _title(r) for r in rest])
    return city


def _title(text: str) -> str:
    return " ".join(w if w.isupper() and len(w) <= 3 else w.capitalize() for w in text.split())
