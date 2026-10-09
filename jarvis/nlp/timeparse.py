"""Turn spoken reminder times into datetimes.

Handles the phrasings people actually say to a voice assistant:

* relative  - "in 10 minutes", "in an hour and a half", "in half an hour",
              "in 1 hour and 30 minutes", "2 hours from now", "in five minutes"
* clock     - "at 5", "at 5 pm", "at 5:30 p.m.", "at 17:45", "at five thirty",
              "quarter past 5", "ten to 6", "at noon", "at midnight", "at 7 in the morning"
* day words - "today", "tonight", "tomorrow (morning)", "day after tomorrow",
              "this evening", "on friday", "next monday"

and combinations such as "tomorrow at 9 am", "at 9 tonight" or
"on friday at 3 in the afternoon".

``parse_time`` returns the resolved datetime plus the character spans it used,
so the caller can cut the time phrase out and keep only the task text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import List, Optional, Tuple

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_FUZZY = {
    "a": 1, "an": 1, "a couple of": 2, "a couple": 2, "couple of": 2, "a few": 3, "few": 3,
    "half an": 0.5, "half a": 0.5, "quarter": 15, "a quarter": 15, "half": 30,
}
_UNIT_SECONDS = {"sec": 1, "second": 1, "min": 60, "minute": 60, "hr": 3600, "hour": 3600,
                 "day": 86400, "week": 604800}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# default clock time when only a part of the day is given
_PERIOD_DEFAULT = {"morning": (9, 0), "afternoon": (15, 0), "evening": (19, 0), "night": (20, 0)}
_MAX_AHEAD = timedelta(days=3650)

_NUMBER_WORDS = (
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
    r"|fifteen|sixteen|seventeen|eighteen|nineteen"
)
_AMOUNT = (
    r"(?:\d+(?:\.\d+)?|a\s+couple\s+of|a\s+couple|couple\s+of|a\s+few|few|half\s+an?|an?|"
    + _NUMBER_WORDS + ")"
)
_UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
_HOUR_WORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_AMPM = r"(?P<ampm>[ap]\.?\s?m\b\.?)"
_PERIOD = r"(?:\s+(?:in\s+the\s+)?(?P<period>morning|afternoon|evening|night))?"

_RELATIVE_RE = re.compile(
    rf"\b(?:in|after|within)\s+(?P<a1>{_AMOUNT})\s+(?P<u1>{_UNIT})"
    rf"(?:\s*,?\s+(?:and\s+)?(?P<a2>{_AMOUNT})\s+(?P<u2>{_UNIT}))?"
    rf"(?P<half>\s+and\s+a\s+half)?\b",
    re.I,
)
_FROM_NOW_RE = re.compile(
    rf"\b(?P<a1>{_AMOUNT})\s+(?P<u1>{_UNIT})(?P<half>\s+and\s+a\s+half)?\s+from\s+now\b", re.I
)
# "at 5", "at 5:30 pm", "6 30 pm", "five thirty", "17:45", "10 o'clock in the morning"
_CLOCK_RE = re.compile(
    rf"(?:\b(?P<prefix>at|by|around|before|for)\s+)?"
    rf"\b(?P<hour>\d{{1,2}}|{_HOUR_WORD})"
    rf"(?:[:.](?P<minute>\d{{2}})|\s+(?P<minword>fifteen|thirty|forty[\s-]five|o'?\s?clock)"
    rf"|\s+(?P<minute_sp>[0-5]\d)(?=\s*[ap]\.?\s?m\b))?"
    rf"(?:\s*{_AMPM})?"
    rf"(?P<oclock>\s+o'?\s?clock)?"
    rf"{_PERIOD}"
    rf"(?![\w:])",
    re.I,
)
# "quarter past 5", "half past six", "ten to 6", "20 minutes past 4 pm"
_PAST_TO_RE = re.compile(
    rf"(?:\b(?P<prefix>at|by|around|for)\s+)?"
    rf"\b(?P<amount>a\s+quarter|quarter|half|\d{{1,2}}|{_NUMBER_WORDS})(?:\s+minutes?)?"
    rf"\s+(?P<rel>past|after|to|till|before)\s+(?P<hour>\d{{1,2}}|{_HOUR_WORD})"
    rf"(?:\s*{_AMPM})?{_PERIOD}(?![\w:])",
    re.I,
)
_SPECIAL_RE = re.compile(r"\b(?:(?:at|by|around|for)\s+)?(?:12\s+)?(?P<word>noon|midday|midnight)\b", re.I)
_DAY_RE = re.compile(
    r"(?:\b(?:for|by|until|on)\s+)?"
    r"\b(?P<day>day\s+after\s+tomorrow|tomorrow|today|tonight"
    r"|(?:(?:on|this|next|coming)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))"
    r"(?:\s+(?P<period>morning|afternoon|evening|night))?\b",
    re.I,
)
_PERIOD_RE = re.compile(r"\b(?:(?:in\s+the|this)\s+)(?P<period>morning|afternoon|evening)\b|\b(?P<tonight>tonight)\b", re.I)


@dataclass
class ParsedTime:
    when: datetime
    spans: List[Tuple[int, int]] = field(default_factory=list)
    relative: bool = False

    def strip_from(self, text: str) -> str:
        """Remove the time phrase(s) from ``text`` and tidy the leftovers."""
        for start, end in sorted(self.spans, reverse=True):
            text = text[:start] + " " + text[end:]
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"\s+([,.?!])", r"\1", text).strip(" ,")


def to_number(text: str) -> Optional[float]:
    """'10' -> 10, 'ten' -> 10, 'twenty five' -> 25, 'a couple of' -> 2, 'half an' -> 0.5."""
    s = re.sub(r"[\s-]+", " ", text.strip().lower())
    try:
        return float(s)
    except ValueError:
        pass
    if s in _FUZZY:
        return float(_FUZZY[s])
    if s in _ONES:
        return float(_ONES[s])
    if s in _TENS:
        return float(_TENS[s])
    parts = s.split()
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES and _ONES[parts[1]] < 10:
        return float(_TENS[parts[0]] + _ONES[parts[1]])
    return None


def _unit_seconds(unit: str) -> int:
    unit = unit.lower().rstrip("s")
    return _UNIT_SECONDS.get(unit, _UNIT_SECONDS.get(unit[:3], 0))


def _parse_relative(text: str, now: datetime) -> Optional[ParsedTime]:
    m = _RELATIVE_RE.search(text) or _FROM_NOW_RE.search(text)
    if not m:
        return None
    first = to_number(m.group("a1"))
    if first is None:
        return None
    seconds = first * _unit_seconds(m.group("u1"))
    if m.groupdict().get("a2"):
        second = to_number(m.group("a2"))
        if second is not None:
            seconds += second * _unit_seconds(m.group("u2"))
    if m.group("half"):
        seconds += 0.5 * _unit_seconds(m.group("u1"))
    if seconds <= 0 or seconds > _MAX_AHEAD.total_seconds():
        return None
    return ParsedTime(now + timedelta(seconds=round(seconds)), [m.span()], relative=True)


def _overlaps(span: Tuple[int, int], others: List[Tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in others)


def _hour_value(text: str) -> int:
    text = text.lower()
    return int(text) if text.isdigit() else _ONES[text]


def _clean_ampm(m: re.Match) -> Optional[str]:
    return (m.group("ampm") or "").lower().replace(".", "").replace(" ", "") or None


def _find_clock(text: str, taken: List[Tuple[int, int]]):
    """First clock expression that is clearly a time (not just a bare number).

    Returns (match, hour, minute, ampm, period) or None.
    """
    for m in _PAST_TO_RE.finditer(text):
        if _overlaps(m.span(), taken):
            continue
        amount = to_number(m.group("amount"))
        hour = _hour_value(m.group("hour"))
        if amount is None or not 0 < amount < 60 or hour > 23:
            continue
        minute = int(amount)
        if m.group("rel").lower() in {"to", "till", "before"}:
            hour, minute = (hour - 1) % 24, 60 - minute
            if hour == 0 and not m.group("ampm"):
                hour = 12
        return m, hour, minute, _clean_ampm(m), (m.group("period") or "").lower() or None

    for m in _CLOCK_RE.finditer(text):
        if _overlaps(m.span(), taken):
            continue
        strong_prefix = (m.group("prefix") or "").lower() in {"at", "by", "around", "before"}
        has_marker = strong_prefix or any(m.group(g) for g in ("minute", "minute_sp", "minword", "ampm", "oclock", "period"))
        if not has_marker:
            continue  # "buy 2 apples", "a table for 2" - a number, not a time
        hour = _hour_value(m.group("hour"))
        minute = 0
        if m.group("minute") or m.group("minute_sp"):
            minute = int(m.group("minute") or m.group("minute_sp"))
        elif m.group("minword") and "clock" not in m.group("minword").lower():
            minute = int(to_number(m.group("minword").replace("-", " ")) or 0)
        if hour > 23 or minute > 59:
            continue
        return m, hour, minute, _clean_ampm(m), (m.group("period") or "").lower() or None
    return None


def _to_24h(hour: int, ampm: Optional[str], period: Optional[str]) -> Optional[int]:
    """Resolve a 12-hour clock reading; None when it is genuinely ambiguous."""
    if hour > 12 or hour == 0:
        return hour  # already 24-hour
    if ampm == "am" or period == "morning":
        return 0 if hour == 12 else hour
    if ampm == "pm" or period in {"afternoon", "evening", "night"}:
        return 12 if hour == 12 else hour + 12
    return None


def _guess_24h(hour: int) -> int:
    """No am/pm on a future day: 7-11 sounds like morning, 12-6 like afternoon/evening."""
    if hour == 12:
        return 12
    return hour if 7 <= hour <= 11 else hour + 12


def _resolve_day(match: re.Match, today: date) -> Tuple[date, str]:
    """Return (date, kind) for a day phrase; kind is 'today', 'fixed' or 'weekday'."""
    word = re.sub(r"\s+", " ", match.group("day").lower())
    if word in {"today", "tonight"}:
        return today, "today"
    if word == "tomorrow":
        return today + timedelta(days=1), "fixed"
    if word == "day after tomorrow":
        return today + timedelta(days=2), "fixed"
    parts = word.split()
    weekday = _WEEKDAYS.index(parts[-1])
    ahead = (weekday - today.weekday()) % 7
    if parts[0] == "next" and ahead == 0:
        ahead = 7
    return today + timedelta(days=ahead), "weekday"


def parse_time(text: str, now: Optional[datetime] = None) -> Optional[ParsedTime]:
    """Find a reminder time in ``text``. Returns None when there is no (future) time phrase."""
    now = now or datetime.now()
    relative = _parse_relative(text, now)
    if relative:
        return relative

    spans: List[Tuple[int, int]] = []
    day_match = _DAY_RE.search(text)
    day, day_kind = now.date(), "none"
    period: Optional[str] = None
    if day_match:
        spans.append(day_match.span())
        day, day_kind = _resolve_day(day_match, now.date())
        period = (day_match.group("period") or "").lower() or None
        if day_match.group("day").lower() == "tonight":
            period = "night"

    hour = minute = None
    ampm: Optional[str] = None
    special = _SPECIAL_RE.search(text)
    clock = _find_clock(text, spans)
    if special and not _overlaps(special.span(), spans):
        spans.append(special.span())
        word = special.group("word").lower()
        hour, minute, ampm = (0, 0, "am") if word == "midnight" else (12, 0, "pm")
        if word == "midnight" and day_kind in {"none", "today"}:
            day, day_kind = now.date() + timedelta(days=1), "fixed"
    elif clock:
        m, hour, minute, ampm, clock_period = clock
        spans.append(m.span())
        period = clock_period or period

    # "in the morning", "this evening", "tonight" anywhere else in the sentence
    loose = _PERIOD_RE.search(text)
    if loose and not _overlaps(loose.span(), spans):
        spans.append(loose.span())
        period = period or ("night" if loose.group("tonight") else loose.group("period").lower())

    explicit_day = day_kind in {"fixed", "weekday"}
    if hour is None:
        if period is None and not explicit_day:
            return None  # only "today" - no actual time to schedule
        hour, minute = _PERIOD_DEFAULT[period or "morning"]
        ampm = "am" if hour < 12 else "pm"

    if period == "night" and ampm is None and hour in {12, 1, 2, 3, 4}:
        # "tonight at 12" is midnight, "tonight at 1" is 1 AM - both after the date changes
        hour24 = 0 if hour == 12 else hour
        if day_kind in {"none", "today"}:
            day, day_kind = day + timedelta(days=1), "fixed"
    else:
        hour24 = _to_24h(hour, ampm, period)

    if hour24 is not None:
        when = datetime.combine(day, time(hour24, minute))
        if when <= now and day_kind in {"none", "today"}:
            when += timedelta(days=1)
    elif explicit_day:
        when = datetime.combine(day, time(_guess_24h(hour), minute))
    else:
        # "at 5" with no am/pm: the next time the clock reads 5:00, skipping the small
        # hours (nobody means 3 AM unless they say so). At 2 PM "at 5" is 5 PM today;
        # at 11 PM "at 6" is 6 AM tomorrow; at 2 PM "at 1" is 1 PM tomorrow.
        options = sorted(datetime.combine(day + timedelta(days=d), time(h, minute))
                         for d in (0, 1) for h in (hour % 12, hour % 12 + 12))
        upcoming = [o for o in options if o > now and o.hour >= 6]
        when = upcoming[0] if upcoming else datetime.combine(day + timedelta(days=1), time(_guess_24h(hour), minute))

    if when <= now and day_kind == "weekday":
        when += timedelta(days=7)  # "on tuesday at 1 pm" said on a Tuesday afternoon
    if when <= now:
        return None
    return ParsedTime(when, spans)


def describe_when(when: datetime, now: Optional[datetime] = None) -> str:
    """Speakable description: 'in 10 minutes', 'at 5:30 PM', 'tomorrow at 9 AM', 'on Friday at 3 PM'."""
    now = now or datetime.now()
    delta = when - now
    if timedelta(0) < delta < timedelta(hours=1):
        minutes = max(1, round(delta.total_seconds() / 60))
        if delta.total_seconds() < 90:
            secs = round(delta.total_seconds())
            return f"in {secs} seconds" if secs < 60 else "in a minute"
        return f"in {minutes} minutes"
    clock = format_clock(when)
    if when.date() == now.date():
        return f"at {clock}"
    if when.date() == now.date() + timedelta(days=1):
        return f"tomorrow at {clock}"
    if when.date() - now.date() < timedelta(days=7):
        return f"on {when.strftime('%A')} at {clock}"
    return f"on {when.strftime('%B')} {when.day} at {clock}"


def format_clock(when: datetime) -> str:
    """5:30 PM / 9 AM (no leading zero, works on every platform)."""
    hour12 = when.hour % 12 or 12
    suffix = "AM" if when.hour < 12 else "PM"
    return f"{hour12} {suffix}" if when.minute == 0 else f"{hour12}:{when.minute:02d} {suffix}"
