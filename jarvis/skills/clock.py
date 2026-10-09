"""Time and date answers."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..nlp.timeparse import format_clock


def _ordinal(day: int) -> str:
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def time_answer(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    return f"It's {format_clock(now)}."


def date_answer(now: Optional[datetime] = None, text: str = "") -> str:
    now = now or datetime.now()
    if "year" in text.lower():
        return f"It's {now.year}."
    if "month" in text.lower():
        return f"It's {now.strftime('%B')}."
    if "weekend" in text.lower():
        return "Yes, it's the weekend." if now.weekday() >= 5 else f"Not yet. Today is {now.strftime('%A')}."
    return f"Today is {now.strftime('%A')}, {now.strftime('%B')} {_ordinal(now.day)}, {now.year}."


def part_of_day(now: Optional[datetime] = None) -> str:
    hour = (now or datetime.now()).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    return "evening"
