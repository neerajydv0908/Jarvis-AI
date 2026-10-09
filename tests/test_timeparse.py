from datetime import datetime

import pytest

from jarvis.nlp.timeparse import describe_when, format_clock, parse_time, to_number

NOW = datetime(2026, 10, 6, 14, 0)  # Tuesday 2 PM


@pytest.mark.parametrize("text, expected", [
    ("remind me to call mom in 10 minutes", datetime(2026, 10, 6, 14, 10)),
    ("in an hour and a half", datetime(2026, 10, 6, 15, 30)),
    ("in half an hour", datetime(2026, 10, 6, 14, 30)),
    ("in 1 hour and 30 minutes", datetime(2026, 10, 6, 15, 30)),
    ("in five minutes", datetime(2026, 10, 6, 14, 5)),
    ("in twenty five minutes", datetime(2026, 10, 6, 14, 25)),
    ("in a couple of hours", datetime(2026, 10, 6, 16, 0)),
    ("2 hours from now", datetime(2026, 10, 6, 16, 0)),
    ("in 2 days", datetime(2026, 10, 8, 14, 0)),
    ("at 5 pm", datetime(2026, 10, 6, 17, 0)),
    ("at 5 p.m.", datetime(2026, 10, 6, 17, 0)),
    ("at 5", datetime(2026, 10, 6, 17, 0)),
    ("at 1", datetime(2026, 10, 7, 13, 0)),
    ("at 1 p.m.", datetime(2026, 10, 7, 13, 0)),
    ("at 17:45", datetime(2026, 10, 6, 17, 45)),
    ("at five thirty pm", datetime(2026, 10, 6, 17, 30)),
    ("at 5:30 in the evening", datetime(2026, 10, 6, 17, 30)),
    ("at 7 in the morning", datetime(2026, 10, 7, 7, 0)),
    ("at noon", datetime(2026, 10, 7, 12, 0)),
    ("at midnight", datetime(2026, 10, 7, 0, 0)),
    ("tomorrow at 9 am", datetime(2026, 10, 7, 9, 0)),
    ("tomorrow at 5", datetime(2026, 10, 7, 17, 0)),
    ("tomorrow morning", datetime(2026, 10, 7, 9, 0)),
    ("tonight", datetime(2026, 10, 6, 20, 0)),
    ("tonight at 8", datetime(2026, 10, 6, 20, 0)),
    ("this evening", datetime(2026, 10, 6, 19, 0)),
    ("on friday at 3 pm", datetime(2026, 10, 9, 15, 0)),
    ("on tuesday at 1 pm", datetime(2026, 10, 13, 13, 0)),
    ("next monday", datetime(2026, 10, 12, 9, 0)),
    ("day after tomorrow at 10", datetime(2026, 10, 8, 10, 0)),
    ("remind me to take my medicine at 9 tonight", datetime(2026, 10, 6, 21, 0)),
    ("at 11 tonight", datetime(2026, 10, 6, 23, 0)),
    ("tonight at 12", datetime(2026, 10, 7, 0, 0)),
    ("tonight at 1", datetime(2026, 10, 7, 1, 0)),
    ("tomorrow night at 9", datetime(2026, 10, 7, 21, 0)),
    ("remind me to stretch in the morning at 7", datetime(2026, 10, 7, 7, 0)),
    ("set a reminder for 5 pm to call mom", datetime(2026, 10, 6, 17, 0)),
    ("in seventy minutes", datetime(2026, 10, 6, 15, 10)),
    ("at 6 30 pm", datetime(2026, 10, 6, 18, 30)),
    ("quarter past 5", datetime(2026, 10, 6, 17, 15)),
    ("half past six", datetime(2026, 10, 6, 18, 30)),
    ("ten to 6", datetime(2026, 10, 6, 17, 50)),
    ("at 12 noon", datetime(2026, 10, 7, 12, 0)),
])
def test_parse_time(text, expected):
    parsed = parse_time(text, NOW)
    assert parsed is not None, text
    assert parsed.when == expected


@pytest.mark.parametrize("text", ["remind me to buy 2 apples", "remind me today to stretch", "set a reminder",
                                  "book a table for 2", "remind me in 99999999 weeks"])
def test_no_time(text):
    assert parse_time(text, NOW) is None


def test_period_before_clock_early_morning():
    assert parse_time("this evening at 6", datetime(2026, 10, 6, 5, 0)).when == datetime(2026, 10, 6, 18, 0)


def test_ambiguous_hour_picks_next_daytime_occurrence():
    late = datetime(2026, 10, 6, 22, 45)
    assert parse_time("at 6", late).when == datetime(2026, 10, 7, 6, 0)
    assert parse_time("at 12", late).when == datetime(2026, 10, 7, 12, 0)


@pytest.mark.parametrize("text, rest", [
    ("remind me to call mom at 5 p.m.", "remind me to call mom"),
    ("remind me at 6 to take out the trash", "remind me to take out the trash"),
    ("set a reminder for tomorrow at 9 am to submit the assignment", "set a reminder to submit the assignment"),
    ("remind me in 10 minutes to check the oven", "remind me to check the oven"),
    ("remind me to read chapter 5 at 7 pm", "remind me to read chapter 5"),
    ("remind me to take my medicine at 9 tonight", "remind me to take my medicine"),
    ("set a reminder for 5 pm to call mom", "set a reminder to call mom"),
    ("remind me to stretch in the morning at 7", "remind me to stretch"),
])
def test_strip_time_phrase(text, rest):
    assert parse_time(text, NOW).strip_from(text) == rest


def test_to_number():
    assert to_number("10") == 10
    assert to_number("twenty-five") == 25
    assert to_number("a couple of") == 2
    assert to_number("half an") == 0.5
    assert to_number("banana") is None


def test_describe_and_format():
    assert format_clock(datetime(2026, 1, 1, 17, 0)) == "5 PM"
    assert format_clock(datetime(2026, 1, 1, 0, 5)) == "12:05 AM"
    assert describe_when(datetime(2026, 10, 6, 14, 10), NOW) == "in 10 minutes"
    assert describe_when(datetime(2026, 10, 6, 17, 30), NOW) == "at 5:30 PM"
    assert describe_when(datetime(2026, 10, 7, 9, 0), NOW) == "tomorrow at 9 AM"
    assert describe_when(datetime(2026, 10, 9, 15, 0), NOW) == "on Friday at 3 PM"
