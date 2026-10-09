import pytest

from jarvis.nlp.entities import normalize_city, second_person


@pytest.mark.parametrize("text, city", [
    ("what's the weather in Paris", "Paris"),
    ("what's the weather in paris", "Paris"),
    ("how hot is it in lawrence kansas", "Lawrence,KS,US"),
    ("weather in Lawrence, KS", "Lawrence,KS,US"),
    ("weather in new york today", "New York"),
    ("Tokyo weather", "Tokyo"),
    ("is it going to rain tomorrow in Chicago", "Chicago"),
    ("what is the weather like in Kansas City Missouri", "Kansas City,MO,US"),
    ("how's the weather looking in seattle this weekend", "Seattle"),
    ("what's the weather in the morning in chicago", "Chicago"),
    ("what's the temperature in London, England", "London"),
    ("what's the weather like in the afternoon in chicago", "Chicago"),
    ("forecast for the next few days in seattle", "Seattle"),
])
def test_extract_city(extractor, text, city):
    assert extractor.extract_city(text) == city


@pytest.mark.parametrize("text", ["what's the weather like", "weather in the morning",
                                  "what's the forecast for tomorrow", "weather in my city", "is it raining",
                                  "what's the weather like in the afternoon", "weather near me",
                                  "forecast for the week", "weather for the rest of the week"])
def test_no_city(extractor, text):
    assert extractor.extract_city(text) is None


@pytest.mark.parametrize("text, task, connector", [
    ("remind me to call mom", "call mom", "to"),
    ("set a reminder to submit the assignment", "submit the assignment", "to"),
    ("remind me about the team meeting", "the team meeting", "about"),
    ("can you please remind me to take my medicine", "take my medicine", "to"),
    ("set a reminder for my dentist appointment", "my dentist appointment", "about"),
    ("remind me that the pizza is in the oven", "the pizza is in the oven", "that"),
    ("don't let me forget to pay rent", "pay rent", "to"),
    ("ping me to check the laundry", "check the laundry", "to"),
    ("I need a reminder to buy milk", "buy milk", "to"),
])
def test_extract_reminder_task(extractor, text, task, connector):
    found = extractor.extract_reminder_task(text)
    assert found is not None, text
    assert (found.task, found.connector) == (task, connector)


@pytest.mark.parametrize("text", ["remind me", "set a reminder for", "remind me to"])
def test_reminder_without_task(extractor, text):
    assert extractor.extract_reminder_task(text) is None


@pytest.mark.parametrize("text, topic", [
    ("who is alan turing", "alan turing"),
    ("tell me about black holes", "black holes"),
    ("what is machine learning", "machine learning"),
    ("what does serendipity mean", "serendipity"),
    ("define photosynthesis", "photosynthesis"),
    ("search wikipedia for the eiffel tower", "eiffel tower"),
    ("can you tell me who Alan Turing was", "Alan Turing"),
    ("who invented the telephone", "who invented the telephone"),
])
def test_extract_topic(extractor, text, topic):
    assert extractor.extract_topic(text) == topic


@pytest.mark.parametrize("text, verb", [("call mom", True), ("water the plants", True),
                                         ("the team meeting", False), ("my dentist appointment", False)])
def test_starts_with_verb(extractor, text, verb):
    assert extractor.starts_with_verb(text) is verb


def test_second_person():
    assert second_person("call my mom") == "call your mom"
    assert second_person("tell me that I am late") == "tell you that you are late"
    assert second_person("My meeting") == "Your meeting"


def test_normalize_city():
    assert normalize_city("lawrence kansas") == "Lawrence,KS,US"
    assert normalize_city("new york") == "New York"
    assert normalize_city("Austin TX") == "Austin,TX,US"
