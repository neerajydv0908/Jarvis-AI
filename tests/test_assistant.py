"""End-to-end conversations through Assistant.handle (real NLP + model, fake network services)."""

from datetime import datetime, timedelta

import pytest
from conftest import NOW

from jarvis.skills.reminders import Reminder


def test_greeting_and_wake_only(make_assistant):
    assistant = make_assistant()
    assert assistant.handle("hey jarvis").text == "Yes, sir?"
    assert assistant.handle("good morning jarvis").text == "Good afternoon, sir. How can I help?"


def test_identity_uses_creator(make_assistant):
    reply = make_assistant().handle("who made you")
    assert reply.intent == "identity"
    assert "Test Creator" in reply.text


def test_time_and_date(make_assistant):
    assistant = make_assistant()
    assert assistant.handle("what time is it").text == "It's 2 PM."
    assert assistant.handle("jarvis what's today's date").text == "Today is Tuesday, October 6th, 2026."


def test_set_and_list_reminder_in_one_go(make_assistant):
    assistant = make_assistant()
    reply = assistant.handle("remind me to call my mom at 5 pm")
    assert reply.intent == "set_reminder"
    assert reply.text == "Okay, I'll remind you to call your mom at 5 PM."
    [stored] = assistant.reminders.pending()
    assert (stored.task, stored.due) == ("call my mom", datetime(2026, 10, 6, 17, 0))
    assert assistant.handle("what are my reminders").text == \
        "You have one reminder: to call your mom at 5 PM."


def test_reminder_asks_for_missing_time_then_task(make_assistant):
    assistant = make_assistant()
    assert assistant.handle("remind me to water the plants").text == "When should I remind you to water the plants?"
    assert assistant.handle("tomorrow at 8 am").text == \
        "Okay, I'll remind you to water the plants tomorrow at 8 AM."
    assert assistant.handle("set a reminder").text == "Sure. What should I remind you about?"
    assert assistant.handle("submit the lab report").text == "When should I remind you to submit the lab report?"
    assert assistant.handle("in 20 minutes").text == "Okay, I'll remind you to submit the lab report in 20 minutes."
    assert len(assistant.reminders.pending()) == 2


def test_follow_up_can_be_cancelled_or_replaced(make_assistant):
    assistant = make_assistant()
    assistant.handle("remind me to stretch")
    assert assistant.handle("never mind").text == "Okay, no reminder."
    assistant.handle("remind me to stretch")
    assert assistant.handle("what time is it").intent == "time"  # a new command wins
    assert assistant.reminders.pending() == []


def test_new_command_with_a_time_word_is_not_taken_as_the_answer(make_assistant):
    assistant = make_assistant()
    assistant.handle("remind me to water the plants")
    assert assistant.handle("will it rain tomorrow in chicago").text == "Tomorrow's forecast for Chicago."
    assistant.handle("remind me to water the plants")
    reply = assistant.handle("actually remind me to call dad at 6")
    assert reply.text == "Okay, I'll remind you to call dad at 6 PM."
    assert [r.task for r in assistant.reminders.pending()] == ["call dad"]


def test_filler_around_a_time_answer_is_fine(make_assistant):
    assistant = make_assistant()
    assistant.handle("remind me to water the plants")
    assert assistant.handle("um, make it 5 pm please").text == "Okay, I'll remind you to water the plants at 5 PM."


@pytest.mark.parametrize("answer", ["cancel that", "nah forget it", "no thanks", "never mind."])
def test_cancel_phrases(make_assistant, answer):
    assistant = make_assistant()
    assistant.handle("remind me tomorrow at 9")
    assert assistant.handle(answer).text == "Okay, no reminder."
    assert assistant.reminders.pending() == []


def test_one_word_commands_escape_a_question(make_assistant):
    assistant = make_assistant()
    assistant.handle("remind me tomorrow at 9")
    assert assistant.handle("exit").shutdown
    assistant.handle("what's the weather like")
    assert assistant.handle("thanks").intent == "thanks"
    assert assistant.reminders.pending() == []


def test_noun_phrase_answer_uses_about(make_assistant):
    assistant = make_assistant()
    assistant.handle("remind me tomorrow at 9")
    assert assistant.handle("the team meeting").text == "Okay, I'll remind you about the team meeting tomorrow at 9 AM."
    assistant.handle("set a reminder")
    assert assistant.handle("my dentist appointment").text == "When should I remind you about your dentist appointment?"


def test_unanswered_question_expires(make_assistant):
    times = [NOW]
    assistant = make_assistant()
    assistant.clock = lambda: times[0]
    assistant.handle("remind me to call mom")
    times[0] = NOW + timedelta(minutes=10)  # long after the question was asked
    assistant.handle("tomorrow at 9")
    assert assistant.reminders.pending() == []  # "call mom" was not silently scheduled


def test_impossible_follow_up_time_does_not_crash(make_assistant):
    assistant = make_assistant()
    assistant.handle("remind me to stretch")
    assert "didn't catch a time" in assistant.handle("in 99999999 weeks").text


def test_clear_reminders_needs_confirmation(make_assistant):
    assistant = make_assistant()
    assistant.handle("remind me to a in 5 minutes")
    assistant.handle("remind me to b in 10 minutes")
    assert assistant.handle("delete all my reminders").text == "That will delete 2 reminders. Should I go ahead?"
    assert assistant.handle("no").text == "Okay, I'll keep them."
    assert len(assistant.reminders.pending()) == 2
    assistant.handle("clear my reminders")
    assert assistant.handle("yes").text == "Done. I deleted 2 reminders."
    assert assistant.reminders.pending() == []


def test_weather_with_city_and_tomorrow(make_assistant):
    assistant = make_assistant()
    assert assistant.handle("what's the weather in lawrence kansas").text == "Current weather in Lawrence,KS,US."
    assert assistant.handle("will it rain tomorrow in chicago").text == "Tomorrow's forecast for Chicago."


def test_weather_asks_for_city_or_uses_default(make_assistant):
    assistant = make_assistant()
    assert assistant.handle("what's the weather like").text == "Which city should I check?"
    assert assistant.handle("denver").text == "Current weather in Denver."
    assert make_assistant(default_city="Lawrence,KS,US").handle("how hot is it").text == \
        "Current weather in Lawrence,KS,US."


def test_weather_without_api_key(make_assistant):
    assistant = make_assistant()
    assistant.weather.api_key = ""
    assert "OPENWEATHER_API_KEY" in assistant.handle("what's the weather in paris").text


def test_knowledge_question(make_assistant):
    assistant = make_assistant()
    reply = assistant.handle("who is alan turing")
    assert reply.intent == "knowledge"
    assert assistant.knowledge.queries == ["alan turing"]


def test_small_talk_and_unknown_input(make_assistant):
    assistant = make_assistant()
    assert assistant.handle("i'm bored").intent == "general_chat"
    assert assistant.handle("blah blah blah").intent == "general_chat"


def test_goodbye_and_shutdown(make_assistant):
    assistant = make_assistant()
    bye = assistant.handle("that's all, thanks")
    assert bye.end_conversation and not bye.shutdown
    off = assistant.handle("shut down")
    assert off.shutdown


def test_reminder_announcements(make_assistant):
    assistant = make_assistant()
    on_time = Reminder(1, "call my mom", NOW, NOW, "to")
    assert assistant.announcement(on_time) == "Excuse me, sir, it's time to call your mom."
    late = Reminder(2, "the meeting", NOW - timedelta(hours=1), NOW, "about")
    assert assistant.announcement(late) == \
        "Excuse me, sir, here's your reminder about the meeting. It was due at 1 PM."
    assistant.reminder_due(on_time)
    assert assistant.notifications.get_nowait().startswith("Excuse me")


def test_intent_without_code_uses_its_responses(make_assistant):
    assistant = make_assistant()
    assistant.responses["coin_flip"] = ["Heads.", "Tails."]  # a new intent added only in intents.json
    assert assistant._handler_for("coin_flip")("flip a coin").text in {"Heads.", "Tails."}
    assert assistant._handler_for("no_such_intent")("hello there").intent == "general_chat"


def test_honorific_can_be_disabled(make_assistant):
    assert make_assistant(honorific="").handle("jarvis").text == "Yes?"
