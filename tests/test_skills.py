import time
from datetime import datetime, timezone

import pytest
import requests

from jarvis.skills.clock import date_answer, part_of_day, time_answer
from jarvis.skills.jokes import JOKES, JokeTeller
from jarvis.skills.knowledge import KnowledgeError, Wikipedia, clean_for_speech
from jarvis.skills.weather import WeatherError, WeatherService


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


CURRENT = {"name": "Lawrence", "sys": {"country": "US"}, "weather": [{"description": "clear sky"}],
           "main": {"temp": 71.6, "feels_like": 70.2, "humidity": 40}, "wind": {"speed": 5.4}}


# -- weather ---------------------------------------------------------------------------------------
def test_current_weather():
    session = FakeSession([FakeResponse(200, CURRENT)])
    report = WeatherService("key", session=session).current("Lawrence,KS,US")
    assert session.calls[0][1]["q"] == "Lawrence,KS,US"
    assert session.calls[0][1]["units"] == "imperial"
    assert report.speak() == ("Right now in Lawrence it's 72 degrees with clear sky. It feels like 70, "
                              "humidity is 40 percent and the wind is at 5 miles per hour.")


def test_weather_errors():
    with pytest.raises(WeatherError, match="API key"):
        WeatherService("").current("Paris")
    with pytest.raises(WeatherError, match="activate"):
        WeatherService("bad", session=FakeSession([FakeResponse(401)])).current("Paris")
    with pytest.raises(WeatherError, match="couldn't find a place called Atlantis"):
        WeatherService("k", session=FakeSession([FakeResponse(404)])).current("Atlantis")
    with pytest.raises(WeatherError, match="internet"):
        WeatherService("k", session=FakeSession([requests.ConnectionError("down")])).current("Paris")


def test_weather_retries_without_country_on_404():
    session = FakeSession([FakeResponse(404), FakeResponse(200, CURRENT)])
    WeatherService("k", session=session).current("London,England")
    assert [c[1]["q"] for c in session.calls] == ["London,England", "London"]


def test_forecast_tomorrow():
    tomorrow_noon = int(time.time()) + 86400
    day = datetime.fromtimestamp(tomorrow_noon, timezone.utc).replace(hour=0, minute=0, second=0)
    slots = []
    for hour, temp, desc, pop in [(9, 60, "light rain", 0.6), (12, 75, "light rain", 0.4), (15, 80, "clouds", 0.1)]:
        slots.append({"dt": int(day.replace(hour=hour).timestamp()), "main": {"temp_max": temp, "temp_min": temp - 5},
                      "weather": [{"description": desc}], "pop": pop})
    payload = {"city": {"name": "Lawrence", "timezone": 0}, "list": slots}
    forecast = WeatherService("k", session=FakeSession([FakeResponse(200, payload)])).forecast("Lawrence")
    assert (forecast.high, forecast.low, forecast.description, forecast.rain_chance) == (80, 55, "light rain", 60)
    assert "Tomorrow in Lawrence" in forecast.speak() and "60 percent chance of rain" in forecast.speak()


# -- Wikipedia ------------------------------------------------------------------------------------------
def test_wikipedia_answer_skips_disambiguation():
    session = FakeSession([
        FakeResponse(200, {"query": {"search": [{"title": "Mercury"}, {"title": "Mercury (planet)"}]}}),
        FakeResponse(200, {"query": {"pages": {"1": {"extract": "Mercury may refer to: a planet, an element."}}}}),
        FakeResponse(200, {"query": {"pages": {"2": {"extract": "Mercury (symbol ☿) is the first planet from the Sun. "
                                                                "It is the smallest planet. It has no moons."}}}}),
    ])
    answer = Wikipedia(sentences=2, session=session).answer("mercury")
    assert answer == "Mercury is the first planet from the Sun. It is the smallest planet."
    assert session.headers["User-Agent"].startswith("JARVIS")


def test_wikipedia_errors():
    with pytest.raises(KnowledgeError, match="couldn't find"):
        Wikipedia(session=FakeSession([FakeResponse(200, {"query": {"search": []}})])).answer("zxqv")
    with pytest.raises(KnowledgeError, match="reach Wikipedia"):
        Wikipedia(session=FakeSession([requests.ConnectionError("offline")])).answer("python")


def test_clean_for_speech():
    text = "Alan Turing (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954) was an English mathematician.[1]"
    assert clean_for_speech(text) == "Alan Turing was an English mathematician."


# -- clock & jokes -------------------------------------------------------------------------------------
def test_clock_answers():
    now = datetime(2026, 10, 6, 14, 5)
    assert time_answer(now) == "It's 2:05 PM."
    assert date_answer(now) == "Today is Tuesday, October 6th, 2026."
    assert date_answer(now, "what year is it") == "It's 2026."
    assert date_answer(datetime(2026, 10, 11), "is it the weekend") == "Yes, it's the weekend."
    assert part_of_day(now) == "afternoon"


def test_jokes_do_not_repeat_until_all_told():
    teller = JokeTeller(seed=1)
    told = [teller.tell() for _ in JOKES]
    assert sorted(told) == sorted(JOKES)
