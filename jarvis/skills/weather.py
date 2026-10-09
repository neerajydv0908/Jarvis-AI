"""Weather updates from the OpenWeatherMap API (roadmap step 5).

Uses two free endpoints:
  * /data/2.5/weather   - current conditions
  * /data/2.5/forecast  - 5-day / 3-hour forecast (for "tomorrow")

Get a free key at https://home.openweathermap.org/api_keys and put it in .env
as OPENWEATHER_API_KEY. New keys can take a couple of hours to activate.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5"


class WeatherError(Exception):
    """Raised with a message that is safe to speak to the user."""


@dataclass
class CurrentWeather:
    city: str
    country: str
    description: str
    temperature: float
    feels_like: float
    humidity: int
    wind_speed: float
    units: str

    def speak(self) -> str:
        unit = "degrees"
        wind_unit = "miles per hour" if self.units == "imperial" else "meters per second"
        return (f"Right now in {self.city} it's {round(self.temperature)} {unit} with {self.description}. "
                f"It feels like {round(self.feels_like)}, humidity is {self.humidity} percent "
                f"and the wind is at {round(self.wind_speed)} {wind_unit}.")


@dataclass
class DayForecast:
    city: str
    day_label: str
    description: str
    high: float
    low: float
    rain_chance: int

    def speak(self) -> str:
        text = (f"{self.day_label.capitalize()} in {self.city}: {self.description}, "
                f"with a high of {round(self.high)} and a low of {round(self.low)} degrees.")
        if self.rain_chance >= 20:
            text += f" There's a {self.rain_chance} percent chance of rain."
        return text


class WeatherService:
    def __init__(self, api_key: str, units: str = "imperial", timeout: float = 8.0,
                 session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.units = units if units in {"imperial", "metric"} else "imperial"
        self.timeout = timeout
        self.session = session or requests.Session()

    # -- HTTP --------------------------------------------------------------------
    def _get(self, endpoint: str, city: str) -> dict:
        if not self.api_key:
            raise WeatherError("I need an OpenWeatherMap API key for weather updates. "
                               "Add OPENWEATHER_API_KEY to your .env file.")
        attempts = [city]
        if "," in city:  # "London,England" -> retry as "London"
            attempts.append(city.split(",")[0].strip())
        last_status = None
        for query in attempts:
            try:
                response = self.session.get(f"{BASE_URL}/{endpoint}",
                                            params={"q": query, "appid": self.api_key, "units": self.units},
                                            timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("Weather request failed: %s", exc)
                raise WeatherError("I couldn't reach the weather service. Check your internet connection.") from exc
            last_status = response.status_code
            if response.status_code == 200:
                return response.json()
            if response.status_code == 401:
                raise WeatherError("The weather service rejected the API key. If you just created it, "
                                   "it can take a couple of hours to activate.")
            if response.status_code != 404:
                break
        if last_status == 404:
            raise WeatherError(f"I couldn't find a place called {city.split(',')[0]}.")
        raise WeatherError("The weather service returned an error. Please try again in a moment.")

    # -- public API ----------------------------------------------------------------
    def current(self, city: str) -> CurrentWeather:
        data = self._get("weather", city)
        try:
            return CurrentWeather(
                city=data.get("name") or city.split(",")[0],
                country=data.get("sys", {}).get("country", ""),
                description=data["weather"][0]["description"],
                temperature=float(data["main"]["temp"]),
                feels_like=float(data["main"].get("feels_like", data["main"]["temp"])),
                humidity=int(data["main"].get("humidity", 0)),
                wind_speed=float(data.get("wind", {}).get("speed", 0.0)),
                units=self.units,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise WeatherError("The weather service sent a response I couldn't read.") from exc

    def forecast(self, city: str, days_ahead: int = 1) -> DayForecast:
        """Summarise the forecast for a future day (1 = tomorrow) in the city's local time."""
        data = self._get("forecast", city)
        try:
            offset = timedelta(seconds=int(data.get("city", {}).get("timezone", 0)))
            target = (datetime.now(timezone.utc) + offset).date() + timedelta(days=days_ahead)
            slots = [item for item in data["list"]
                     if (datetime.fromtimestamp(item["dt"], timezone.utc) + offset).date() == target]
            if not slots:
                raise WeatherError("I don't have a forecast that far ahead.")
            descriptions = Counter(item["weather"][0]["description"] for item in slots)
            label = "tomorrow" if days_ahead == 1 else target.strftime("%A")
            return DayForecast(
                city=data.get("city", {}).get("name") or city.split(",")[0],
                day_label=label,
                description=descriptions.most_common(1)[0][0],
                high=max(float(item["main"]["temp_max"]) for item in slots),
                low=min(float(item["main"]["temp_min"]) for item in slots),
                rain_chance=round(100 * max(float(item.get("pop", 0)) for item in slots)),
            )
        except WeatherError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise WeatherError("The weather service sent a forecast I couldn't read.") from exc
