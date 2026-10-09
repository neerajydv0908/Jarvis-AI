"""Shared fixtures. The intent model is trained once per test session (CPU, a few seconds)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jarvis.config import Settings  # noqa: E402

NOW = datetime(2026, 10, 6, 14, 0)  # a Tuesday afternoon


@pytest.fixture(scope="session")
def preprocessor():
    from jarvis.nlp.preprocess import TextPreprocessor

    return TextPreprocessor()


@pytest.fixture(scope="session")
def extractor():
    from jarvis.nlp.entities import EntityExtractor

    return EntityExtractor()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, preprocessor):
    from jarvis.brain.train import train

    directory = tmp_path_factory.mktemp("models")
    train(backend="torch", model_dir=directory, device="cpu", val_split=0.0, verbose=False,
          preprocessor=preprocessor, eval_files=())
    return directory


@pytest.fixture(scope="session")
def classifier(model_dir, preprocessor):
    from jarvis.brain.classifier import IntentClassifier

    return IntentClassifier(model_dir, "torch", "cpu", preprocessor)


@dataclass
class _Spoken:
    text: str

    def speak(self) -> str:
        return self.text


class FakeWeather:
    def __init__(self, api_key: str = "test-key"):
        self.api_key = api_key
        self.calls: List[tuple] = []

    def current(self, city):
        self.calls.append(("current", city))
        return _Spoken(f"Current weather in {city}.")

    def forecast(self, city, days_ahead=1):
        self.calls.append(("forecast", city, days_ahead))
        return _Spoken(f"Tomorrow's forecast for {city}.")


class FakeKnowledge:
    def __init__(self):
        self.queries: List[str] = []

    def answer(self, query):
        self.queries.append(query)
        return f"Here is what I found about {query}."


@pytest.fixture
def make_assistant(tmp_path, classifier, extractor):
    from jarvis.assistant import Assistant
    from jarvis.brain.chat import ChatEngine
    from jarvis.brain.dataset import load_intents, responses_by_tag
    from jarvis.skills.reminders import ReminderStore

    created = []

    def factory(now: datetime = NOW, **overrides):
        params = dict(db_path=tmp_path / "reminders.db", model_dir=tmp_path, chat_enabled=False,
                      default_city="", creator="Test Creator")
        params.update(overrides)
        settings = Settings(**params)
        responses = responses_by_tag(load_intents())
        assistant = Assistant(settings, classifier, extractor,
                              ChatEngine(enabled=False, fallback_responses=responses["general_chat"], seed=0),
                              FakeWeather(), ReminderStore(settings.db_path), FakeKnowledge(),
                              responses=responses, clock=lambda: now, seed=0)
        created.append(assistant)
        return assistant

    yield factory
    for assistant in created:
        assistant.close()
