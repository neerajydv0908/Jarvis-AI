"""The intent model must understand realistic commands it was never trained on."""

import json
from pathlib import Path

import pytest

from jarvis.nlp.preprocess import normalize_wake_word

DATA = Path(__file__).parent / "data"


def _accuracy(classifier, filename):
    rows = json.loads((DATA / filename).read_text())
    errors = []
    for row in rows:
        prediction = classifier.predict(normalize_wake_word(row["text"]))
        if prediction.intent != row["intent"]:
            errors.append((row["text"], row["intent"], prediction.intent))
    return 1 - len(errors) / len(rows), errors


def test_heldout_commands(classifier):
    accuracy, errors = _accuracy(classifier, "heldout_utterances.json")
    assert accuracy >= 0.95, errors


def test_blind_commands(classifier):
    # written after the training data was finalised and never used for tuning
    accuracy, errors = _accuracy(classifier, "blind_utterances.json")
    assert accuracy >= 0.85, errors


@pytest.mark.parametrize("text, intent", [
    ("what's the weather in chicago", "weather"),
    ("remind me to call mom at 5 pm", "set_reminder"),
    ("who is alan turing", "knowledge"),
    ("what time is it", "time"),
    ("who are you", "identity"),
    ("how are you", "how_are_you"),
    ("delete all my reminders", "clear_reminders"),
    ("what are my reminders", "list_reminders"),
    ("shut down", "shutdown"),
    ("tell me a joke", "joke"),
])
def test_core_commands(classifier, text, intent):
    prediction = classifier.predict(text)
    assert prediction.intent == intent
    assert prediction.confidence > 0.6


def test_prediction_details(classifier):
    prediction = classifier.predict("blah blah")
    assert prediction.known_tokens == 0  # the assistant routes these to small talk
    assert len(prediction.ranking) == 3
    assert classifier.predict("").intent == "general_chat"


def test_tensorflow_backend(tmp_path, preprocessor):
    pytest.importorskip("tensorflow")
    from jarvis.brain.classifier import IntentClassifier
    from jarvis.brain.train import train

    train(backend="tf", model_dir=tmp_path, val_split=0.0, verbose=False, preprocessor=preprocessor,
          eval_files=(), epochs=150)
    tf_classifier = IntentClassifier(tmp_path, "tf", preprocessor=preprocessor)
    assert tf_classifier.predict("what time is it").intent == "time"
    assert tf_classifier.predict("remind me to call mom at 5 pm").intent == "set_reminder"


def test_metadata_written(model_dir):
    meta = json.loads((model_dir / "intent_torch.json").read_text())
    assert meta["backend"] == "torch"
    assert meta["vocab"][0] == "<unk>"
    assert {"weather", "set_reminder", "knowledge", "general_chat"} <= set(meta["labels"])
