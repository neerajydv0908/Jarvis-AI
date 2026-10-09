"""Training data: intents.json -> normalised token lists -> bag-of-words vectors."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..nlp.preprocess import SLOT, UNK, TextPreprocessor

INTENTS_PATH = Path(__file__).with_name("intents.json")
_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def load_intents(path: Path = INTENTS_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["intents"]


def responses_by_tag(intents: Sequence[dict]) -> Dict[str, List[str]]:
    return {intent["tag"]: list(intent.get("responses", [])) for intent in intents}


@dataclass
class Sample:
    text: str
    tokens: List[str]
    label: str


def build_samples(intents: Sequence[dict], preprocessor: TextPreprocessor) -> List[Sample]:
    """Normalise every pattern. ``{city}``-style placeholders become the <unk> feature."""
    samples = []
    for intent in intents:
        for pattern in intent["patterns"]:
            tokens = preprocessor.normalize(_PLACEHOLDER_RE.sub(SLOT, pattern))
            if tokens:
                samples.append(Sample(pattern, tokens, intent["tag"]))
    return samples


def augment_with_unknowns(samples: Sequence[Sample]) -> List[Sample]:
    """Add a copy of every pattern with one extra unknown word.

    Real commands contain words the model never saw ("good night *jarvis*",
    "hiya *mate*"). Without this, the <unk> feature only ever appears in
    templates like "who is {x}", and any unseen word drags a prediction
    towards those intents.
    """
    extra = [Sample(s.text + " <unk>", s.tokens + [UNK], s.label) for s in samples if UNK not in s.tokens]
    return list(samples) + extra


class Vocabulary:
    """Maps normalised tokens to bag-of-words positions; index 0 is <unk>."""

    def __init__(self, words: Iterable[str]):
        self.itos: List[str] = [UNK] + sorted({w for w in words if w != UNK})
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def from_samples(cls, samples: Iterable[Sample]) -> "Vocabulary":
        return cls(token for sample in samples for token in sample.tokens)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Binary bag of words: 1.0 where a token occurs (unknown words light up <unk>)."""
        vector = np.zeros(len(self.itos), dtype=np.float32)
        for token in tokens:
            vector[self.stoi.get(token, 0)] = 1.0
        return vector

    def known_count(self, tokens: Sequence[str]) -> int:
        return sum(1 for t in tokens if t != UNK and t in self.stoi)


def encode_samples(samples: Sequence[Sample], vocab: Vocabulary, labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    index = {label: i for i, label in enumerate(labels)}
    X = np.stack([vocab.encode(s.tokens) for s in samples]) if samples else np.zeros((0, len(vocab)), np.float32)
    y = np.array([index[s.label] for s in samples], dtype=np.int64)
    return X, y


def stratified_split(samples: Sequence[Sample], val_fraction: float, seed: int = 42,
                     ) -> Tuple[List[Sample], List[Sample]]:
    """Hold out ``val_fraction`` of every intent's patterns for an honest accuracy estimate."""
    by_label: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)
    rng = random.Random(seed)
    train: List[Sample] = []
    val: List[Sample] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n_val = int(round(len(items) * val_fraction)) if val_fraction > 0 else 0
        n_val = min(max(n_val, 1 if val_fraction > 0 else 0), len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def load_labelled_utterances(path: Path) -> List[Tuple[str, str]]:
    """Read a JSON list of {"text": ..., "intent": ...} (used for held-out evaluation)."""
    with open(path, encoding="utf-8") as fh:
        return [(row["text"], row["intent"]) for row in json.load(fh)]


def label_set(samples: Iterable[Sample], extra: Optional[Iterable[str]] = None) -> List[str]:
    labels = {s.label for s in samples}
    if extra:
        labels.update(extra)
    return sorted(labels)
