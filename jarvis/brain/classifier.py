"""Load a trained intent model and classify utterances."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..nlp.preprocess import TextPreprocessor, default_preprocessor
from .dataset import Vocabulary
from .train import meta_path, weights_path

log = logging.getLogger(__name__)


class ModelNotTrainedError(FileNotFoundError):
    pass


@dataclass
class Prediction:
    intent: str
    confidence: float
    tokens: List[str] = field(default_factory=list)
    known_tokens: int = 0
    ranking: List[Tuple[str, float]] = field(default_factory=list)


class IntentClassifier:
    def __init__(self, model_dir: Path, backend: str = "torch", device: str = "auto",
                 preprocessor: Optional[TextPreprocessor] = None):
        self.backend = backend.lower()
        meta_file = meta_path(model_dir, self.backend)
        weights = weights_path(model_dir, self.backend)
        if not meta_file.exists() or not weights.exists():
            raise ModelNotTrainedError(
                f"No trained {self.backend} intent model in {model_dir}. "
                f"Run: python -m jarvis.brain.train --backend {self.backend}"
            )
        self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.labels: List[str] = self.meta["labels"]
        self.vocab = Vocabulary(self.meta["vocab"][1:])
        if self.vocab.itos != self.meta["vocab"]:
            raise ValueError("Vocabulary in the model metadata is corrupted; retrain the model.")
        self.preprocessor = preprocessor or default_preprocessor()
        self._device = None
        if self.backend == "torch":
            self._model = self._load_torch(weights, device)
        else:
            self._model = self._load_tf(weights)

    def _load_torch(self, weights: Path, device: str):
        import torch

        from ..device import get_torch_device
        from .model_torch import IntentNet

        self._device = get_torch_device(device)
        model = IntentNet(len(self.vocab), len(self.labels), self.meta["hidden"], self.meta["dropout"],
                          self.meta["input_dropout"])
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        return model.to(self._device).eval()

    @staticmethod
    def _load_tf(weights: Path):
        from tensorflow import keras

        return keras.models.load_model(weights)

    def probabilities(self, vector: np.ndarray) -> np.ndarray:
        batch = vector[None, :]
        if self.backend == "torch":
            import torch

            with torch.no_grad():
                logits = self._model(torch.from_numpy(batch).to(self._device))
                return torch.softmax(logits, dim=-1)[0].cpu().numpy()
        import tensorflow as tf

        return tf.nn.softmax(self._model(batch, training=False), axis=-1).numpy()[0]

    def predict(self, text: str) -> Prediction:
        tokens = self.preprocessor.normalize(text)
        known = self.vocab.known_count(tokens)
        if not tokens:
            return Prediction("general_chat", 0.0, tokens, 0, [])
        probs = self.probabilities(self.vocab.encode(tokens))
        order = np.argsort(probs)[::-1]
        ranking = [(self.labels[i], float(probs[i])) for i in order[:3]]
        return Prediction(ranking[0][0], ranking[0][1], tokens, known, ranking)
