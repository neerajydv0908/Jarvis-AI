"""The same intent classifier in TensorFlow / Keras.

On an Apple Silicon Mac with ``tensorflow-metal`` installed, Keras runs it on
the GPU through Metal. Kept architecture-identical to ``model_torch.IntentNet``
so the two backends can be compared like for like.
"""

from __future__ import annotations

import os
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up chatter


def build_keras_model(input_size: int, num_classes: int, hidden_sizes: Sequence[int] = (128, 64),
                      dropout: float = 0.3, input_dropout: float = 0.1, learning_rate: float = 1e-3,
                      label_smoothing: float = 0.05):
    import tensorflow as tf
    from tensorflow import keras

    model_layers = [keras.Input(shape=(input_size,))]
    if input_dropout > 0:
        model_layers.append(keras.layers.Dropout(input_dropout))
    for size in hidden_sizes:
        model_layers += [keras.layers.Dense(size, activation="relu"), keras.layers.Dropout(dropout)]
    model_layers.append(keras.layers.Dense(num_classes))  # logits
    model = keras.Sequential(model_layers, name="intent_net")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    tf.random.set_seed(42)
    return model
