"""Feed-forward intent classifier in PyTorch (roadmap step 4, option 1).

    bag-of-words (V) -> Dropout -> Linear(V,128) -> ReLU -> Dropout
                     -> Linear(128,64) -> ReLU -> Dropout -> Linear(64, n_intents) -> logits
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class IntentNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: Sequence[int] = (128, 64),
                 dropout: float = 0.3, input_dropout: float = 0.1):
        super().__init__()
        layers = [nn.Dropout(input_dropout)] if input_dropout > 0 else []
        previous = input_size
        for size in hidden_sizes:
            layers += [nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits; softmax is applied at inference time
