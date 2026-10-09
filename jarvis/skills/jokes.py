"""A few clean, mostly nerdy jokes. Never repeats one until it has told them all."""

from __future__ import annotations

import random
from typing import List, Optional

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "I told my computer I needed a break, and it said: no problem, I'll go to sleep.",
    "Why did the neural network break up with the dataset? It said it needed more space to generalise.",
    "There are 10 kinds of people in the world: those who understand binary and those who don't.",
    "Why was the robot so tired after the road trip? It had a hard drive.",
    "A SQL query walks into a bar, walks up to two tables and asks: can I join you?",
    "Why don't scientists trust atoms? Because they make up everything.",
    "Why did the gradient go to therapy? It kept vanishing whenever things got deep.",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
    "I would tell you a UDP joke, but you might not get it.",
    "Why did the Python developer wear glasses? Because they couldn't C.",
    "My reinforcement learning agent finally learned to relax. It found the reward in doing nothing.",
]


class JokeTeller:
    def __init__(self, jokes: Optional[List[str]] = None, seed: Optional[int] = None):
        self._jokes = list(jokes or JOKES)
        self._rng = random.Random(seed)
        self._queue: List[str] = []

    def tell(self) -> str:
        if not self._queue:
            self._queue = self._jokes[:]
            self._rng.shuffle(self._queue)
        return self._queue.pop()
