"""J.A.R.V.I.S. - Just A Rather Very Intelligent System.

A voice-activated personal assistant: wake word -> speech-to-text -> NLP / intent
classification -> action modules (weather, reminders, Q&A, chat) -> text-to-speech.
"""

import os

__version__ = "1.0.0"

# Let PyTorch run the few operators MPS lacks on the CPU instead of failing. PyTorch reads
# this when it initialises, so it is set here - before any jarvis module imports torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
