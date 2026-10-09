"""Hardware acceleration helpers (roadmap step 1).

* PyTorch uses Apple's Metal Performance Shaders backend (``mps``) on Apple
  Silicon, CUDA on NVIDIA machines, and falls back to the CPU otherwise.
* TensorFlow uses the ``tensorflow-metal`` plug-in, which exposes the Apple GPU
  as a regular TensorFlow ``GPU`` device.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict

# PYTORCH_ENABLE_MPS_FALLBACK is set in jarvis/__init__.py so it is in place before torch loads.

log = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_torch_device(preference: str = "auto"):
    """Return the best available ``torch.device``.

    ``preference`` may be ``auto`` (MPS > CUDA > CPU), ``mps``, ``cuda`` or ``cpu``.
    An unavailable explicit choice falls back to the CPU with a warning.
    """
    import torch

    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but not available - using CPU.")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available - using CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def torch_info() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    return {
        "installed": True,
        "version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected": str(get_torch_device()),
    }


def tensorflow_info() -> Dict[str, Any]:
    """Report whether TensorFlow is installed and which GPUs (Metal) it can see."""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
    except ImportError:
        return {"installed": False}
    except Exception as exc:  # e.g. a tensorflow-metal / tensorflow version mismatch
        return {"installed": True, "error": str(exc)}
    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    return {"installed": True, "version": tf.__version__, "gpus": gpus, "metal": bool(gpus) and is_apple_silicon()}
