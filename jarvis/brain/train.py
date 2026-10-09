"""Train the intent classifier (roadmap step 4).

    python -m jarvis.brain.train                  # PyTorch - uses MPS on Apple Silicon
    python -m jarvis.brain.train --backend tf     # TensorFlow/Keras - uses Metal if tensorflow-metal is installed
    python -m jarvis.brain.train --device cpu     # force the CPU

Training runs twice: first on 80% of the patterns to measure accuracy on the
held-out 20%, then on everything for the model that gets saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Settings
from ..nlp.preprocess import TextPreprocessor, normalize_wake_word
from .dataset import (Sample, Vocabulary, augment_with_unknowns, build_samples, encode_samples, label_set,
                      load_intents, load_labelled_utterances, stratified_split)

log = logging.getLogger(__name__)

DEFAULTS = dict(epochs=300, lr=1e-3, batch_size=32, hidden=(128, 64), dropout=0.3, input_dropout=0.1,
                label_smoothing=0.05, weight_decay=1e-4, seed=42, augment=True)
TEST_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
# heldout_utterances.json: realistic commands that are not training patterns (used while designing intents.json)
# blind_utterances.json:   written afterwards and never used for tuning - the honest accuracy estimate
EVAL_FILES = (TEST_DIR / "heldout_utterances.json", TEST_DIR / "blind_utterances.json")


def meta_path(model_dir: Path, backend: str) -> Path:
    return Path(model_dir) / f"intent_{backend}.json"


def weights_path(model_dir: Path, backend: str) -> Path:
    return Path(model_dir) / ("intent_torch.pt" if backend == "torch" else "intent_tf.keras")


# --------------------------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------------------------
def fit_torch(X: np.ndarray, y: np.ndarray, num_classes: int, device: str = "auto", epochs: int = 300,
              lr: float = 1e-3, batch_size: int = 32, hidden: Sequence[int] = (128, 64), dropout: float = 0.3,
              input_dropout: float = 0.1, label_smoothing: float = 0.05, weight_decay: float = 1e-4,
              seed: int = 42, verbose: bool = False):
    import torch
    from torch import nn

    from ..device import get_torch_device
    from .model_torch import IntentNet

    torch.manual_seed(seed)
    dev = get_torch_device(device)
    model = IntentNet(X.shape[1], num_classes, hidden, dropout, input_dropout).to(dev)
    X_t = torch.from_numpy(X).to(dev)
    y_t = torch.from_numpy(y).to(dev)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(X_t), generator=generator).to(dev)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(idx)
        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print(f"  epoch {epoch:4d}/{epochs}  loss {epoch_loss / len(X_t):.4f}")
    model.eval()
    return model, dev


def predict_torch(model, X: np.ndarray, device) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        return torch.softmax(logits, dim=-1).cpu().numpy()


# --------------------------------------------------------------------------------------------
# TensorFlow / Keras
# --------------------------------------------------------------------------------------------
def fit_tf(X: np.ndarray, y: np.ndarray, num_classes: int, epochs: int = 300, lr: float = 1e-3,
           batch_size: int = 32, hidden: Sequence[int] = (128, 64), dropout: float = 0.3,
           input_dropout: float = 0.1, label_smoothing: float = 0.05, seed: int = 42, verbose: bool = False,
           **_unused):
    import tensorflow as tf

    from .model_tf import build_keras_model

    tf.keras.utils.set_random_seed(seed)
    model = build_keras_model(X.shape[1], num_classes, hidden, dropout, input_dropout, lr, label_smoothing)
    one_hot = tf.keras.utils.to_categorical(y, num_classes)

    class _Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs):
                print(f"  epoch {epoch + 1:4d}/{epochs}  loss {logs['loss']:.4f}")

    model.fit(X, one_hot, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0, callbacks=[_Progress()])
    return model, None


def predict_tf(model, X: np.ndarray, _device=None) -> np.ndarray:
    import tensorflow as tf

    return tf.nn.softmax(model(X, training=False), axis=-1).numpy()


# --------------------------------------------------------------------------------------------
def _fit(backend: str, X, y, num_classes, **kwargs):
    if backend == "torch":
        return fit_torch(X, y, num_classes, **kwargs)
    kwargs.pop("device", None)
    kwargs.pop("weight_decay", None)
    return fit_tf(X, y, num_classes, **kwargs)


def _predict(backend: str, model, X, device) -> np.ndarray:
    return predict_torch(model, X, device) if backend == "torch" else predict_tf(model, X)


def evaluate(backend: str, model, device, samples: Sequence[Sample], vocab: Vocabulary,
             labels: Sequence[str]) -> Dict:
    X, y = encode_samples(samples, vocab, labels)
    if len(X) == 0:
        return {"accuracy": None, "errors": []}
    pred = _predict(backend, model, X, device).argmax(axis=1)
    errors = [{"text": s.text, "expected": labels[t], "predicted": labels[p]}
              for s, t, p in zip(samples, y, pred) if t != p]
    return {"accuracy": float((pred == y).mean()), "errors": errors}


def train(backend: str = "torch", model_dir: Optional[Path] = None, device: str = "auto",
          val_split: float = 0.2, verbose: bool = True, intents: Optional[List[dict]] = None,
          preprocessor: Optional[TextPreprocessor] = None, eval_files: Sequence[Path] = EVAL_FILES,
          **hyper) -> Dict:
    """Train, evaluate on a held-out split, retrain on all data and save. Returns the metadata."""
    backend = backend.lower()
    if backend not in {"torch", "tf"}:
        raise ValueError("backend must be 'torch' or 'tf'")
    params = {**DEFAULTS, **{k: v for k, v in hyper.items() if v is not None}}
    model_dir = Path(model_dir or Settings.from_env().model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    pre = preprocessor or TextPreprocessor()
    intents = intents or load_intents()
    samples = build_samples(intents, pre)
    labels = label_set(samples)
    fit_kwargs = dict(epochs=params["epochs"], lr=params["lr"], batch_size=params["batch_size"],
                      hidden=tuple(params["hidden"]), dropout=params["dropout"],
                      input_dropout=params["input_dropout"], label_smoothing=params["label_smoothing"],
                      weight_decay=params["weight_decay"], seed=params["seed"], device=device)
    if verbose:
        counts = Counter(s.label for s in samples)
        print(f"Loaded {len(samples)} patterns across {len(labels)} intents "
              f"({', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))})")

    augment = augment_with_unknowns if params["augment"] else list
    metrics: Dict = {}
    if val_split > 0:
        train_s, val_s = stratified_split(samples, val_split, params["seed"])
        vocab = Vocabulary.from_samples(train_s)
        X, y = encode_samples(augment(train_s), vocab, labels)
        if verbose:
            print(f"[1/2] Hold-out run: training on {len(train_s)}, validating on {len(val_s)} ...")
        model, dev = _fit(backend, X, y, len(labels), verbose=False, **fit_kwargs)
        metrics["validation"] = evaluate(backend, model, dev, val_s, vocab, labels)
        if verbose:
            print(f"      validation accuracy: {metrics['validation']['accuracy']:.1%}")
            for err in metrics["validation"]["errors"]:
                print(f"        x {err['text']!r}: expected {err['expected']}, got {err['predicted']}")

    vocab = Vocabulary.from_samples(samples)
    X, y = encode_samples(augment(samples), vocab, labels)
    if verbose:
        print(f"[2/2] Final run on all {len(samples)} patterns ({X.shape[1]}-word vocabulary) ...")
    started = time.perf_counter()
    model, dev = _fit(backend, X, y, len(labels), verbose=verbose, **fit_kwargs)
    elapsed = time.perf_counter() - started
    metrics["train_accuracy"] = evaluate(backend, model, dev, samples, vocab, labels)["accuracy"]
    for eval_file in eval_files or ():
        if not Path(eval_file).exists():
            continue
        commands = [Sample(text, pre.normalize(normalize_wake_word(text)), intent)
                    for text, intent in load_labelled_utterances(Path(eval_file)) if intent in labels]
        result = evaluate(backend, model, dev, commands, vocab, labels)
        metrics[Path(eval_file).stem] = result
        if verbose:
            correct = len(commands) - len(result["errors"])
            print(f"      {Path(eval_file).name}: {correct}/{len(commands)} commands correct ({result['accuracy']:.1%})")

    weights = weights_path(model_dir, backend)
    if backend == "torch":
        import torch

        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights)
    else:
        model.save(weights)

    meta = {
        "backend": backend,
        "labels": labels,
        "vocab": vocab.itos,
        "hidden": list(params["hidden"]),
        "dropout": params["dropout"],
        "input_dropout": params["input_dropout"],
        "preprocessing": {"method": pre.method, "remove_stopwords": pre.remove_stopwords},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev) if dev is not None else "tensorflow",
        "training_seconds": round(elapsed, 2),
        "num_patterns": len(samples),
        "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "metrics": {"validation_accuracy": metrics.get("validation", {}).get("accuracy"),
                    "heldout_accuracy": metrics.get("heldout_utterances", {}).get("accuracy"),
                    "blind_accuracy": metrics.get("blind_utterances", {}).get("accuracy"),
                    "train_accuracy": metrics["train_accuracy"]},
    }
    meta_path(model_dir, backend).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"Saved {weights.name} + {meta_path(model_dir, backend).name} to {model_dir} "
              f"(device: {meta['device']}, {elapsed:.1f}s, train accuracy {metrics['train_accuracy']:.1%})")
    return meta


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train J.A.R.V.I.S.'s intent classifier.")
    parser.add_argument("--backend", choices=["torch", "tf"], default=None,
                        help="torch (default, MPS on Apple Silicon) or tf (Keras, Metal)")
    parser.add_argument("--device", default=None, help="auto | mps | cuda | cpu (PyTorch only)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.2, help="fraction held out for evaluation (0 to skip)")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    train(backend=args.backend or settings.intent_backend, model_dir=args.model_dir or settings.model_dir,
          device=args.device or settings.torch_device, val_split=args.val_split, epochs=args.epochs,
          lr=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
