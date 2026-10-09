"""Fine-tuning loop for the generative chat model (used by scripts/finetune_chat.py).

Each (prompt, response) pair becomes::

    User: <prompt>
    Jarvis: <response><|endoftext|>

and the loss is computed on the response tokens only (prompt tokens get label
-100), so the model learns to *answer* rather than to imitate the user.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

USER_TAG = "User:"
BOT_TAG = "Jarvis:"


def format_prompt(message: str, history: Sequence[Tuple[str, str]] = ()) -> str:
    turns = [f"{USER_TAG} {u}\n{BOT_TAG} {b}" for u, b in history]
    turns.append(f"{USER_TAG} {message}\n{BOT_TAG}")
    return "\n".join(turns)


class DialoguePairs(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, str]], tokenizer, max_length: int = 64):
        self.items: List[Tuple[List[int], List[int]]] = []
        eos = tokenizer.eos_token_id
        for prompt, response in pairs:
            prompt_ids = tokenizer(format_prompt(prompt), add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(" " + response, add_special_tokens=False)["input_ids"] + [eos]
            ids = (prompt_ids + response_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
            if any(label != -100 for label in labels):
                self.items.append((ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def make_collate(pad_id: int):
    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        input_ids, attention, labels = [], [], []
        for ids, lab in batch:
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention),
                "labels": torch.tensor(labels)}
    return collate


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        batches += 1
    model.train()
    return total / max(batches, 1)


def fine_tune(model, tokenizer, pairs: Sequence[Tuple[str, str]], device, epochs: int = 1,
              batch_size: int = 16, lr: float = 5e-5, max_length: int = 64, warmup_ratio: float = 0.05,
              val_fraction: float = 0.02, log_every: int = 50, seed: int = 42) -> Dict[str, float]:
    """Fine-tune ``model`` in place. Returns validation loss/perplexity before and after."""
    random.seed(seed)
    torch.manual_seed(seed)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pairs = list(pairs)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction)) if len(pairs) > 10 else 0
    train_set = DialoguePairs(pairs[n_val:], tokenizer, max_length)
    val_set = DialoguePairs(pairs[:n_val], tokenizer, max_length)
    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate)

    model.to(device)
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, epochs * len(train_loader))
    warmup = max(1, int(total_steps * warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min((step + 1) / warmup, max(0.0, (total_steps - step) / max(1, total_steps - warmup))))

    metrics: Dict[str, float] = {}
    if len(val_set):
        metrics["val_loss_before"] = evaluate_loss(model, val_loader, device)
    step, started, running = 0, time.time(), 0.0
    for epoch in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()
            step += 1
            running += loss.item()
            if log_every and step % log_every == 0:
                elapsed = time.time() - started
                eta = elapsed / step * (total_steps - step)
                print(f"  epoch {epoch + 1} step {step}/{total_steps}  loss {running / log_every:.3f}  "
                      f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left)")
                running = 0.0
    if len(val_set):
        metrics["val_loss_after"] = evaluate_loss(model, val_loader, device)
        for key in ("before", "after"):
            metrics[f"val_perplexity_{key}"] = math.exp(min(metrics[f"val_loss_{key}"], 20))
    metrics["steps"] = step
    metrics["train_pairs"] = len(train_set)
    model.eval()
    return metrics
