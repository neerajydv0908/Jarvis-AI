"""Fine-tune DistilGPT-2 on the Cornell Movie-Dialogs Corpus (roadmap step 4, option 2).

    python scripts/finetune_chat.py --max-pairs 5000            # quick trial run
    python scripts/finetune_chat.py                             # 50k pairs, 1 epoch (default)
    python scripts/finetune_chat.py --base-model microsoft/DialoGPT-small

The corpus is downloaded to data/cornell/ the first time (or unzip it there
yourself). The model is saved to models/chat/, where Jarvis picks it up
automatically for small talk. Runs on the Apple GPU (MPS) when available.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jarvis.brain import cornell  # noqa: E402
from jarvis.brain.chat_training import fine_tune, format_prompt  # noqa: E402
from jarvis.device import get_torch_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a small GPT-2 on movie dialogue for Jarvis's small talk.")
    parser.add_argument("--base-model", default="distilgpt2", help="Hugging Face model id or local path")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "cornell")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "chat")
    parser.add_argument("--max-pairs", type=int, default=50_000, help="0 = use every pair (~220k)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=25, help="skip lines longer than this")
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("transformers is not installed: pip install transformers")

    if not cornell.find_corpus(args.data_dir):
        print(f"Corpus not found in {args.data_dir}; downloading it ...")
        try:
            cornell.download(args.data_dir)
        except RuntimeError as exc:
            sys.exit(f"{exc}\n\nDownload the corpus manually (ConvoKit 'movie-corpus' or the original "
                     f"cornell_movie_dialogs_corpus.zip) and unzip it into {args.data_dir}.")

    pairs = cornell.load_pairs(args.data_dir, max_words=args.max_words)
    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"Using {len(pairs):,} dialogue pairs, e.g. {pairs[0]}")

    device = get_torch_device(args.device)
    print(f"Loading {args.base_model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    started = time.time()
    metrics = fine_tune(model, tokenizer, pairs, device, epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, max_length=args.max_length, seed=args.seed)
    metrics["minutes"] = round((time.time() - started) / 60, 1)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    info = {"base_model": args.base_model, "device": str(device), "pairs": len(pairs), "epochs": args.epochs, **metrics}
    (args.output / "training_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    # quick sanity check
    import torch

    model.eval()
    for message in ["How are you today?", "What do you want to do tonight?", "I think I'm in love."]:
        inputs = tokenizer(format_prompt(message), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_p=0.92, temperature=0.8,
                                 pad_token_id=tokenizer.pad_token_id)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"User: {message}\nJarvis: {reply.splitlines()[0].strip() if reply.strip() else '(empty)'}\n")
    print(f"Saved to {args.output}. Jarvis will use it for small talk on the next start.")


if __name__ == "__main__":
    main()
