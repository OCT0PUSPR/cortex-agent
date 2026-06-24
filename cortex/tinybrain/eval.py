"""Held-out perplexity evaluation for a trained TinyBrain checkpoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch

from .data import TokenDataset, encode_corpus
from .generate import load_model


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_dir: str = ".cortex/tinybrain/data",
    eval_iters: int = 200,
    batch_size: int = 32,
    device=None,
) -> Dict[str, float]:
    """Compute held-out validation loss and perplexity.

    Re-tokenizes the corpus with the checkpoint's tokenizer if needed, then
    averages the loss over ``eval_iters`` random batches of the validation split.
    """
    model, tokenizer, device = load_model(checkpoint_path, device)
    block_size = model.config.block_size

    val_bin = Path(data_dir) / "val.bin"
    if not val_bin.exists():
        corpus = Path(data_dir) / "corpus.txt"
        encode_corpus(str(corpus), tokenizer, data_dir)

    val_ds = TokenDataset(str(val_bin), block_size)

    losses = torch.zeros(eval_iters)
    model.eval()
    for k in range(eval_iters):
        x, y = val_ds.get_batch(batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    mean_loss = losses.mean().item()
    ppl = math.exp(min(mean_loss, 20))
    return {
        "val_loss": mean_loss,
        "perplexity": ppl,
        "eval_iters": eval_iters,
        "block_size": block_size,
        "device": str(device),
    }


def main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Evaluate TinyBrain perplexity.")
    p.add_argument("--checkpoint", default=".cortex/tinybrain")
    p.add_argument("--data-dir", default=".cortex/tinybrain/data")
    p.add_argument("--eval-iters", type=int, default=200)
    args = p.parse_args()
    result = evaluate(args.checkpoint, args.data_dir, args.eval_iters)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
