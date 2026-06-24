"""Text generation from a trained TinyBrain checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .device import select_device
from .model import TinyBrain, TinyBrainConfig
from .tokenizer import load_tokenizer


def load_model(checkpoint_path: str, device=None):
    """Load a model + tokenizer from a checkpoint directory or file."""
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        ckpt_dir = ckpt_path
        # Prefer the slim, committable inference checkpoint, then the resumable
        # best/last checkpoints (which also carry optimizer state).
        ckpt_file = next(
            (ckpt_dir / name for name in ("model.pt", "best.pt", "last.pt") if (ckpt_dir / name).exists()),
            ckpt_dir / "best.pt",
        )
    else:
        ckpt_file = ckpt_path
        ckpt_dir = ckpt_path.parent

    device = device or select_device("auto")
    # nosec B614: loads only our own locally-produced checkpoints written by
    # this package's training loop, not untrusted third-party files.
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)  # nosec B614
    model_cfg = TinyBrainConfig(**ckpt["model_config"])
    model = TinyBrain(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = load_tokenizer(str(ckpt_dir / "tokenizer.json"))
    return model, tokenizer, device


@torch.no_grad()
def generate_text(
    checkpoint_path: str,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    device=None,
    model_tok=None,
) -> str:
    """Generate a continuation of ``prompt`` from a trained checkpoint."""
    if model_tok is not None:
        model, tokenizer, device = model_tok
    else:
        model, tokenizer, device = load_model(checkpoint_path, device)

    ids = tokenizer.encode(prompt) or [tokenizer.bos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate text from TinyBrain.")
    p.add_argument("--checkpoint", default=".cortex/tinybrain")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()
    print(generate_text(args.checkpoint, args.prompt, args.max_new_tokens, args.temperature, args.top_k))


if __name__ == "__main__":  # pragma: no cover
    main()
