"""Corpus loading, tokenization, and batching for TinyBrain training.

Downloads TinyShakespeare (a tiny, license-clean corpus) on first use, or uses
a bundled fallback sample if offline. The tokenized corpus is memmapped as a
uint16 array of token ids; batches are sampled as contiguous ``block_size``
windows for next-token prediction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# A small bundled fallback so training works fully offline.
_FALLBACK_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die—to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep, perchance to dream—ay, there's the rub:\n"
) * 200


def download_corpus(dest: str, url: str = TINYSHAKESPEARE_URL) -> str:
    """Download the training corpus to ``dest`` (or write the fallback)."""
    path = Path(dest)
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import httpx

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            path.write_text(resp.text, encoding="utf-8")
    except Exception:
        # Offline / blocked: write the bundled fallback so training still runs.
        path.write_text(_FALLBACK_TEXT, encoding="utf-8")
    return str(path)


def encode_corpus(corpus_path: str, tokenizer, out_dir: str) -> Tuple[str, str, int]:
    """Tokenize a corpus and write train/val uint16 ``.bin`` files.

    Returns ``(train_bin, val_bin, total_tokens)``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    ids = np.array(tokenizer.encode(text), dtype=np.uint32)

    # 90/10 train/val split.
    n = len(ids)
    split = int(n * 0.9)
    train_ids = ids[:split].astype(np.uint16)
    val_ids = ids[split:].astype(np.uint16)

    train_bin = str(out / "train.bin")
    val_bin = str(out / "val.bin")
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)
    return train_bin, val_bin, n


class TokenDataset:
    """Memmapped token corpus with random contiguous-window batch sampling."""

    def __init__(self, bin_path: str, block_size: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) <= block_size + 1:
            raise ValueError(f"corpus at {bin_path} is too small ({len(self.data)} tokens) for block_size {block_size}")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, batch_size: int, device, generator=None):
        """Sample a random batch of (x, y) next-token pairs.

        ``x`` and ``y`` are ``(batch_size, block_size)`` int64 tensors on
        ``device``; ``y`` is ``x`` shifted by one position.
        """
        import torch

        max_start = len(self.data) - self.block_size - 1
        if generator is not None:
            ix = torch.randint(0, max_start, (batch_size,), generator=generator)
        else:
            ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        # Pin + non-blocking transfer for CUDA; plain .to() elsewhere.
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


def prepare_data(
    data_dir: str, tokenizer, block_size: int, url: str = TINYSHAKESPEARE_URL
) -> Tuple["TokenDataset", "TokenDataset", str]:
    """End-to-end: download (if needed), tokenize, and return train/val datasets."""
    corpus = download_corpus(os.path.join(data_dir, "corpus.txt"), url=url)
    train_bin, val_bin, _ = encode_corpus(corpus, tokenizer, data_dir)
    return (
        TokenDataset(train_bin, block_size),
        TokenDataset(val_bin, block_size),
        corpus,
    )
