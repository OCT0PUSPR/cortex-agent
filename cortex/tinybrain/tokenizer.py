"""BPE tokenizer for TinyBrain.

Merge-learning is delegated to the ``tokenizers`` library (byte-level BPE), as
explicitly permitted; the encode/decode wiring, special-token handling, and the
training/IO plumbing are implemented here. The trained tokenizer serializes to a
single JSON file that the training and serving code load.

A pure-stdlib character-level fallback (:class:`CharTokenizer`) is provided so
the package and tests work even if ``tokenizers`` is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

# Special tokens used by the model.
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class BPETokenizer:
    """Byte-level BPE tokenizer (merges trained via ``tokenizers``)."""

    def __init__(self, tokenizer=None) -> None:
        self._tok = tokenizer  # a tokenizers.Tokenizer
        self.bos_id = 0
        self.eos_id = 0
        if tokenizer is not None:
            self._resolve_special_ids()

    def _resolve_special_ids(self) -> None:
        self.bos_id = self._tok.token_to_id(BOS) or 0
        self.eos_id = self._tok.token_to_id(EOS) or 0

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    # -- training ------------------------------------------------------- #
    @classmethod
    def train(cls, text_files: List[str], vocab_size: int = 8192) -> "BPETokenizer":
        """Train a byte-level BPE tokenizer on the given text files."""
        from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
        from tokenizers.models import BPE

        tok = Tokenizer(BPE(unk_token=UNK))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tok.train(text_files, trainer)
        return cls(tok)

    # -- encode / decode wiring ----------------------------------------- #
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token ids, optionally framing with BOS/EOS."""
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text, skipping special tokens."""
        specials = {self.bos_id, self.eos_id}
        # token_to_id for pad/unk too
        for name in (PAD, UNK):
            sid = self._tok.token_to_id(name)
            if sid is not None:
                specials.add(sid)
        clean = [i for i in ids if i not in specials]
        return self._tok.decode(clean)

    # -- IO ------------------------------------------------------------- #
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))


class CharTokenizer:
    """A trivial character-level tokenizer (stdlib-only fallback)."""

    def __init__(self, vocab: Optional[List[str]] = None) -> None:
        self.bos_id = 0
        self.eos_id = 1
        base = [PAD, EOS]  # ids 0,1 reserved-ish (bos==pad slot here)
        chars = vocab or []
        self.itos = base + [c for c in chars if c not in base]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.bos_id = 0
        self.eos_id = 1

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def train(cls, text_files: List[str], vocab_size: int = 0) -> "CharTokenizer":
        chars = set()
        for f in text_files:
            chars |= set(Path(f).read_text(encoding="utf-8", errors="replace"))
        return cls(sorted(chars))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(c, 0) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        specials = {self.bos_id, self.eos_id}
        return "".join(self.itos[i] for i in ids if i not in specials and i < len(self.itos))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"kind": "char", "itos": self.itos}), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.itos = data["itos"]
        tok.stoi = {c: i for i, c in enumerate(tok.itos)}
        return tok


def load_tokenizer(path: str):
    """Load a tokenizer, detecting BPE (tokenizers JSON) vs char fallback."""
    data = Path(path).read_text(encoding="utf-8", errors="replace")
    if '"kind": "char"' in data or '"kind":"char"' in data:
        return CharTokenizer.load(path)
    return BPETokenizer.load(path)


def train_tokenizer(text_files: List[str], vocab_size: int = 8192):
    """Train a BPE tokenizer, falling back to char-level if tokenizers is absent."""
    try:
        import tokenizers  # noqa: F401

        return BPETokenizer.train(text_files, vocab_size=vocab_size)
    except ImportError:  # pragma: no cover
        return CharTokenizer.train(text_files)
