"""Tests for the TinyBrain serving path: backend adapter, data, device, eval.

Complements ``test_tinybrain.py`` (which covers the model math). These exercise
the pieces that turn a trained checkpoint into a working ``LLMBackend``: the BPE
tokenizer, memmapped batching, device selection, held-out perplexity eval, and
the ``TinyBrainBackend.complete``/``stream`` adapters via a micro checkpoint.

Skipped wholesale when torch is unavailable so the no-torch CI path stays green.
"""

from __future__ import annotations

import pytest

from cortex.tinybrain import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")

if torch_available():
    import torch


# --------------------------------------------------------------------------- #
# RMSNorm + device
# --------------------------------------------------------------------------- #


def test_rmsnorm_unit_rms():
    from cortex.tinybrain.model import RMSNorm

    norm = RMSNorm(8)
    x = torch.randn(4, 8) * 7.0
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_select_device_explicit_and_auto():
    from cortex.tinybrain.device import device_name, select_device

    assert select_device("cpu").type == "cpu"
    assert "cpu" in device_name(select_device("cpu"))
    assert select_device("auto").type in {"mps", "cuda", "cpu"}


# --------------------------------------------------------------------------- #
# Data: batching + guards
# --------------------------------------------------------------------------- #


def test_token_dataset_batch_shapes_and_shift(tmp_path):
    import numpy as np

    from cortex.tinybrain.data import TokenDataset

    p = tmp_path / "train.bin"
    np.arange(0, 600, dtype=np.uint16).tofile(str(p))
    ds = TokenDataset(str(p), block_size=16)
    assert len(ds) == 600
    x, y = ds.get_batch(8, torch.device("cpu"))
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert int(y[0, 0]) == int(x[0, 0]) + 1  # next-token shift


def test_token_dataset_too_small(tmp_path):
    import numpy as np

    from cortex.tinybrain.data import TokenDataset

    p = tmp_path / "t.bin"
    np.arange(0, 5, dtype=np.uint16).tofile(str(p))
    with pytest.raises(ValueError):
        TokenDataset(str(p), block_size=16)


def test_encode_corpus_writes_train_val(tmp_path):
    from cortex.tinybrain.data import encode_corpus
    from cortex.tinybrain.tokenizer import CharTokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabcabc " * 100, encoding="utf-8")
    tok = CharTokenizer.train([str(corpus)])
    train_bin, val_bin, total = encode_corpus(str(corpus), tok, str(tmp_path))
    assert (tmp_path / "train.bin").exists()
    assert (tmp_path / "val.bin").exists()
    assert total > 0


def test_download_corpus_offline_fallback(tmp_path):
    from cortex.tinybrain.data import download_corpus

    dest = tmp_path / "c.txt"
    # An unreachable URL forces the bundled-fallback path.
    path = download_corpus(str(dest), url="http://127.0.0.1:1/nope.txt")
    assert dest.exists()
    assert len(dest.read_text(encoding="utf-8")) > 0
    # Second call short-circuits on the existing file.
    assert download_corpus(str(dest)) == path


# --------------------------------------------------------------------------- #
# BPE tokenizer
# --------------------------------------------------------------------------- #


def test_bpe_tokenizer_train_and_roundtrip(tmp_path):
    pytest.importorskip("tokenizers")
    from cortex.tinybrain.tokenizer import BPETokenizer, load_tokenizer

    corpus = tmp_path / "c.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 300, encoding="utf-8")
    tok = BPETokenizer.train([str(corpus)], vocab_size=400)
    ids = tok.encode("the quick brown fox", add_bos=True, add_eos=True)
    assert ids and isinstance(ids, list)
    assert "quick" in tok.decode(ids)
    out = tmp_path / "bpe.json"
    tok.save(str(out))
    reloaded = load_tokenizer(str(out))
    assert reloaded.vocab_size == tok.vocab_size
    assert "fox" in reloaded.decode(reloaded.encode("the brown fox"))


# --------------------------------------------------------------------------- #
# Micro checkpoint -> backend + eval
# --------------------------------------------------------------------------- #


@pytest.fixture
def micro_ckpt(tmp_path):
    """Build a tiny trained checkpoint dir the serving code can load."""
    from cortex.tinybrain.model import TinyBrain, TinyBrainConfig
    from cortex.tinybrain.tokenizer import CharTokenizer
    from cortex.tinybrain.train import save_model_only

    out = tmp_path / "ckpt"
    out.mkdir()
    corpus = out / "corpus.txt"
    corpus.write_text("To be, or not to be, that is the question. " * 80, encoding="utf-8")
    tok = CharTokenizer.train([str(corpus)])
    tok.save(str(out / "tokenizer.json"))

    cfg = TinyBrainConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = TinyBrain(cfg)
    # A couple of optimization steps so generation is not pure noise.
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    ids = torch.tensor([tok.encode(corpus.read_text(encoding="utf-8"))], dtype=torch.long)
    seq = ids[0, : 16 * 8].reshape(8, 16)
    for _ in range(15):
        _, loss = model(seq[:, :16], seq[:, :16])
        opt.zero_grad()
        loss.backward()
        opt.step()
    save_model_only(str(out / "model.pt"), model, 15, cfg)
    return str(out)


def test_load_model_prefers_slim_checkpoint(micro_ckpt):
    from cortex.tinybrain.generate import load_model

    model, tok, device = load_model(micro_ckpt, torch.device("cpu"))
    assert model.config.block_size == 16
    assert tok.vocab_size > 0
    assert device.type == "cpu"


def test_generate_text_produces_string(micro_ckpt):
    from cortex.tinybrain.generate import generate_text

    out = generate_text(micro_ckpt, prompt="To be", max_new_tokens=20, device=torch.device("cpu"))
    assert isinstance(out, str) and len(out) > 0


def test_backend_complete_returns_response(micro_ckpt):
    from cortex.llm.base import LLMBackend, LLMResponse, Message
    from cortex.tinybrain.backend import TinyBrainBackend

    backend = TinyBrainBackend(checkpoint_path=micro_ckpt, max_new_tokens=20, device="cpu")
    assert isinstance(backend, LLMBackend)
    assert backend.name == "tinybrain"
    resp = backend.complete([Message(role="user", content="To be")], max_tokens=40)
    assert isinstance(resp, LLMResponse)
    assert resp.model == "tinybrain"
    assert resp.usage["output_tokens"] >= 1
    assert resp.stop_reason in {"end_turn", "tool_use"}


def test_backend_stream_yields_chunks(micro_ckpt):
    from cortex.llm.base import Message
    from cortex.tinybrain.backend import TinyBrainBackend

    backend = TinyBrainBackend(checkpoint_path=micro_ckpt, max_new_tokens=12, device="cpu")
    chunks = list(backend.stream([Message(role="user", content="To be")], max_tokens=40))
    assert isinstance(chunks, list)  # it ran end to end as a generator


def test_backend_missing_checkpoint_errors(tmp_path):
    from cortex.tinybrain.backend import TinyBrainBackend

    backend = TinyBrainBackend(checkpoint_path=str(tmp_path / "nope"), device="cpu")
    from cortex.llm.base import Message

    with pytest.raises((RuntimeError, FileNotFoundError)):
        backend.complete([Message(role="user", content="hi")])


def test_get_backend_tinybrain(micro_ckpt):
    from cortex.llm import get_backend

    backend = get_backend("tinybrain", model=micro_ckpt, device="cpu")
    assert backend.name == "tinybrain"


def test_eval_perplexity(micro_ckpt):
    from cortex.tinybrain.data import encode_corpus
    from cortex.tinybrain.eval import evaluate
    from cortex.tinybrain.tokenizer import load_tokenizer

    tok = load_tokenizer(micro_ckpt + "/tokenizer.json")
    encode_corpus(micro_ckpt + "/corpus.txt", tok, micro_ckpt)
    metrics = evaluate(micro_ckpt, data_dir=micro_ckpt, eval_iters=3, device=torch.device("cpu"))
    assert metrics["val_loss"] > 0
    assert metrics["perplexity"] > 1.0
