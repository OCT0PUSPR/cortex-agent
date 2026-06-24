"""Tests for the from-scratch TinyBrain model.

These are skipped automatically when torch is not installed, so the CI path
(MockLLM, no torch) stays green. When torch is present, they verify the core
ML correctness properties: shapes, causality, RoPE rotation, weight tying, and a
single training step that reduces the loss.
"""

from __future__ import annotations

import pytest

from cortex.tinybrain import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_model_forward_and_loss():
    import torch

    from cortex.tinybrain.model import TinyBrain, TinyBrainConfig

    cfg = TinyBrainConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = TinyBrain(cfg)
    idx = torch.randint(0, 64, (3, 16))
    tgt = torch.randint(0, 64, (3, 16))
    logits, loss = model(idx, tgt)
    assert logits.shape == (3, 16, 64)
    assert loss.item() > 0


def test_weight_tying():
    from cortex.tinybrain.model import TinyBrain, TinyBrainConfig

    model = TinyBrain(TinyBrainConfig(vocab_size=32, n_layer=1, n_head=2, n_embd=16))
    assert model.lm_head.weight is model.tok_emb.weight


def test_attention_is_causal():
    import torch

    from cortex.tinybrain.model import TinyBrain, TinyBrainConfig

    cfg = TinyBrainConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = TinyBrain(cfg)
    model.eval()
    idx = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        l1, _ = model(idx, idx)  # full logits via targets path
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 1) % 32  # change only the last token
        l2, _ = model(idx2, idx2)
    # Logits at all positions before the last must be unchanged (causality).
    assert torch.allclose(l1[:, :-1, :], l2[:, :-1, :], atol=1e-5)


def test_rope_preserves_norm():
    import torch

    from cortex.tinybrain.model import apply_rope, build_rope_cache

    cos, sin = build_rope_cache(4, 8, 10000.0, torch.device("cpu"))
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    qr, kr = apply_rope(q, k, cos, sin)
    # A rotation preserves vector norm.
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4)


def test_single_training_step_reduces_loss():
    import torch

    from cortex.tinybrain.model import TinyBrain, TinyBrainConfig

    torch.manual_seed(0)
    cfg = TinyBrainConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = TinyBrain(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    # Overfit a single fixed batch — loss must drop over a few steps.
    idx = torch.randint(0, 32, (4, 16))
    tgt = torch.randint(0, 32, (4, 16))
    _, first = model(idx, tgt)
    for _ in range(20):
        _, loss = model(idx, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first.item()


def test_generation_shape():
    import torch

    from cortex.tinybrain.model import TinyBrain, TinyBrainConfig

    cfg = TinyBrainConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = TinyBrain(cfg)
    idx = torch.randint(0, 32, (1, 4))
    out = model.generate(idx, max_new_tokens=8, temperature=0.8, top_k=10)
    assert out.shape == (1, 12)


def test_tokenizer_char_fallback_roundtrip(tmp_path):
    from cortex.tinybrain.tokenizer import CharTokenizer

    tok = CharTokenizer(vocab=list("hello world"))
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert tok.decode(ids) == "hello"
    path = str(tmp_path / "tok.json")
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.decode(loaded.encode("hello")) == "hello"
