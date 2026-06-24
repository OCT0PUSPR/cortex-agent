"""TinyBrain — a decoder-only Transformer language model, implemented from scratch.

This is a genuine from-scratch implementation (no ``transformers``, no nanoGPT
copy): every component below is written here.

Architecture (LLaMA-style, modernized GPT):

* Token embedding + **weight tying** with the output head.
* **Rotary Positional Embeddings (RoPE)** — implemented here, applied to Q/K.
* **Multi-head causal self-attention** — scaled dot-product with a causal mask,
  using PyTorch SDPA when available and a hand-written fallback otherwise.
* **RMSNorm** — root-mean-square layer norm (implemented here).
* **SwiGLU MLP** — gated feed-forward block.
* Pre-norm residual blocks.

Width (``n_embd``), depth (``n_layer``), heads (``n_head``), context
(``block_size``), and vocab are all configurable via :class:`TinyBrainConfig`.

``torch`` is imported at module top; the package's ``__init__`` guards the
import so the rest of cortex-agent (and the MockLLM CI path) loads without torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyBrainConfig:
    """Hyperparameters for the TinyBrain model."""

    vocab_size: int = 8192
    block_size: int = 256  # max context length
    n_layer: int = 6  # number of transformer blocks (depth)
    n_head: int = 6  # attention heads
    n_embd: int = 384  # embedding / model width
    mlp_ratio: float = 4.0  # hidden-dim multiplier for the MLP (pre-SwiGLU split)
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = True

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, then cast back.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


# --------------------------------------------------------------------------- #
# Rotary positional embeddings (RoPE)
# --------------------------------------------------------------------------- #


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    Returns two tensors of shape ``(seq_len, head_dim)`` where each pair of
    dimensions shares the same rotation frequency (interleaved layout).
    """
    # Frequencies for each of the head_dim/2 rotation planes.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    # (seq_len, head_dim/2)
    freqs = torch.outer(positions, inv_freq)
    # Duplicate each freq to cover the full head_dim (cos/sin applied per pair).
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim into the first (RoPE helper)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    ``q``/``k`` are ``(B, n_head, T, head_dim)``; ``cos``/``sin`` are ``(T, head_dim)``.
    """
    # Broadcast cos/sin over batch and head dims.
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# --------------------------------------------------------------------------- #
# Multi-head causal self-attention
# --------------------------------------------------------------------------- #


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, config: TinyBrainConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Fused QKV projection, then output projection.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # Prefer PyTorch's fused scaled_dot_product_attention when present.
        self._use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to Q and K.
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        if self._use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:  # pragma: no cover - exercised only on old torch
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


# --------------------------------------------------------------------------- #
# SwiGLU MLP
# --------------------------------------------------------------------------- #


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: TinyBrainConfig) -> None:
        super().__init__()
        hidden = int(config.mlp_ratio * config.n_embd)
        # Round to a multiple of 8 for hardware-friendly shapes.
        hidden = ((hidden + 7) // 8) * 8
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #


class Block(nn.Module):
    """A pre-norm transformer block: attention + MLP with residuals."""

    def __init__(self, config: TinyBrainConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# The language model
# --------------------------------------------------------------------------- #


class TinyBrain(nn.Module):
    """Decoder-only Transformer LM."""

    # Declared so static analysis knows these registered buffers are Tensors.
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, config: TinyBrainConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights.
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE cache (registered as a buffer; rebuilt on device move).
        cos, sin = build_rope_cache(config.block_size, config.head_dim, config.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Count trainable parameters (excluding the tied head by default)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_weights:
            # The head shares the embedding; nothing extra to subtract, but the
            # token embedding itself dominates — report the full count.
            return n
        return n

    def _rope(self, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_cos = self.rope_cos
        if rope_cos.device != device or rope_cos.size(0) < T:
            cos, sin = build_rope_cache(
                max(T, self.config.block_size), self.config.head_dim, self.config.rope_theta, device
            )
            self.rope_cos, self.rope_sin = cos, sin
            rope_cos = cos
        return rope_cos[:T], self.rope_sin[:T]

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            idx: ``(B, T)`` token ids.
            targets: optional ``(B, T)`` next-token targets for the loss.

        Returns:
            ``(logits, loss)`` where loss is None when targets is None. During
            training we return logits only for the final position to save memory
            when targets is None at inference.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} exceeds block_size"
        device = idx.device
        cos, sin = self._rope(T, device)

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

        # Inference: only the last position's logits are needed.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` continuations."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if temperature <= 1e-6:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
