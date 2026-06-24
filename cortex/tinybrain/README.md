# TinyBrain — a decoder-only Transformer LM, from scratch

TinyBrain is a small GPT-style language model implemented **from scratch** in
PyTorch inside cortex-agent. No `transformers`, no nanoGPT copy — every
component (rotary embeddings, RMSNorm, multi-head causal attention, SwiGLU MLP,
weight tying, the training loop) is written in this package. It exists to
demonstrate ML fundamentals and to provide a fully self-contained, zero-API
"local brain" backend for the agent.

> The `tokenizers` library is used **only** to learn BPE merges; the model,
> training pipeline, eval, generation, and the encode/decode wiring are ours.

## What's implemented

| Component | File | Notes |
|---|---|---|
| Rotary positional embeddings (RoPE) | `model.py` | `build_rope_cache`, `apply_rope` — implemented + norm-preserving |
| RMSNorm | `model.py` | float32 compute, learnable scale |
| Multi-head causal self-attention | `model.py` | SDPA fast-path + hand-written causal fallback |
| SwiGLU MLP | `model.py` | gated FFN |
| Pre-norm residual blocks, weight tying | `model.py` | configurable width/depth/heads/context |
| BPE tokenizer | `tokenizer.py` | merges via `tokenizers`; encode/decode wiring + char fallback |
| Data pipeline | `data.py` | TinyShakespeare auto-download, memmapped uint16, batching |
| Training loop | `train.py` | AdamW, cosine LR + warmup, grad clip, checkpoint, resume |
| Eval | `eval.py` | held-out perplexity |
| Generation | `generate.py` | top-k sampling |
| Backend adapter | `backend.py` | implements the cortex `LLMBackend` protocol |
| Device select | `device.py` | MPS > CUDA > CPU |

## Quickstart (local, MPS/CPU)

```bash
pip install -r requirements-train.txt   # torch, numpy, tokenizers

# Train a small model (auto-downloads TinyShakespeare):
python -m cortex.tinybrain.train \
  --out-dir .cortex/tinybrain \
  --n-layer 6 --n-head 6 --n-embd 384 --block-size 256 \
  --vocab-size 4096 --batch-size 32 --max-steps 3000

# Evaluate held-out perplexity:
python -m cortex.tinybrain.eval --checkpoint .cortex/tinybrain

# Generate text:
python -m cortex.tinybrain.generate --checkpoint .cortex/tinybrain \
  --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8

# Use it as an agent backend:
cortex --backend tinybrain --model .cortex/tinybrain run "Continue this story"
```

Training is **resumable**: re-run with `--resume` to continue from `last.pt`.

## Scale up (GPU)

The from-scratch code scales — only hyperparameters change. On a single modern
GPU:

```bash
python -m cortex.tinybrain.scale_up \
  --out-dir runs/big \
  --n-layer 12 --n-head 12 --n-embd 768 --block-size 512 \
  --vocab-size 16384 --batch-size 64 --max-steps 50000 \
  --lr 6e-4 --warmup-steps 2000 --compile
```

Point it at a larger corpus with `--corpus-url <url>` (or drop a `corpus.txt`
into the data dir). `--compile` enables `torch.compile` on CUDA.

## Model sizing

`n_embd` (width), `n_layer` (depth), `n_head`, and `block_size` (context) are
all configurable. Parameter count ≈ `vocab*n_embd` (tied embeddings) +
`n_layer * (4*n_embd² attn + 3*mlp_ratio*n_embd² MLP)`.

| Preset | L / H / D | ctx | ≈ params |
|---|---|---|---|
| local | 6 / 6 / 384 | 256 | ~10M |
| scale-up | 12 / 12 / 768 | 512 | ~85M |

## Notes & honesty

TinyBrain is **tiny** — a few-million-parameter model trained for thousands of
steps on a ~1MB corpus. It learns the *style* of the corpus (character names,
line structure, archaic diction) but is not a capable assistant. As an agent
backend it is positioned as a **zero-dependency local demo brain**; tool-calls
are best-effort (it parses a `TOOL: name {json}` convention but rarely emits
it). For real agentic work use the Anthropic or HF backends; use TinyBrain to
show a genuine from-scratch model running inside the same `LLMBackend`
interface.
