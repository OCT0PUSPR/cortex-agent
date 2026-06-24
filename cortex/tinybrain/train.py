"""Training pipeline for TinyBrain.

Implements a real training loop:

* AdamW optimizer with decoupled weight decay (no decay on norms/biases),
* cosine learning-rate schedule with linear warmup,
* gradient clipping,
* periodic train/val evaluation (loss + perplexity),
* checkpointing (best + last) with full resumability (model + optimizer + step),
* device auto-select (MPS > CUDA > CPU).

Run via ``python -m cortex.tinybrain.train`` (see ``__main__`` block) or call
:func:`train` directly.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch

from .data import TINYSHAKESPEARE_URL, TokenDataset, prepare_data
from .device import select_device
from .model import TinyBrain, TinyBrainConfig
from .tokenizer import train_tokenizer


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    out_dir: str = ".cortex/tinybrain"
    data_dir: str = ".cortex/tinybrain/data"
    corpus_url: str = TINYSHAKESPEARE_URL
    vocab_size: int = 4096

    # model
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

    # optimization
    batch_size: int = 32
    max_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # logging / eval / checkpoint
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 50
    seed: int = 1337

    device: str = "auto"
    compile_model: bool = False

    log: List[Dict[str, float]] = field(default_factory=list)


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def configure_optimizer(model: TinyBrain, cfg: TrainConfig, device) -> torch.optim.Optimizer:
    """AdamW with weight decay on 2D params only (not norms/embeddings biases)."""
    decay: List[torch.Tensor] = []
    no_decay: List[torch.Tensor] = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device.type == "cuda"
    try:
        return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=fused)
    except (TypeError, RuntimeError):  # fused unsupported
        return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def estimate_loss(
    model: TinyBrain, train_ds: TokenDataset, val_ds: TokenDataset, cfg: TrainConfig, device
) -> Dict[str, float]:
    """Estimate mean train/val loss over ``eval_iters`` batches."""
    model.eval()
    out: Dict[str, float] = {}
    for name, ds in (("train", train_ds), ("val", val_ds)):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: str, model: TinyBrain, optimizer, step: int, cfg: TrainConfig, model_cfg: TinyBrainConfig
) -> None:
    """Persist a resumable checkpoint (model + optimizer state, for --resume)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "train_config": asdict(cfg),
            "model_config": asdict(model_cfg),
        },
        path,
    )


def save_model_only(path: str, model: TinyBrain, step: int, model_cfg: TinyBrainConfig) -> int:
    """Persist a slim, inference-only checkpoint (weights only, no optimizer).

    This is the small "proof" artifact suitable for committing — it omits the
    AdamW moment buffers that bloat the resumable checkpoint roughly 3x. The
    weights are moved to CPU for a portable, device-agnostic file. Returns the
    written file size in bytes.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Unwrap a possibly torch.compile-d model (it stashes the original under
    # `_orig_mod`); fall back to the model itself. cast keeps mypy happy whether
    # or not torch's types are available (CI runs mypy without torch installed).
    base = cast(TinyBrain, getattr(model, "_orig_mod", model))
    state = {k: v.detach().to("cpu") for k, v in base.state_dict().items()}
    torch.save(
        {"model": state, "step": step, "model_config": asdict(model_cfg), "slim": True},
        path,
    )
    return Path(path).stat().st_size


def train(cfg: Optional[TrainConfig] = None, resume: bool = False) -> Dict[str, Any]:
    """Run the full training loop; return final metrics.

    Returns a dict with ``val_loss``, ``val_perplexity``, ``train_loss``,
    ``steps``, and the path to the best checkpoint.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    print(f"[tinybrain] device = {device}")

    # --- tokenizer + data --------------------------------------------- #
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = str(out_dir / "tokenizer.json")

    # Ensure the corpus exists, then train (or load) the tokenizer on it.
    from .data import download_corpus

    corpus_path = download_corpus(str(Path(cfg.data_dir) / "corpus.txt"), url=cfg.corpus_url)
    if resume and Path(tok_path).exists():
        from .tokenizer import load_tokenizer

        tokenizer = load_tokenizer(tok_path)
        print(f"[tinybrain] loaded tokenizer (vocab={tokenizer.vocab_size})")
    else:
        print("[tinybrain] training BPE tokenizer ...")
        tokenizer = train_tokenizer([corpus_path], vocab_size=cfg.vocab_size)
        tokenizer.save(tok_path)
        print(f"[tinybrain] tokenizer trained (vocab={tokenizer.vocab_size})")

    train_ds, val_ds, _ = prepare_data(cfg.data_dir, tokenizer, cfg.block_size, url=cfg.corpus_url)
    print(f"[tinybrain] train tokens={len(train_ds)} val tokens={len(val_ds)}")

    # --- model -------------------------------------------------------- #
    model_cfg = TinyBrainConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = TinyBrain(model_cfg).to(device)
    n_params = model.num_params()
    print(f"[tinybrain] model params = {n_params / 1e6:.2f}M (L={cfg.n_layer} H={cfg.n_head} D={cfg.n_embd})")

    optimizer = configure_optimizer(model, cfg, device)
    start_step = 0
    best_val = float("inf")
    best_path = str(out_dir / "best.pt")
    last_path = str(out_dir / "last.pt")

    if resume and Path(last_path).exists():
        # nosec B614: resumes from our own checkpoint written below.
        ckpt = torch.load(last_path, map_location=device)  # nosec B614
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"[tinybrain] resumed from step {start_step}")

    if cfg.compile_model and hasattr(torch, "compile"):  # pragma: no cover
        model = torch.compile(model)  # type: ignore[assignment]

    # --- loop --------------------------------------------------------- #
    model.train()
    t0 = time.time()
    last_metrics: Dict[str, float] = {}
    for step in range(start_step, cfg.max_steps + 1):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodic evaluation + checkpoint.
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            metrics = estimate_loss(model, train_ds, val_ds, cfg, device)
            val_ppl = math.exp(min(metrics["val"], 20))
            elapsed = time.time() - t0
            print(
                f"[tinybrain] step {step:5d} | train {metrics['train']:.4f} | "
                f"val {metrics['val']:.4f} | ppl {val_ppl:.2f} | lr {lr:.2e} | {elapsed:.1f}s"
            )
            cfg.log.append(
                {
                    "step": step,
                    "train_loss": metrics["train"],
                    "val_loss": metrics["val"],
                    "val_ppl": val_ppl,
                    "lr": lr,
                }
            )
            last_metrics = {
                "val_loss": metrics["val"],
                "val_perplexity": val_ppl,
                "train_loss": metrics["train"],
            }
            save_checkpoint(last_path, model, optimizer, step, cfg, model_cfg)
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                save_checkpoint(best_path, model, optimizer, step, cfg, model_cfg)

        if step == cfg.max_steps:
            break

        # Training step.
        x, y = train_ds.get_batch(cfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            print(f"[tinybrain]   step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

    # Persist the training log alongside checkpoints.
    import json

    (out_dir / "train_log.json").write_text(json.dumps(cfg.log, indent=2), encoding="utf-8")

    # Export a slim, inference-only "proof" checkpoint from the best weights.
    slim_path = str(out_dir / "model.pt")
    slim_bytes = 0
    if Path(best_path).exists():
        # nosec B614: re-loads our own best checkpoint to write the slim model.
        best_ckpt = torch.load(best_path, map_location="cpu")  # nosec B614
        slim_model = TinyBrain(TinyBrainConfig(**best_ckpt["model_config"]))
        slim_model.load_state_dict(best_ckpt["model"])
        slim_bytes = save_model_only(slim_path, slim_model, best_ckpt["step"], model_cfg)
        print(f"[tinybrain] slim checkpoint -> {slim_path} ({slim_bytes / 1e6:.1f} MB)")

    result = {
        **last_metrics,
        "steps": cfg.max_steps,
        "best_checkpoint": best_path,
        "slim_checkpoint": slim_path,
        "slim_checkpoint_mb": round(slim_bytes / 1e6, 2),
        "best_val_loss": best_val,
        "best_val_perplexity": math.exp(min(best_val, 20)),
        "params_millions": n_params / 1e6,
        "device": str(device),
    }
    print(f"[tinybrain] DONE. best val loss {best_val:.4f} | best ppl {result['best_val_perplexity']:.2f}")
    return result


def _build_cli():
    import argparse

    p = argparse.ArgumentParser(description="Train TinyBrain from scratch.")
    p.add_argument("--out-dir", default=".cortex/tinybrain")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = _build_cli().parse_args()
    cfg = TrainConfig(
        out_dir=args.out_dir,
        data_dir=str(Path(args.out_dir) / "data"),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        vocab_size=args.vocab_size,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        dropout=args.dropout,
        device=args.device,
    )
    train(cfg, resume=args.resume)


if __name__ == "__main__":  # pragma: no cover
    main()
