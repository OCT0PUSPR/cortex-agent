"""Scale-up training entry point for a larger TinyBrain on a GPU.

Same from-scratch model and pipeline as :mod:`cortex.tinybrain.train`, but with
defaults tuned for a single modern GPU (e.g. an A100/4090): a wider/deeper model,
larger context and batch, a bigger BPE vocab, and a longer schedule. Use a
larger corpus (e.g. WikiText-103 or your own text) via ``--corpus-url`` or by
placing a ``corpus.txt`` in the data dir.

Example (single GPU)::

    python -m cortex.tinybrain.scale_up \\
        --out-dir runs/big --max-steps 50000 --batch-size 64 \\
        --n-layer 12 --n-head 12 --n-embd 768 --block-size 512 \\
        --vocab-size 16384 --lr 6e-4

This is the *same code path* as the small local run — only the hyperparameters
change — which is the point: the from-scratch implementation scales.
"""

from __future__ import annotations

from pathlib import Path

from .train import TrainConfig, train


def _build_cli():
    import argparse

    p = argparse.ArgumentParser(description="Scale-up TinyBrain training (GPU).")
    p.add_argument("--out-dir", default="runs/big")
    p.add_argument("--corpus-url", default=None, help="Override the corpus download URL.")
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (CUDA).")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = _build_cli().parse_args()
    cfg = TrainConfig(
        out_dir=args.out_dir,
        data_dir=str(Path(args.out_dir) / "data"),
        corpus_url=args.corpus_url or TrainConfig.corpus_url,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        vocab_size=args.vocab_size,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        device=args.device,
        compile_model=args.compile,
    )
    result = train(cfg, resume=args.resume)
    print("\n=== scale-up training complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":  # pragma: no cover
    main()
