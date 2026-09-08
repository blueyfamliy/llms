"""Training loop.

Run it with a config file:

    python -m llms.train --config configs/tiny.yaml

Any field can be overridden on the command line with dotted keys, e.g.
``--train.lr=1e-3 --model.n_layer=12``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .config import Config, TrainConfig, config_to_dict, load_config
from .data import TokenDataset, read_meta
from .model import GPT

# --------------------------------------------------------------------- runtime


def resolve_device(requested: str) -> torch.device:
    """Pick the best available device when the config says ``auto``."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Pick the autocast dtype.

    bfloat16 is preferred where supported: it has float32's exponent range, so
    unlike float16 it needs no loss scaling. MPS and CPU fall back to float32,
    where autocast buys little and can be numerically worse.
    """
    if requested != "auto":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            requested
        ]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    datasets: dict[str, TokenDataset],
    cfg: TrainConfig,
    device: torch.device,
    autocast_ctx,
) -> dict[str, float]:
    """Average the loss over ``eval_iters`` batches of each split."""
    model.eval()
    out: dict[str, float] = {}
    for split, dataset in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = dataset.get_batch(cfg.batch_size, device)
            with autocast_ctx:
                _, loss, _ = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ----------------------------------------------------------------- checkpoints


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: Config,
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_to_dict(config),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


# ------------------------------------------------------------------------ main


def train(config: Config, resume: bool = False) -> GPT:
    cfg = config.train
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
        # TF32 costs a little precision in matmuls for a large speedup on Ampere+.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype is not torch.float32 and device.type in ("cuda", "cpu")
        else nullcontext()
    )
    # float16 needs loss scaling to keep small gradients from flushing to zero.
    use_scaler = dtype is torch.float16 and device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_scaler)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_scaler)
    )

    # The dataset's vocabulary is authoritative -- a mismatch here silently
    # produces an unusable model, so adopt it and say so.
    meta = read_meta(config.data.data_dir)
    if config.model.vocab_size != int(meta["vocab_size"]):
        print(
            f"note: setting model.vocab_size to {meta['vocab_size']} from "
            f"{config.data.data_dir}/meta.json (config said {config.model.vocab_size})"
        )
        config.model.vocab_size = int(meta["vocab_size"])

    datasets = {
        split: TokenDataset(config.data.data_dir, split, config.model.block_size)
        for split in ("train", "val")
    }

    model = GPT(config.model).to(device)
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device.type
    )

    out_dir = Path(cfg.out_dir)
    ckpt_path = out_dir / "ckpt.pt"
    start_step = 0
    best_val_loss = float("inf")

    if resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume given but {ckpt_path} does not exist")
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {ckpt_path} at step {start_step}")

    if cfg.compile:
        print("compiling the model (this takes a minute on the first run)...")
        model = torch.compile(model)  # type: ignore[assignment]

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * config.model.block_size
    print(
        f"device={device} dtype={dtype} "
        f"params={raw_model.num_parameters() / 1e6:.2f}M (non-embedding) "
        f"tokens/step={tokens_per_step:,} "
        f"train_tokens={len(datasets['train']):,}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config_to_dict(config), indent=2))

    train_ds = datasets["train"]
    x, y = train_ds.get_batch(cfg.batch_size, device)  # prefetch the first batch
    t0 = time.time()

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _micro_step in range(cfg.grad_accum_steps):
            with autocast_ctx:
                _, loss, _ = model(x, targets=y)
                # Average over accumulation steps so the effective batch behaves
                # like one large batch rather than scaling the gradient by N.
                loss = loss / cfg.grad_accum_steps
            # Fetch the next batch while the backward pass runs.
            x, y = train_ds.get_batch(cfg.batch_size, device)
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            full_loss = loss.item() * cfg.grad_accum_steps
            print(
                f"step {step:>6} | loss {full_loss:.4f} | lr {lr:.2e} | "
                f"{dt * 1000 / max(1, cfg.log_interval):.0f} ms/step"
            )

        is_last = step == cfg.max_steps - 1
        if (step > 0 and step % cfg.eval_interval == 0) or is_last:
            losses = estimate_loss(model, datasets, cfg, device, autocast_ctx)
            print(
                f"step {step:>6} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | val ppl {math.exp(min(losses['val'], 20)):.2f}"
            )
            if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses["val"])
                save_checkpoint(ckpt_path, raw_model, optimizer, config, step, best_val_loss)
                print(f"saved checkpoint to {ckpt_path}")
            t0 = time.time()

    return raw_model


def _parse_override(text: str) -> tuple[str, Any]:
    """Parse ``--section.key=value``; values are read as Python literals."""
    key, sep, value = text.lstrip("-").partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected section.key=value, got {text!r}")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value  # plain strings such as paths
    return key, parsed


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a language model from scratch.",
        epilog="Override any config field with --section.key=value, e.g. --train.lr=1e-3",
    )
    parser.add_argument("--config", default="configs/tiny.yaml", help="path to a YAML config")
    parser.add_argument("--resume", action="store_true", help="continue from out_dir/ckpt.pt")
    args, extra = parser.parse_known_args(argv)

    overrides = dict(_parse_override(item) for item in extra)
    config = load_config(args.config, overrides)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
