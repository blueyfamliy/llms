"""Generate text from a trained checkpoint.

    python -m llms.sample --prompt "To be, or not to be" --max-new-tokens 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .model import GPT
from .tokenizer import BPETokenizer
from .train import resolve_device


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[GPT, Config]:
    """Rebuild the model from a checkpoint written by ``llms.train``."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw = ckpt["config"]
    config = Config(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
    )
    model = GPT(config.model)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, config


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
    num_samples: int = 1,
    device: torch.device | None = None,
    seed: int | None = None,
) -> list[str]:
    """Sample ``num_samples`` continuations of ``prompt``."""
    device = device or next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    # allowed_special=False so a prompt containing "<|endoftext|>" is treated as
    # literal text rather than a document boundary.
    ids = tokenizer.encode(prompt, allowed_special=False)
    primed = False
    if not ids:
        # An empty prompt still needs a starting token; use the document boundary
        # marker, which is what the model saw at the start of every document.
        ids = [tokenizer.eot_id] if tokenizer.eot_id is not None else [0]
        primed = True  # drop it again before returning, it is not generated text

    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    context = context.expand(num_samples, -1).contiguous()

    out = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eot_id,
    )
    if primed:
        out = out[:, 1:]
    return [tokenizer.decode(row.tolist()) for row in out]


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained model.")
    parser.add_argument("--ckpt", default="out/ckpt.pt", help="path to the checkpoint")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="path to tokenizer.json (default: the one recorded in the checkpoint's config)",
    )
    parser.add_argument("--prompt", default="", help="text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    parser.add_argument("--top-k", type=int, default=50, help="0 to disable")
    parser.add_argument("--top-p", type=float, default=None, help="nucleus sampling threshold")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, config = load_model(args.ckpt, device)
    tokenizer_path = args.tokenizer or config.data.tokenizer_path
    tokenizer = BPETokenizer.load(tokenizer_path)

    samples = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
        top_p=args.top_p,
        num_samples=args.num_samples,
        device=device,
        seed=args.seed,
    )
    for i, text in enumerate(samples):
        if len(samples) > 1:
            print(f"\n===== sample {i + 1} =====")
        print(text)


if __name__ == "__main__":
    main()
