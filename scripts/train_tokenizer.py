#!/usr/bin/env python3
"""Train a standalone BPE tokenizer on a text corpus.

    python scripts/train_tokenizer.py --input corpus.txt --vocab-size 8192 \
        --out data/mine/tokenizer.json

Useful when you want one tokenizer shared across several datasets; otherwise
``scripts/prepare_data.py`` trains one for you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="text file(s)")
    parser.add_argument("--out", type=Path, required=True, help="where to write tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--sample-mb", type=float, default=20.0, help="cap the text used for training"
    )
    args = parser.parse_args()

    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in args.input)
    budget = int(args.sample_mb * 1_000_000)
    if len(text) > budget:
        print(f"truncating {len(text):,} -> {budget:,} characters for tokenizer training")
        text = text[:budget]

    tokenizer = BPETokenizer().train(text, args.vocab_size, verbose=True)
    tokenizer.save(args.out)

    n_tokens = len(tokenizer.encode(text[:200_000], allowed_special=False))
    print(f"\nwrote {args.out} (vocab_size={tokenizer.vocab_size})")
    print(f"compression: {min(len(text), 200_000) / max(1, n_tokens):.2f} chars/token")


if __name__ == "__main__":
    main()
