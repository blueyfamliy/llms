#!/usr/bin/env python3
"""Turn a raw text corpus into the token shards the training loop reads.

Two ways to use it:

    # built-in demo corpus (~1.1 MB, downloaded on first run)
    python scripts/prepare_data.py --dataset tinyshakespeare --vocab-size 4096

    # your own text
    python scripts/prepare_data.py --input corpus.txt --out-dir data/mine --vocab-size 8192

Each document boundary is marked with ``<|endoftext|>`` so the model learns
where texts start and stop. Output is ``train.bin``, ``val.bin``,
``tokenizer.json``, and ``meta.json`` in the output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llms.data import dtype_for_vocab, write_meta, write_split  # noqa: E402
from llms.tokenizer import BPETokenizer  # noqa: E402

DATASETS = {
    "tinyshakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ),
}


def download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` unless it is already there."""
    import urllib.request

    if dest.exists():
        print(f"using cached {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
    return dest


def read_documents(paths: list[Path], split_on_blank: bool) -> list[str]:
    """Read the input files as a list of documents."""
    docs: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if split_on_blank:
            docs.extend(d.strip() for d in text.split("\n\n\n") if d.strip())
        else:
            docs.append(text)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=sorted(DATASETS), help="a built-in demo corpus")
    source.add_argument("--input", nargs="+", type=Path, help="one or more .txt files")
    parser.add_argument("--out-dir", type=Path, default=None, help="where to write the shards")
    parser.add_argument("--vocab-size", type=int, default=4096, help="BPE vocabulary size")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="reuse an existing tokenizer.json instead of training a new one",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.01, help="share of tokens held out for validation"
    )
    parser.add_argument(
        "--split-on-blank-lines",
        action="store_true",
        help="treat runs of blank lines as document boundaries",
    )
    parser.add_argument(
        "--tokenizer-sample-mb",
        type=float,
        default=20.0,
        help="cap the text used to train the tokenizer; BPE training is the slow part",
    )
    args = parser.parse_args()

    if args.dataset:
        out_dir = args.out_dir or Path("data") / args.dataset
        raw = download(DATASETS[args.dataset], out_dir / "input.txt")
        paths = [raw]
    else:
        if args.out_dir is None:
            parser.error("--out-dir is required when using --input")
        out_dir = args.out_dir
        paths = args.input
        missing = [p for p in paths if not p.exists()]
        if missing:
            parser.error(f"input file(s) not found: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    docs = read_documents(paths, args.split_on_blank_lines)
    total_chars = sum(len(d) for d in docs)
    print(f"read {len(docs)} document(s), {total_chars:,} characters")

    tokenizer_path = out_dir / "tokenizer.json"
    if args.tokenizer:
        tokenizer = BPETokenizer.load(args.tokenizer)
        print(f"loaded tokenizer from {args.tokenizer} (vocab_size={tokenizer.vocab_size})")
    else:
        budget = int(args.tokenizer_sample_mb * 1_000_000)
        sample = "\n\n".join(docs)[:budget]
        print(f"training BPE on {len(sample):,} characters -> vocab_size={args.vocab_size}")
        tokenizer = BPETokenizer().train(sample, args.vocab_size, verbose=True)
        tokenizer.save(tokenizer_path)
        print(f"wrote {tokenizer_path}")

    eot = tokenizer.eot_id
    ids: list[int] = []
    for i, doc in enumerate(docs):
        if eot is not None:
            ids.append(eot)  # prefix each document with the boundary marker
        ids.extend(tokenizer.encode(doc, allowed_special=False))
        if (i + 1) % 1000 == 0:
            print(f"tokenized {i + 1}/{len(docs)} documents")

    tokens = np.array(ids, dtype=np.int64)
    if len(tokens) < 2:
        raise SystemExit("corpus produced fewer than 2 tokens; is the input empty?")

    # A contiguous tail is held out rather than a random subset: random windows
    # would overlap the training set and make validation loss look better than it is.
    n_val = max(1, int(len(tokens) * args.val_fraction))
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]

    dtype = dtype_for_vocab(tokenizer.vocab_size)
    write_split(out_dir / "train.bin", train_tokens, dtype)
    write_split(out_dir / "val.bin", val_tokens, dtype)
    write_meta(
        out_dir,
        {
            "vocab_size": tokenizer.vocab_size,
            "dtype": dtype.name,
            "train_tokens": int(len(train_tokens)),
            "val_tokens": int(len(val_tokens)),
            "tokenizer": tokenizer_path.name,
        },
    )

    ratio = total_chars / len(tokens) if len(tokens) else 0.0
    print(
        f"\ntrain {len(train_tokens):,} tokens | val {len(val_tokens):,} tokens | "
        f"vocab {tokenizer.vocab_size} | {ratio:.2f} chars/token"
    )
    print(f"wrote shards to {out_dir}/")


if __name__ == "__main__":
    main()
