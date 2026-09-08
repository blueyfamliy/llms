"""Loading pre-tokenized data.

``scripts/prepare_data.py`` writes each split as a flat array of token ids in a
``.bin`` file plus a ``meta.json`` describing the dtype. Training memory-maps
those files, so a corpus far larger than RAM costs nothing to open and batches
are sampled by reading random windows straight from the page cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Smallest unsigned integer type that can hold every id in the vocabulary."""
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def write_split(path: str | Path, tokens: np.ndarray, dtype: np.dtype) -> None:
    """Write one split as a flat binary array of token ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(dtype, copy=False).tofile(path)


def write_meta(data_dir: str | Path, meta: dict) -> None:
    """Write the sidecar describing dtype, vocab size, and split lengths."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    (Path(data_dir) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(data_dir: str | Path) -> dict:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found -- run scripts/prepare_data.py to build the dataset first"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


class TokenDataset:
    """Random windows over one memory-mapped split.

    Batches are sampled with replacement from uniformly random offsets rather
    than by walking the file in order. Over a long run this covers the corpus
    just as well, and it keeps the loader stateless and trivially resumable.
    """

    def __init__(self, data_dir: str | Path, split: str, block_size: int) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.block_size = block_size

        meta = read_meta(self.data_dir)
        self.dtype = np.dtype(meta["dtype"])
        self.vocab_size = int(meta["vocab_size"])

        self.path = self.data_dir / f"{split}.bin"
        if not self.path.exists():
            raise FileNotFoundError(f"missing split file: {self.path}")

        # mmap_mode="r" keeps the array lazy; np.memmap is recreated per batch in
        # get_batch to avoid the memory leak from holding one open across a long run.
        self.n_tokens = self.path.stat().st_size // self.dtype.itemsize
        if self.n_tokens <= block_size:
            raise ValueError(
                f"{self.path} has {self.n_tokens} tokens, which is not enough for "
                f"block_size={block_size}; use a larger corpus or a smaller block_size"
            )

    def __len__(self) -> int:
        return self.n_tokens

    def get_batch(
        self, batch_size: int, device: torch.device, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` windows and return ``(inputs, targets)``.

        ``targets`` is ``inputs`` shifted one position left: at every index the
        model predicts the token that follows it.
        """
        data = np.memmap(self.path, dtype=self.dtype, mode="r")
        # -1 because each window needs one extra token for the shifted target.
        offsets = torch.randint(
            len(data) - self.block_size - 1, (batch_size,), generator=generator
        )
        x = torch.stack(
            [torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in offsets]
        )
        y = torch.stack(
            [
                torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64))
                for i in offsets
            ]
        )

        if device.type == "cuda":
            # Pinned memory + non_blocking lets the H2D copy overlap with compute.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
