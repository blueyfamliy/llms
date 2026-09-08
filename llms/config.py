"""Configuration dataclasses for model, data, and training.

Configs are plain dataclasses so they can be constructed in code, and they can
also be loaded from YAML (see ``load_config``) for reproducible runs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Architecture of the decoder-only transformer."""

    vocab_size: int = 8192
    block_size: int = 512  # maximum context length in tokens
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int | None = None  # None -> multi-head attention (n_kv_head == n_head)
    n_embd: int = 512
    ffn_hidden: int | None = None  # None -> derived as ~8/3 * n_embd, rounded to a multiple of 64
    dropout: float = 0.0
    bias: bool = False  # biases in linear layers; False is standard for modern LLMs
    rope_theta: float = 10000.0
    tie_embeddings: bool = True  # share weights between token embedding and output head

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        if self.ffn_hidden is None:
            # SwiGLU uses three matrices instead of two, so the hidden size is scaled by 2/3
            # to keep the parameter count comparable to a 4x GELU MLP.
            hidden = int(8 * self.n_embd / 3)
            self.ffn_hidden = 64 * ((hidden + 63) // 64)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class DataConfig:
    """Where the pre-tokenized ``.bin`` shards live."""

    data_dir: str = "data/tinyshakespeare"
    tokenizer_path: str = "data/tinyshakespeare/tokenizer.json"


@dataclass
class TrainConfig:
    """Optimization, schedule, and runtime knobs."""

    out_dir: str = "out"
    # batching
    batch_size: int = 32  # micro-batch per device
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps * world_size
    # schedule
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # evaluation & checkpointing
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    always_save_checkpoint: bool = False
    # runtime
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    dtype: str = "auto"  # "auto" | "bfloat16" | "float16" | "float32"
    compile: bool = False  # torch.compile; big speedup on CUDA, flaky elsewhere
    seed: int = 1337
    num_workers: int = 0


@dataclass
class Config:
    """Top-level config combining the three sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _from_dict(cls: type[T], values: dict[str, Any]) -> T:
    """Build a dataclass from a dict, rejecting unknown keys so typos fail loudly."""
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)  # type: ignore[call-arg]


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config with ``model`` / ``data`` / ``train`` sections.

    ``overrides`` is a flat mapping of ``"section.key" -> value`` applied after
    the file is read, which is what the CLI uses for ``--train.lr=1e-3``.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sections = {"model": {}, "data": {}, "train": {}}
    unknown_sections = set(raw) - set(sections)
    if unknown_sections:
        raise ValueError(f"unknown config sections: {sorted(unknown_sections)}")
    for name in sections:
        sections[name] = dict(raw.get(name) or {})

    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key or section not in sections:
            raise ValueError(f"override must be 'section.key=value', got {dotted!r}")
        sections[section][key] = value

    return Config(
        model=_from_dict(ModelConfig, sections["model"]),
        data=_from_dict(DataConfig, sections["data"]),
        train=_from_dict(TrainConfig, sections["train"]),
    )


def config_to_dict(config: Config) -> dict[str, Any]:
    """Serialize a config back to nested plain dicts (for checkpoints and logs)."""
    return dataclasses.asdict(config)
