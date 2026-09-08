"""Train a decoder-only language model from scratch."""

from .config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from .data import TokenDataset
from .model import GPT
from .tokenizer import BPETokenizer

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
    "TokenDataset",
    "GPT",
    "BPETokenizer",
]
