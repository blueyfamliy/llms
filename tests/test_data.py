"""Tests for config loading and the memory-mapped token dataset."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llms.config import load_config  # noqa: E402
from llms.data import TokenDataset, dtype_for_vocab, write_meta, write_split  # noqa: E402


@pytest.fixture
def dataset_dir(tmp_path):
    tokens = np.arange(1000, dtype=np.int64) % 50
    dtype = dtype_for_vocab(50)
    write_split(tmp_path / "train.bin", tokens, dtype)
    write_split(tmp_path / "val.bin", tokens[:200], dtype)
    write_meta(tmp_path, {"vocab_size": 50, "dtype": dtype.name, "train_tokens": 1000})
    return tmp_path


def test_dtype_scales_with_vocab_size():
    assert dtype_for_vocab(50_000) == np.dtype(np.uint16)
    assert dtype_for_vocab(70_000) == np.dtype(np.uint32)


def test_batch_shapes_and_device(dataset_dir):
    ds = TokenDataset(dataset_dir, "train", block_size=16)
    x, y = ds.get_batch(4, torch.device("cpu"))
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == torch.long


def test_targets_are_inputs_shifted_by_one(dataset_dir):
    """y[t] must be the token that follows x[t] -- the whole training signal."""
    ds = TokenDataset(dataset_dir, "train", block_size=16)
    x, y = ds.get_batch(8, torch.device("cpu"))
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_batches_stay_inside_the_vocabulary(dataset_dir):
    ds = TokenDataset(dataset_dir, "train", block_size=16)
    x, y = ds.get_batch(16, torch.device("cpu"))
    assert x.max().item() < 50 and y.max().item() < 50


def test_missing_meta_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        TokenDataset(tmp_path, "train", block_size=8)


def test_block_size_larger_than_corpus_is_rejected(dataset_dir):
    with pytest.raises(ValueError, match="not enough"):
        TokenDataset(dataset_dir, "val", block_size=500)


def test_config_yaml_roundtrip_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  n_layer: 4\ntrain:\n  lr: 0.001\n")
    cfg = load_config(path, {"train.max_steps": 50, "model.n_embd": 128})
    assert cfg.model.n_layer == 4
    assert cfg.model.n_embd == 128
    assert cfg.train.lr == 0.001
    assert cfg.train.max_steps == 50


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  n_layerz: 4\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)
