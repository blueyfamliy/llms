"""Tests for the transformer: shapes, causality, KV cache, and optimization."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llms.config import ModelConfig  # noqa: E402
from llms.model import GPT, build_rope_cache  # noqa: E402


def tiny_config(**kwargs) -> ModelConfig:
    defaults = dict(
        vocab_size=64, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_config()).eval()


# ------------------------------------------------------------------- config


def test_ffn_hidden_is_derived_and_multiple_of_64():
    cfg = tiny_config(n_embd=512)
    assert cfg.ffn_hidden % 64 == 0
    assert cfg.ffn_hidden == pytest.approx(8 * 512 / 3, abs=64)


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=32, n_head=5)


def test_config_rejects_bad_gqa_grouping():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=32, n_head=4, n_kv_head=3)


# -------------------------------------------------------------------- shapes


def test_forward_with_targets_returns_full_logits(model: GPT):
    x = torch.randint(0, 64, (2, 8))
    logits, loss, _ = model(x, targets=x)
    assert logits.shape == (2, 8, 64)
    assert loss.ndim == 0 and loss.item() > 0


def test_forward_without_targets_returns_last_position_only(model: GPT):
    logits, loss, _ = model(torch.randint(0, 64, (2, 8)))
    assert logits.shape == (2, 1, 64)
    assert loss is None


def test_initial_loss_is_near_uniform_entropy(model: GPT):
    """An untrained model should sit close to ln(vocab_size) = ln(64) ~= 4.16.

    Targets must be independent of the inputs here. With tied embeddings the
    residual stream carries the input embedding to the output head, so at
    initialization the model favors copying its own input -- scoring targets==x
    would measure that artifact rather than the initialization.
    """
    torch.manual_seed(0)
    x = torch.randint(0, 64, (8, 16))
    y = torch.randint(0, 64, (8, 16))
    _, loss, _ = model(x, targets=y)
    assert abs(loss.item() - torch.log(torch.tensor(64.0)).item()) < 0.3


def test_sequence_longer_than_block_size_is_rejected(model: GPT):
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 64, (1, 17)))


# ------------------------------------------------------------------ causality


def test_future_tokens_do_not_affect_earlier_positions(model: GPT):
    """The core invariant of a causal LM: position i sees only positions <= i."""
    torch.manual_seed(1)
    a = torch.randint(0, 64, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 7) % 64  # change everything from position 8 onward

    logits_a, _, _ = model(a, targets=a)
    logits_b, _, _ = model(b, targets=b)

    torch.testing.assert_close(logits_a[:, :8], logits_b[:, :8], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(logits_a[:, 8:], logits_b[:, 8:])


# ------------------------------------------------------------------- kv cache


def test_kv_cache_matches_full_forward(model: GPT):
    """Incremental decoding must reproduce the logits of a single full pass."""
    x = torch.randint(0, 64, (2, 10))
    full_logits, _, _ = model(x)

    caches = None
    step_logits = None
    for t in range(x.shape[1]):
        step_logits, _, caches = model(x[:, t : t + 1], caches=caches)

    assert caches is not None and len(caches) == 2
    assert caches[0][0].shape[2] == 10  # keys for all 10 positions
    torch.testing.assert_close(full_logits, step_logits, rtol=1e-4, atol=1e-5)


def test_gqa_shrinks_the_kv_cache():
    torch.manual_seed(0)
    model = GPT(tiny_config(n_head=4, n_kv_head=1)).eval()
    _, _, caches = model(torch.randint(0, 64, (1, 6)))
    keys = caches[0][0]
    assert keys.shape[1] == 1, "one KV head shared by all four query heads"


# ---------------------------------------------------------------------- rope


def test_rope_preserves_vector_norm():
    """RoPE is a rotation, so it must not change magnitudes."""
    cos, sin = build_rope_cache(8, 16, 10000.0, torch.device("cpu"), torch.float32)
    from llms.model import apply_rope

    x = torch.randn(1, 2, 16, 8)
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_rope_cache_shape():
    cos, sin = build_rope_cache(16, 32, 10000.0, torch.device("cpu"), torch.float32)
    assert cos.shape == sin.shape == (32, 16)


# ------------------------------------------------------------------ parameters


def test_embeddings_are_tied_by_default(model: GPT):
    assert model.lm_head.weight is model.tok_emb.weight


def test_untied_embeddings_are_separate_tensors():
    model = GPT(tiny_config(tie_embeddings=False))
    assert model.lm_head.weight is not model.tok_emb.weight


def test_optimizer_excludes_norms_from_weight_decay(model: GPT):
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # Every RMSNorm gain is 1-D and must land in the no-decay group.
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert len(no_decay["params"]) == 2 * 2 + 1  # two norms per block, plus the final one


# ------------------------------------------------------------------ generation


def test_generate_appends_exactly_max_new_tokens(model: GPT):
    out = model.generate(torch.randint(0, 64, (2, 4)), max_new_tokens=5, top_k=10)
    assert out.shape == (2, 9)


def test_generate_slides_the_window_past_block_size(model: GPT):
    """Generating past block_size keeps working, and keeps every token generated.

    block_size is 16 here, so a 14-token prompt plus 20 new tokens forces the
    window to slide twice. The returned sequence must still be prompt + 20.
    """
    prompt = torch.randint(0, 64, (1, 14))
    out = model.generate(prompt, max_new_tokens=20, top_k=5)
    assert out.shape == (1, 34)
    torch.testing.assert_close(out[:, :14], prompt)  # prompt survives the slide
    assert out.max().item() < 64


def test_generate_accepts_a_prompt_longer_than_block_size(model: GPT):
    """An over-long prompt is truncated to the last block_size tokens, not rejected."""
    prompt = torch.randint(0, 64, (1, 25))  # block_size is 16
    out = model.generate(prompt, max_new_tokens=4, top_k=5)
    assert out.shape == (1, 29)
    torch.testing.assert_close(out[:, :25], prompt)


def test_greedy_decoding_is_deterministic(model: GPT):
    prompt = torch.randint(0, 64, (1, 4))
    a = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    torch.testing.assert_close(a, b)


def test_top_k_restricts_the_sampled_set():
    """With top_k=1, sampling collapses to greedy."""
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    prompt = torch.randint(0, 64, (1, 4))
    greedy = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    top1 = model.generate(prompt, max_new_tokens=6, temperature=1.0, top_k=1)
    torch.testing.assert_close(greedy, top1)


# ---------------------------------------------------------------- optimization


def test_model_can_overfit_a_single_batch():
    """A few dozen steps on one batch should drive the loss well below chance."""
    torch.manual_seed(0)
    model = GPT(tiny_config())
    optimizer = model.configure_optimizers(0.0, 3e-3, (0.9, 0.95), "cpu")
    x = torch.randint(0, 64, (4, 12))
    y = torch.randint(0, 64, (4, 12))

    _, first_loss, _ = model(x, targets=y)
    for _ in range(120):
        _, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss.item() * 0.25


def test_all_parameters_receive_gradients():
    """A parameter with no gradient is dead weight -- usually a wiring bug."""
    model = GPT(tiny_config())
    x = torch.randint(0, 64, (2, 8))
    _, loss, _ = model(x, targets=x)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradients: {missing}"
