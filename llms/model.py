"""A decoder-only transformer, in the shape modern LLMs actually use.

Differences from the original GPT-2 architecture, each of which is now standard:

* **RMSNorm** instead of LayerNorm -- no mean subtraction, no bias, slightly cheaper.
* **Rotary position embeddings (RoPE)** instead of learned positional embeddings --
  positions enter through the attention dot product, so the model extrapolates
  better and there is no position table to size.
* **SwiGLU** feed-forward instead of GELU MLP -- a gated activation that performs
  better at equal parameter count.
* **Grouped-query attention (GQA)** -- query heads share key/value heads, which
  shrinks the KV cache during generation.
* **Pre-norm** residual blocks, which is what makes deep stacks trainable.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 even under autocast: the mean of squares
        # underflows in float16 for the activation magnitudes seen late in training.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    head_dim: int, seq_len: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin tables for rotary embeddings.

    Returns two ``(seq_len, head_dim)`` tensors; each frequency is duplicated
    across the two halves so it lines up with :func:`rotate_half`.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim / 2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the halves of the last dimension: ``[a, b] -> [-b, a]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to ``(B, n_head, T, head_dim)`` queries or keys."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(B, n_kv_head, T, D)`` to ``(B, n_kv_head * n_rep, T, D)`` for GQA."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional grouped queries."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.head_dim
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_cache = (k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        kv_len = k.shape[2]
        if kv_len == t:
            # No cache: the usual lower-triangular mask, fused by the kernel.
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            # With a cache, query i sits at absolute position past_len + i and may
            # attend to every key up to it. is_causal would mask against the wrong
            # origin here, so build the mask explicitly.
            past_len = kv_len - t
            causal = torch.ones(t, kv_len, dtype=torch.bool, device=q.device).tril(diagonal=past_len)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal, dropout_p=self.dropout if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y)), new_cache


class SwiGLU(nn.Module):
    """Gated feed-forward network: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ffn_hidden or 4 * config.n_embd
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class GPT(nn.Module):
    """The full language model: embeddings, a stack of blocks, and an output head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Input and output embeddings share one matrix, which saves
            # vocab_size * n_embd parameters and usually improves small models.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down the residual output projections so the variance of the
        # residual stream stays ~constant as depth grows (GPT-2 §2.3).
        scale = 0.02 / math.sqrt(2 * config.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=scale)

        cos, sin = build_rope_cache(
            config.head_dim, config.block_size, config.rope_theta, torch.device("cpu"), torch.float32
        )
        # Buffers so they move with .to(device) but are not saved in checkpoints
        # (they are cheap to rebuild and depend only on the config).
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count parameters; by default excludes the token embedding table."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, KVCache]:
        """Run the model.

        Args:
            idx: ``(B, T)`` token ids.
            targets: ``(B, T)`` next-token ids; when given, the loss is returned.
            caches: per-layer KV cache from a previous call, for incremental decoding.

        Returns:
            ``(logits, loss, caches)``. During generation only the last position's
            logits are computed, so ``logits`` is ``(B, 1, vocab_size)``.
        """
        b, t = idx.shape
        past_len = caches[0][0].shape[2] if caches else 0
        total = past_len + t
        if total > self.config.block_size:
            raise ValueError(
                f"sequence length {total} exceeds block_size {self.config.block_size}"
            )

        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[past_len:total].to(x.dtype)
        sin = self.rope_sin[past_len:total].to(x.dtype)

        new_caches: KVCache = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, caches[i] if caches else None)
            new_caches.append(cache)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        else:
            # Only the final position matters for sampling; skip the rest of the
            # (vocab_size-wide, and therefore expensive) head.
            logits = self.lm_head(x[:, -1:, :])
            loss = None
        return logits, loss, new_caches

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.Optimizer:
        """AdamW with decay applied only to matrices, not to norms and biases."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # The fused kernel is a meaningful speedup and is CUDA-only.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = device_type == "cuda" and fused_available
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample a continuation of ``idx`` ``(B, T)``.

        ``temperature=0`` is greedy decoding. ``top_k`` and ``top_p`` can be
        combined; both are applied before the softmax renormalization.
        """
        self.eval()
        block_size = self.config.block_size
        caches: KVCache | None = None
        # `output` accumulates everything, prompt included, and is what we return.
        # `cursor` is only what still has to be pushed through the model: the whole
        # prompt on the first step, then one token at a time.
        output = idx
        cursor = idx[:, -block_size:]

        for _ in range(max_new_tokens):
            # Once the context is full, slide the window: re-encode the most recent
            # block_size - 1 tokens from scratch and start a fresh cache. `output`
            # is untouched, so nothing generated so far is lost.
            if caches is not None and caches[0][0].shape[2] + cursor.shape[1] > block_size:
                cursor = output[:, -(block_size - 1):]
                caches = None

            logits, _, caches = self(cursor, caches=caches)
            logits = logits[:, -1, :].float()

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    # Keep the first token that crosses top_p, drop everything after.
                    remove_sorted = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_idx, remove_sorted)
                    logits = logits.masked_fill(remove, float("-inf"))
                probs = logits.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            output = torch.cat((output, next_token), dim=1)
            cursor = next_token  # subsequent steps feed only the new token

            if eos_id is not None and (next_token == eos_id).all():
                break

        return output
