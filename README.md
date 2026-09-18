# llms

Train a language model from scratch, in readable PyTorch. Tokenizer, architecture,
training loop, and sampling are all here — no `transformers`, no trainer abstraction,
nothing that hides the part you came to read.

The whole model is ~360 lines in [llms/model.py](llms/model.py), comments included, and
the BPE tokenizer is ~250 in [llms/tokenizer.py](llms/tokenizer.py).

## Quickstart

```bash
pip install -r requirements.txt

# 1. build a tokenizer and token shards from a corpus (~1.1 MB, downloaded once)
python scripts/prepare_data.py --dataset tinyshakespeare --vocab-size 4096

# 2. train (a few minutes on a laptop; much faster on a GPU)
python -m llms.train --config configs/tiny.yaml

# 3. sample from what you trained
python -m llms.sample --ckpt out/tiny/ckpt.pt --prompt "To be, or not to be"
```

To use your own text instead:

```bash
python scripts/prepare_data.py --input corpus.txt --out-dir data/mine --vocab-size 8192
python -m llms.train --config configs/small.yaml --data.data_dir=data/mine
```

Any config field can be overridden from the command line with a dotted key:

```bash
python -m llms.train --config configs/tiny.yaml --train.lr=3e-4 --model.n_layer=12
```

## What the model is

A decoder-only transformer in the shape current LLMs actually use, rather than the
2019 GPT-2 shape:

| Component | Choice | Why |
| --- | --- | --- |
| Normalization | **RMSNorm**, pre-norm | No mean subtraction or bias; pre-norm is what makes deep stacks trainable |
| Positions | **RoPE** | Position enters through the attention dot product — no position table to size, better length behavior |
| Feed-forward | **SwiGLU** | Gated activation; better loss than a GELU MLP at equal parameters |
| Attention | **Grouped-query (optional)** | Query heads share KV heads, shrinking the KV cache during generation |
| Output head | **Tied to the embedding** (optional) | Saves `vocab_size × n_embd` parameters; usually helps small models |
| Attention kernel | `F.scaled_dot_product_attention` | Uses FlashAttention where the hardware supports it |

Attention, RoPE, and SwiGLU are each written out rather than imported, and each has a
test pinning the property that matters (causality, norm preservation, cache equivalence).

## The tokenizer

Byte-level BPE, trained from scratch. Text is split by a GPT-4-style regex, encoded to
UTF-8 bytes, and the most frequent adjacent pair is merged repeatedly until the vocabulary
is full. Because the base alphabet is the 256 bytes, **every input is representable** —
there is no unknown token, and `decode(encode(x)) == x` for any string, including emoji
and scripts the tokenizer never saw in training.

```python
from llms.tokenizer import BPETokenizer

tok = BPETokenizer().train(open("corpus.txt").read(), vocab_size=8192)
tok.save("tokenizer.json")

tok.encode("hello world")           # -> [1247, 2891]
tok.decode([1247, 2891])            # -> "hello world"
tok.encode(untrusted, allowed_special=False)   # can't inject <|endoftext|>
```

## Layout

```
llms/
  config.py      dataclass configs, YAML loading, dotted CLI overrides
  tokenizer.py   byte-level BPE: train, encode, decode, save/load
  model.py       RMSNorm, RoPE, GQA attention, SwiGLU, GPT, generate()
  data.py        memory-mapped token shards and batch sampling
  train.py       training loop: AMP, grad accumulation, cosine LR, checkpoints
  sample.py      generation CLI with temperature / top-k / top-p
scripts/
  prepare_data.py   corpus -> tokenizer.json + train.bin / val.bin / meta.json
  train_tokenizer.py  train a tokenizer on its own
configs/
  tiny.yaml      ~3M params, trains on a laptop — use this to verify the pipeline
  small.yaml     ~35M params, single GPU
```

## Training details

- **Optimizer**: AdamW, `β = (0.9, 0.95)`, weight decay on matrices only — never on
  RMSNorm gains or biases, where decay just pulls a scale toward zero for no reason.
- **Schedule**: linear warmup, then cosine decay to `min_lr`.
- **Precision**: bfloat16 autocast on capable CUDA devices, float32 elsewhere. bfloat16
  keeps float32's exponent range, so no loss scaling is needed; float16 gets a `GradScaler`.
- **Gradient accumulation**: the effective batch is
  `batch_size × grad_accum_steps × block_size` tokens. The loss is divided by the
  accumulation count so it behaves like one large batch.
- **Checkpointing**: written on validation improvement, with model, optimizer, config
  and step. `--resume` picks up exactly where it stopped.
- **Data**: splits are memory-mapped, so a corpus larger than RAM costs nothing to open.
  Validation is a contiguous tail, not a random subset — random windows would overlap
  training data and flatter the validation loss.

### Sizing a run

Compute scales as roughly `6 × parameters × tokens`. The Chinchilla result puts the
compute-optimal ratio near **20 tokens per parameter**, so a 35M-parameter model wants
on the order of 700M tokens. Training well past that still improves the model — it is
just no longer the best use of the next FLOP. Below roughly 5 tokens per parameter,
expect the model to memorize instead of generalize, and `dropout` to start earning its keep.

## Generation

```python
from llms.sample import generate_text, load_model
from llms.tokenizer import BPETokenizer
import torch

model, config = load_model("out/tiny/ckpt.pt", torch.device("cpu"))
tok = BPETokenizer.load(config.data.tokenizer_path)

print(generate_text(model, tok, "the sea was", max_new_tokens=200, top_k=50)[0])
```

Decoding uses a KV cache, so each new token costs one forward pass over one position
rather than over the whole context. Past `block_size`, the window slides: the most recent
`block_size - 1` tokens are re-encoded and the cache is rebuilt, while everything
generated so far is kept.

`temperature=0` is greedy; `top_k` and `top_p` can be combined.

## LLMS Studio (desktop app)

A UI over the same pipeline: pick a config, train, sample, watch the log stream live.

```bash
python llms_studio.py
```

It starts a local server on `127.0.0.1` and opens your browser. Add `--no-browser`
to just get the URL.

<img src="assets/icon.png" width="96" align="right" alt="LLMS Studio icon">

**Why a browser and not a native window.** Apple's system Tcl/Tk is version 8.5,
deprecated for years, and on macOS 26+ it is effectively dead: a bare
`tkinter.Tk()` constructs, but `update()` never returns, so the window is created
and never mapped. Any Tk UI built against the system Python fails there — the app
launches, AppKit logs `No windows open yet`, and terminates it about twenty seconds
later. Serving the UI over HTTP has no such dependency and looks the same on every
platform.

To build a double-clickable app with the icon:

```bash
python build_releases.py
```

That produces `dist/LLMS-Studio.app` (macOS) from `LLMS-Studio.spec`. The icon is
generated by `scripts/make_icon.py` — geometry drawn per size, so the 16px version
stays legible rather than being downsampled to mush.

### Workspace

Run from source, the workspace is the repo. Run as a bundled app, it is
`~/LLMS-Studio`, seeded on first launch with the bundled configs — a bundle is
read-only and is launched with the working directory set to `/`, so datasets and
checkpoints need somewhere writable to live. Startup problems are logged to
`~/LLMS-Studio/llms-studio.log`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

74 tests covering the properties that break silently if you get them wrong: that
attention is causal, that the KV cache reproduces a full forward pass exactly, that RoPE
preserves vector norms, that targets are inputs shifted by one, that BPE round-trips
losslessly on text it never saw, and that the model can overfit a single batch.
The GUI tests cover resource resolution, subprocess argv construction, and the
HTTP endpoints — Tk is never constructed, since that needs a window server.

## License

MIT — see [LICENSE](LICENSE).
