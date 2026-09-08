"""Byte-level BPE tokenizer, implemented from scratch.

The algorithm is the one used by GPT-2/GPT-4: text is first split by a regex
into chunks that never merge across (so " dog" and "dog." stay separate), each
chunk is encoded to UTF-8 bytes, and then the most frequent adjacent byte pair
is merged repeatedly until the vocabulary reaches the requested size.

Working on bytes rather than Unicode characters means every possible input is
representable with a 256-symbol base alphabet -- there is no "unknown token".
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import regex as re

# GPT-4's pre-tokenization pattern. It keeps a leading space attached to a word,
# splits contractions off, caps runs of digits at 3, and never lets a newline run
# merge into a word.
SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

END_OF_TEXT = "<|endoftext|>"

Pair = tuple[int, int]


def _count_pairs(ids: Sequence[int], counts: Counter, weight: int = 1) -> None:
    """Add every adjacent pair in ``ids`` to ``counts``, weighted by ``weight``."""
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight


def _merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """A trainable byte-level BPE tokenizer.

    Attributes:
        merges: ordered ``(pair) -> new_id`` map; insertion order is merge priority.
        vocab: ``id -> bytes`` for every token, built from ``merges``.
        special_tokens: ``str -> id`` for tokens that bypass BPE entirely.
    """

    def __init__(self, pattern: str = SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self.merges: dict[Pair, int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self._inverse_special: dict[int, str] = {}
        self._special_re: re.Pattern | None = None

    # ------------------------------------------------------------------ size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eot_id(self) -> int | None:
        """Id of ``<|endoftext|>`` if it was registered, else ``None``."""
        return self.special_tokens.get(END_OF_TEXT)

    # --------------------------------------------------------------- training

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: Sequence[str] = (END_OF_TEXT,),
        verbose: bool = False,
    ) -> BPETokenizer:
        """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``.

        ``vocab_size`` counts the 256 byte tokens and the learned merges; special
        tokens are added on top of it.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        n_merges = vocab_size - 256

        # Count identical chunks once instead of walking the whole corpus every
        # merge -- for natural text this is a ~100x speedup over the naive loop.
        chunk_counts: Counter = Counter(self._compiled.findall(text))
        words: dict[tuple[int, ...], int] = {}
        for chunk, count in chunk_counts.items():
            ids = tuple(chunk.encode("utf-8"))
            words[ids] = words.get(ids, 0) + count

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(n_merges):
            pair_counts: Counter = Counter()
            for ids, count in words.items():
                if len(ids) > 1:
                    _count_pairs(ids, pair_counts, count)
            if not pair_counts:
                if verbose:
                    print(f"no pairs left after {i} merges; stopping early")
                break

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            new_id = 256 + i
            words = {
                (tuple(_merge(ids, best_pair, new_id)) if len(ids) > 1 else ids): count
                for ids, count in words.items()
            }
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i + 1) % 100 == 0:
                token = self.vocab[new_id]
                print(
                    f"merge {i + 1}/{n_merges}: {best_pair} -> {new_id} "
                    f"({token!r}, {pair_counts[best_pair]} occurrences)"
                )

        self.register_special_tokens(special_tokens)
        return self

    def register_special_tokens(self, tokens: Iterable[str] | Mapping[str, int]) -> None:
        """Assign ids to special tokens, placing them after all BPE tokens.

        Accepts a mapping to restore exact ids from a saved tokenizer, or any
        iterable of strings to number them sequentially.
        """
        if isinstance(tokens, Mapping):
            self.special_tokens = {str(k): int(v) for k, v in tokens.items()}
        else:
            self.special_tokens = {t: len(self.vocab) + i for i, t in enumerate(tokens)}
        clashes = sorted(set(self.special_tokens.values()) & set(self.vocab))
        if clashes:
            raise ValueError(f"special token ids collide with BPE token ids: {clashes}")
        self._inverse_special = {v: k for k, v in self.special_tokens.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )

    # --------------------------------------------------------------- encoding

    def _encode_chunk(self, data: bytes) -> list[int]:
        """BPE-encode one pre-tokenized chunk of bytes."""
        ids = list(data)
        while len(ids) >= 2:
            # Apply the earliest-learned merge present, which reproduces the
            # exact sequence of merges seen during training.
            pair = min(
                (p for p in zip(ids, ids[1:]) if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if pair is None:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as ordinary text."""
        ids: list[int] = []
        for chunk in self._compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text to token ids.

        With ``allowed_special=True`` (the default) occurrences of registered
        special-token strings become their single reserved id. Set it to False
        when encoding untrusted text, so a document containing the literal
        string ``<|endoftext|>`` cannot inject a document boundary.
        """
        if not allowed_special or not self._special_re:
            return self.encode_ordinary(text)

        ids: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    # --------------------------------------------------------------- decoding

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Decode token ids back to text.

        A token boundary can fall in the middle of a multi-byte UTF-8 character
        (common when sampling), so bytes are concatenated first and decoded once;
        ``errors="replace"`` keeps partial output printable.
        """
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self._inverse_special:
                parts.append(self._inverse_special[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise ValueError(f"token id out of range: {i}")
        return b"".join(parts).decode("utf-8", errors=errors)

    # ---------------------------------------------------------------- storage

    def save(self, path: str | Path) -> None:
        """Write the tokenizer to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pattern": self.pattern,
            # JSON keys must be strings; store merges as a flat ordered list.
            "merges": [[int(a), int(b), int(idx)] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Read a tokenizer written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(pattern=payload.get("pattern", SPLIT_PATTERN))
        tok.merges = {}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, idx in payload["merges"]:
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        tok.register_special_tokens(payload.get("special_tokens", {}))
        return tok
