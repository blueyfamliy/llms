"""Tests for the byte-level BPE tokenizer."""

from __future__ import annotations

import pytest

from llms.tokenizer import END_OF_TEXT, BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick and the dog is lazy. "
    "quick quick quick brown brown fox fox fox\n\n"
) * 20


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer().train(CORPUS, vocab_size=400)


def test_untrained_tokenizer_is_byte_level():
    tok = BPETokenizer()
    assert tok.encode("hi") == [104, 105]
    assert tok.decode([104, 105]) == "hi"


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "unseen vocabulary: xyzzy plugh",
        "",
        "   leading and trailing   ",
        "punctuation!? (yes) -- and 12345",
        "emoji 🦊 and accents café naïve",
        "עברית ורוסית: привет",
        "line\nbreaks\n\nand\ttabs",
    ],
)
def test_encode_decode_roundtrip(tokenizer: BPETokenizer, text: str):
    """Byte-level BPE is lossless: decode(encode(x)) == x for any input."""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_merges_actually_compress(tokenizer: BPETokenizer):
    text = "the quick brown fox jumps over the lazy dog"
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(text))
    assert n_tokens < n_bytes, "training should produce merges that shorten the sequence"


def test_vocab_size_matches_request():
    """With enough varied text, every requested merge is learned."""
    varied = "".join(chr(0x61 + i % 26) + chr(0x61 + (i * 7) % 26) for i in range(20_000))
    tok = BPETokenizer().train(varied, vocab_size=300)
    # 300 BPE tokens (256 bytes + 44 merges) plus one special token.
    assert tok.vocab_size == 301
    assert tok.eot_id == 300


def test_training_stops_early_when_the_corpus_runs_out_of_pairs():
    """A tiny corpus cannot fill a large vocabulary; that must not be an error."""
    tok = BPETokenizer().train("abab\n", vocab_size=1000)
    assert tok.vocab_size < 1000
    assert tok.decode(tok.encode("abab\n")) == "abab\n"


def test_train_rejects_vocab_below_byte_alphabet():
    with pytest.raises(ValueError, match="at least 256"):
        BPETokenizer().train(CORPUS, vocab_size=255)


def test_special_token_encoding(tokenizer: BPETokenizer):
    text = f"first{END_OF_TEXT}second"
    ids = tokenizer.encode(text, allowed_special=True)
    assert tokenizer.eot_id in ids
    assert tokenizer.decode(ids) == text


def test_special_token_can_be_disabled(tokenizer: BPETokenizer):
    """Untrusted text must not be able to inject a document boundary."""
    ids = tokenizer.encode(f"a{END_OF_TEXT}b", allowed_special=False)
    assert tokenizer.eot_id not in ids
    assert tokenizer.decode(ids) == f"a{END_OF_TEXT}b"


def test_save_and_load_roundtrip(tokenizer: BPETokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.merges == tokenizer.merges
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.vocab_size == tokenizer.vocab_size
    text = "the quick brown fox 🦊"
    assert loaded.encode(text) == tokenizer.encode(text)


def test_decode_rejects_unknown_id(tokenizer: BPETokenizer):
    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([tokenizer.vocab_size + 10])


def test_ids_are_within_vocab(tokenizer: BPETokenizer):
    ids = tokenizer.encode(CORPUS)
    assert ids and max(ids) < tokenizer.vocab_size
