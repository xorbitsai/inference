# CI: linux

import random

import pytest

from xinference.model.llm.transformers.utils import _IncrementalDecoder

TEXT = (
    "Hello world, it's a test. Don't stop ! Are we there ? Yes , we 're here.\n"
    "Unicode: héllo wörld, 日本語のテキスト, emoji 😀👍🏽 and ß € Ω≈ç√∫.\n"
    "def f(x):\n    return x * 2  # it 's fine\n"
)


def _byte_level_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|im_end|>", "<|tool|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator([TEXT] * 20, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|im_end|>",
        additional_special_tokens=["<|tool|>"],
    )


def _full_decode(tokenizer, tokens):
    return tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )


@pytest.mark.parametrize("seed", range(5))
def test_incremental_decoder_matches_full_decode(seed):
    tokenizer = _byte_level_tokenizer()
    decoder = _IncrementalDecoder(tokenizer)
    assert decoder._incremental

    rng = random.Random(seed)
    special_ids = tokenizer.all_special_ids
    tokens = []
    for _ in range(30):
        start = rng.randrange(len(TEXT))
        tokens += tokenizer.encode(
            TEXT[start : start + rng.randint(1, 40)], add_special_tokens=False
        )
        if rng.random() < 0.3:
            tokens.append(rng.choice(special_ids))
    # Random token ids split multi-byte characters at arbitrary places.
    tokens += [rng.randrange(len(tokenizer)) for _ in range(100)]

    count = 0
    while count < len(tokens):
        count = min(len(tokens), count + rng.choice([1, 1, 2, 3]))
        assert decoder.decode(tokens, count) == _full_decode(tokenizer, tokens[:count])


def test_incremental_decoder_falls_back_without_byte_level_decoder():
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return "".join(map(str, tokens))

    decoder = _IncrementalDecoder(Tokenizer())
    assert not decoder._incremental
    assert decoder.decode([1, 2, 3], 2) == "12"


def test_incremental_decoder_context_must_be_suffix_of_full_text():
    # Space cleanup of runs like "' ' ' " depends on where a decode starts, so
    # the context decoded on its own can differ from the end of the full text.
    tokenizer = _byte_level_tokenizer()
    decoder = _IncrementalDecoder(tokenizer)
    text = "Example: " + "' " * 5
    tokens = [t for c in text for t in tokenizer.encode(c, add_special_tokens=False)]
    for count in range(1, len(tokens) + 1):
        assert decoder.decode(tokens, count) == _full_decode(tokenizer, tokens[:count])
