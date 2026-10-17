# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import pytest

from ..structured import JSONLogitsProcessor, json_schema


def test_schema_validation_preserves_empty_schema():
    assert json_schema({"type": "json_schema", "json_schema": {"schema": {}}}) == {}
    assert json_schema({"type": "json_object"}) == {"type": "object"}
    with pytest.raises(ValueError, match="JSON schema object"):
        json_schema({"type": "json_schema"})


@pytest.mark.parametrize("reasoning", [False, True])
def test_native_mask_enforces_schema_after_reasoning(reasoning):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("llguidance")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {value: index for index, value in enumerate(alphabet)}
    vocab["<eos>"] = len(vocab)
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<eos>")
    processor = JSONLogitsProcessor(
        tokenizer, {"const": {"value": 42}}, "</think>" if reasoning else None
    )
    tokens = []
    logits = mx.zeros((len(vocab),))
    initial = processor(mx.array(tokens, dtype=mx.int32), logits)
    if reasoning:
        assert mx.isneginf(initial[tokenizer.eos_token_id]).item()
        tokens += tokenizer.encode("some reasoning</think>", add_special_tokens=False)
    valid = tokenizer.encode('{"value":42}', add_special_tokens=False)
    for token in valid:
        constrained = processor(mx.array(tokens, dtype=mx.int32), logits)
        assert not mx.isinf(constrained[token]).item()
        if not tokens or (
            reasoning
            and tokens[-1] == tokenizer.encode(">", add_special_tokens=False)[0]
        ):
            invalid = tokenizer.encode("X", add_special_tokens=False)[0]
            assert mx.isneginf(constrained[invalid]).item()
        tokens.append(token)
    constrained = processor(mx.array(tokens, dtype=mx.int32), logits)
    assert not mx.isinf(constrained[tokenizer.eos_token_id]).item()
    assert processor.matcher.is_accepting()
