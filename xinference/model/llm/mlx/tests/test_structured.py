# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import pytest

from ..structured import (
    JSONLogitsProcessor,
    ReasoningBudgetLogitsProcessor,
    json_schema,
)


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


@pytest.mark.parametrize("prompt", [[], [10, 11, 12]])
@pytest.mark.parametrize("batched", [False, True])
def test_budget_forces_multitoken_closer_then_releases_answer(prompt, batched):
    from types import SimpleNamespace

    mx = pytest.importorskip("mlx.core")
    tokenizer = SimpleNamespace(
        encode=lambda text, **kw: list(text.encode()), eos_token_id=0
    )
    processor = ReasoningBudgetLogitsProcessor(tokenizer, "</>", 2)
    logits = mx.zeros((1, 256) if batched else (256,))
    tokens = list(prompt)
    assert mx.isneginf(processor(mx.array(tokens), logits).reshape(-1)[0]).item()
    tokens.append(ord("x"))
    processor(mx.array(tokens), logits)
    tokens.append(ord("x"))
    for token in b"</>":
        masked = processor(mx.array(tokens), logits)
        assert masked.shape == logits.shape
        assert mx.argmax(masked).item() == token
        assert mx.isfinite(masked).sum().item() == 1
        tokens.append(token)
    assert mx.isfinite(processor(mx.array(tokens), logits)).all().item()


def test_budget_accepts_natural_closer_and_finishes_partial_closer():
    from types import SimpleNamespace

    mx = pytest.importorskip("mlx.core")
    tokenizer = SimpleNamespace(
        encode=lambda text, **kw: list(text.encode()), eos_token_id=0
    )
    logits = mx.zeros((256,))
    for budget in (16, 2):
        processor = ReasoningBudgetLogitsProcessor(tokenizer, "</>", budget)
        tokens = []
        processor(mx.array(tokens), logits)
        for token in b"</>":
            masked = processor(mx.array(tokens), logits)
            assert mx.isfinite(masked[token]).item()
            tokens.append(token)
        assert mx.isfinite(processor(mx.array(tokens), logits)).all().item()


@pytest.mark.parametrize("kind", ["json", "budget"])
def test_processors_copy_only_new_tokens_to_host(monkeypatch, kind):
    import sys
    from types import SimpleNamespace

    import numpy as np

    copied = []

    class Tokens:
        def __init__(self, values):
            self.values = values

        def reshape(self, *args):
            return self

        def __len__(self):
            return len(self.values)

        def __getitem__(self, item):
            return Tokens(self.values[item])

        def tolist(self):
            copied.append(len(self.values))
            return self.values

    fake_core = SimpleNamespace()
    fake_guidance = SimpleNamespace(apply_token_bitmask=lambda logits, mask: logits)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "llguidance", SimpleNamespace(mlx=fake_guidance))
    monkeypatch.setitem(sys.modules, "llguidance.mlx", fake_guidance)
    if kind == "budget":
        processor = ReasoningBudgetLogitsProcessor(
            SimpleNamespace(encode=lambda *args, **kwargs: [88], eos_token_ids=[]),
            "</think>",
            100,
        )
    else:
        processor = object.__new__(JSONLogitsProcessor)
        processor._seen = None
        processor._reasoning = True
        processor._reasoning_end = [88]
        processor._tail = []
        processor._eos = []
    logits = np.zeros(128)
    assert processor(Tokens([99] * 4096), logits) is logits
    assert processor(Tokens([99] * 4096 + [10, 11]), logits) is logits
    assert copied == [0, 2]
