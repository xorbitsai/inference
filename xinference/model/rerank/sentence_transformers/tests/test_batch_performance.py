import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ... import core as rerank_core
from ...rerank_family import BUILTIN_RERANK_MODELS
from ..core import SentenceTransformerRerankModel


def score(pair):
    return float(sum(ord(char) for text in pair for char in str(text))) / 10000


class FakeCrossEncoder:
    def __init__(self):
        self.calls = []
        self.batches = []

    def predict(self, pairs, batch_size=32, **kwargs):
        self.calls.append((pairs, batch_size, kwargs))
        for start in range(0, len(pairs), batch_size):
            self.batches.append(pairs[start : start + batch_size])
        return torch.tensor([score(pair) for pair in pairs])


@pytest.fixture
def model():
    family = BUILTIN_RERANK_MODELS["bge-reranker-base"][0].copy(deep=True)
    model = SentenceTransformerRerankModel(
        "test", "/model", family, "none", batch_interval=0.01
    )
    model._model = FakeCrossEncoder()

    def tokenize(pairs, **kwargs):
        lengths = [sum(len(text) for text in pair) + 3 for pair in pairs]
        mask = torch.zeros((len(pairs), max(lengths, default=0)), dtype=torch.long)
        for row, length in zip(mask, lengths):
            row[:length] = 1
        return {"attention_mask": mask}

    model._tokenizer = Mock(side_effect=tokenize)
    return model


@pytest.mark.parametrize("return_len", [False, True])
def test_mixed_pairs_restore_scores_and_token_usage(model, return_len):
    documents = ["x" * (1024 + i if i % 8 == 0 else 40 + i) for i in range(32)]
    queries = ["q" + str(i) for i in range(32)]
    scores = model._rerank(documents, queries, None, None, False, return_len)
    assert [len(batch) for batch in model._model.batches] == [4, 28]
    torch.testing.assert_close(
        scores, torch.tensor([score(pair) for pair in zip(queries, documents)])
    )
    assert model._tokenizer.call_count == int(return_len)
    tokens = [len(q) + len(d) + 3 for q, d in zip(queries, documents)]
    assert model._get_input_tokens() == (tokens if return_len else [])
    assert model._get_n_tokens() == (sum(tokens) if return_len else 0)


def test_short_mixed_pairs_keep_one_forward(model):
    pairs = [["q", "x" * (10 + i * 110 // 31)] for i in range(32)]
    model._predict_sentence_pairs(pairs)
    assert [len(batch) for batch in model._model.batches] == [32]
    assert len(model._model.calls) == 1


def test_uniform_pairs_keep_native_batch_size(model):
    model._predict_sentence_pairs([["q", "x" * 100]] * 65)
    assert [len(batch) for batch in model._model.batches] == [32, 32, 1]
    assert len(model._model.calls) == 1


@pytest.mark.parametrize("longest,expected", [(256, [2]), (257, [1, 1])])
def test_length_split_has_an_absolute_floor(model, longest, expected):
    model._predict_sentence_pairs([["", "x" * longest], ["", "y"]])
    assert [len(batch) for batch in model._model.batches] == expected


def test_prediction_kwargs_and_multilabel_shape_are_preserved(model):
    pairs = [["q", "a"], ["q", "x" * 1024], ["q", "bc"]]
    calls = []

    def predict(batch, **kwargs):
        calls.append(kwargs)
        return torch.tensor([[score(pair), -score(pair)] for pair in batch])

    model._model.predict = predict
    activation = torch.nn.Identity()
    result = model._predict_sentence_pairs(
        pairs, batch_size=3, prompt="prefix", activation_fn=activation
    )
    torch.testing.assert_close(
        result, torch.tensor([[score(pair), -score(pair)] for pair in pairs])
    )
    assert len(calls) == 2
    assert all(
        kwargs
        == dict(
            batch_size=3,
            prompt="prefix",
            activation_fn=activation,
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        for kwargs in calls
    )


@pytest.mark.parametrize("structured", [False, True])
def test_multimodal_or_structured_pairs_keep_native_prediction(model, structured):
    pairs = [["q", "x" * 1024], ["q", "short"]]
    if structured:
        pairs[0][1] = {"text": "long"}
    else:
        model.model_family.model_ability = ["rerank_vision"]
    result = model._predict_sentence_pairs(pairs)
    assert len(model._model.calls) == 1
    assert model._model.calls[0][0] == pairs
    torch.testing.assert_close(result, torch.tensor([score(pair) for pair in pairs]))


def test_unused_token_counts_are_reset_and_tokenizer_errors_do_not_break_scores(model):
    model._rerank(["a"], ["q"], None, None, False, True)
    assert model._get_n_tokens() > 0
    model._rerank(["a"], ["q"], None, None, False, False)
    assert model._get_n_tokens() == 0
    model._tokenizer.side_effect = ValueError("token counting failed")
    result = model._rerank(["a"], ["q"], None, None, False, True)
    torch.testing.assert_close(result, torch.tensor([score(["q", "a"])]))
    assert model._get_n_tokens() == 0


def test_standalone_rerank_preserves_top_n_documents_and_usage(model):
    result = model._func(["short", "x" * 1024], "q", 1, None, True, True)
    assert result["results"][0]["index"] == 1
    assert result["results"][0]["document"]["text"] == "x" * 1024
    assert result["meta"]["tokens"]["input_tokens"] == sum(
        len(d) + 4 for d in ["short", "x" * 1024]
    )
    assert model._counter == 1


@pytest.mark.asyncio
async def test_auto_batch_counts_inference_groups_and_preserves_request_results(
    model, monkeypatch
):
    collect, release = Mock(), Mock()
    monkeypatch.setattr(rerank_core, "RERANK_EMPTY_CACHE_COUNT", 10)
    monkeypatch.setattr(rerank_core, "RERANK_GC_INTERVAL", 60)
    monkeypatch.setattr(rerank_core, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(rerank_core.gc, "collect", collect)
    monkeypatch.setattr(rerank_core, "empty_cache", release)
    documents = ["x" * (1024 if i == 0 else 40 + i) for i in range(32)]
    try:
        for _ in range(20):
            results = await asyncio.gather(
                *[model.rerank([d], "q", None, None, True, True) for d in documents]
            )
            for document, result in zip(documents, results):
                assert result["results"][0]["index"] == 0
                assert result["results"][0]["document"]["text"] == document
                assert result["results"][0]["relevance_score"] == pytest.approx(
                    score(["q", document])
                )
                assert result["meta"]["tokens"]["input_tokens"] == len(document) + 4
            if model._counter == 1:
                assert collect.call_count == release.call_count == 0
        assert model._counter == 20
        assert collect.call_count == 1
        assert release.call_count == 2
    finally:
        model._process_batch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await model._process_batch_task


@pytest.mark.asyncio
async def test_different_kwargs_still_form_separate_inference_groups(model):
    try:
        one, two = await asyncio.gather(
            model.rerank(["a", "bbb"], "q", 1, None, False, False),
            model.rerank(["long", "x" * 1024], "qq", None, None, True, True),
        )
        assert len(one["results"]) == 1 and one["results"][0]["index"] == 1
        assert one["results"][0]["document"] is None
        assert one["meta"]["tokens"] is None
        assert len(two["results"]) == 2 and two["results"][0]["index"] == 1
        assert two["meta"]["tokens"]["input_tokens"] == sum(
            len(d) + 5 for d in ["long", "x" * 1024]
        )
        assert model._counter == 2
    finally:
        model._process_batch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await model._process_batch_task
