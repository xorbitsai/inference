# Copyright 2022-2026 Xinference Holdings Pte. Ltd

import asyncio

import pytest
import torch

from ...core import EmbeddingModelFamilyV2, TransformersEmbeddingSpecV1
from ..core import SentenceTransformerEmbeddingModel, _get_sentence_batch_ranges


def _vector(text):
    return [float(len(text)), float(sum(map(ord, text)) % 101), 1.0]


class _Encoder:
    device = torch.device("cpu")
    prompts: dict[str, str] = {}
    default_prompt_name = None
    truncate_dim = None

    def __init__(self):
        self.batches = []

    def eval(self):
        pass

    def tokenize(self, sentences):
        self.batches.append(sentences)
        lengths = [len(text) + 2 for text in sentences]
        mask = torch.zeros((len(sentences), max(lengths)), dtype=torch.long)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        return {
            "attention_mask": mask,
            "sentence_embedding": torch.tensor([_vector(s) for s in sentences]),
            "token_embeddings": mask.float().unsqueeze(-1),
        }

    def forward(self, features, **kwargs):
        return features


def _make_model(monkeypatch, abilities=None):
    family = EmbeddingModelFamilyV2(
        version=2,
        model_name="test-length-batching",
        dimensions=3,
        max_tokens=8192,
        language=["en"],
        model_ability=abilities or [],
        model_specs=[
            TransformersEmbeddingSpecV1(
                model_format="pytorch", model_id="test/test", quantization="none"
            )
        ],
    )
    model = SentenceTransformerEmbeddingModel("test", "/unused", family)
    model._model = _Encoder()
    monkeypatch.setattr(model, "_clean_cache_if_needed", lambda _: None)
    return model


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_length_batches_reduce_padding_and_preserve_order_and_usage(
    monkeypatch, batch_size
):
    model = _make_model(monkeypatch)
    texts = [f"{i:02d}" + "x" * (510 if i % 8 == 0 else 30) for i in range(32)]
    result = model._create_embedding(
        texts, batch_size=batch_size, show_progress_bar=False
    )
    expected = torch.nn.functional.normalize(torch.tensor([_vector(s) for s in texts]))
    actual = torch.tensor([item["embedding"] for item in result["data"]])
    torch.testing.assert_close(actual, expected)
    assert [item["index"] for item in result["data"]] == list(range(len(texts)))
    assert result["usage"]["prompt_tokens"] == sum(len(s) + 2 for s in texts)
    assert result["usage"]["total_tokens"] == result["usage"]["prompt_tokens"]
    assert max(map(len, model._model.batches)) <= batch_size
    padded_tokens = sum(
        len(batch) * max(len(s) + 2 for s in batch) for batch in model._model.batches
    )
    assert padded_tokens < len(texts) * (512 + 2)


def test_uniform_lengths_keep_full_batches(monkeypatch):
    model = _make_model(monkeypatch)
    model._create_embedding(["x" * 32] * 65, show_progress_bar=False)
    assert [len(batch) for batch in model._model.batches] == [32, 32, 1]


def test_short_mixed_queries_keep_one_forward_pass(monkeypatch):
    model = _make_model(monkeypatch)
    texts = ["x" * (10 + i * 110 // 31) for i in range(32)]
    result = model._create_embedding(texts, show_progress_bar=False)
    assert [len(batch) for batch in model._model.batches] == [32]
    expected = torch.nn.functional.normalize(torch.tensor([_vector(s) for s in texts]))
    torch.testing.assert_close(
        torch.tensor([item["embedding"] for item in result["data"]]), expected
    )
    assert result["usage"]["prompt_tokens"] == sum(len(text) + 2 for text in texts)


@pytest.mark.parametrize("longest,expected", [(256, [(0, 2)]), (257, [(0, 1), (1, 2)])])
def test_length_split_requires_an_absolute_floor(longest, expected):
    assert (
        _get_sentence_batch_ranges(["x" * longest, "y"], 32, split_by_length=True)
        == expected
    )


def test_prompt_and_dimensions_are_preserved_without_normalization(monkeypatch):
    model = _make_model(monkeypatch)
    model._model.truncate_dim = 2
    texts = ["short", "long " * 100]
    result = model._create_embedding(
        texts,
        prompt="query: ",
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    prompted = ["query: " + text for text in texts]
    assert [item["embedding"] for item in result["data"]] == [
        _vector(text)[:2] for text in prompted
    ]
    assert result["usage"]["prompt_tokens"] == sum(len(text) + 2 for text in prompted)


def test_multimodal_family_retains_fixed_batches_for_text(monkeypatch):
    model = _make_model(monkeypatch, ["embed_vision"])
    model._create_embedding(["x" * 512, "y"] * 16, show_progress_bar=False)
    assert [len(batch) for batch in model._model.batches] == [32]


def test_token_embeddings_retain_fixed_batches_and_unpadded_lengths(monkeypatch):
    model = _make_model(monkeypatch)
    texts = ["x" * 512, "y", "z" * 16]
    result = model._create_embedding(
        texts, output_value="token_embeddings", batch_size=2, show_progress_bar=False
    )
    assert [len(batch) for batch in model._model.batches] == [2, 1]
    assert [len(item["embedding"]) for item in result["data"]] == [514, 3, 18]


def test_batch_ranges_handle_empty_and_structured_inputs():
    assert _get_sentence_batch_ranges([], 32, split_by_length=True) == []
    assert _get_sentence_batch_ranges(["", "", ""], 2, split_by_length=True) == [
        (0, 2),
        (2, 3),
    ]
    assert _get_sentence_batch_ranges(
        [{"text": "x" * 512}, {"text": "y"}], 32, split_by_length=True
    ) == [(0, 2)]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_does_not_stall(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        _get_sentence_batch_ranges(["text"], batch_size, split_by_length=True)


async def test_concurrent_requests_are_split_back_after_length_batching(monkeypatch):
    model = _make_model(monkeypatch)
    requests = [["x" * 512, f"short {i}"] for i in range(16)]
    try:
        results = await asyncio.gather(
            *(model.create_embedding(texts) for texts in requests)
        )
    finally:
        model._process_batch_task.cancel()
        await asyncio.gather(model._process_batch_task, return_exceptions=True)
    for texts, result in zip(requests, results):
        expected = torch.nn.functional.normalize(
            torch.tensor([_vector(text) for text in texts])
        )
        torch.testing.assert_close(
            torch.tensor([item["embedding"] for item in result["data"]]), expected
        )
        assert [item["index"] for item in result["data"]] == [0, 1]
    assert sorted(map(len, model._model.batches)) == [16, 16]
