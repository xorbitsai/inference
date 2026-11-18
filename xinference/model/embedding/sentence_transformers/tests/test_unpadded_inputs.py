# Copyright 2022-2026 Xinference Holdings Pte. Ltd

# CI: linux

import asyncio
import sys
from types import ModuleType

import pytest
import torch

from ...core import EmbeddingModelFamilyV2, TransformersEmbeddingSpecV1
from .. import core


class _InputModule(torch.nn.Module):
    def __init__(self, supports_unpadding, attention):
        super().__init__()
        self.attention = attention
        if supports_unpadding:
            self.unpad_inputs = None


class _Encoder(torch.nn.Module):
    device = torch.device("cpu")
    default_prompt_name = None
    prompts: dict[str, str] = {}

    def __init__(self, path, *, model_kwargs=None, truncate_dim=None, **kwargs):
        super().__init__()
        self.input_module = _InputModule(
            self.supports_unpadding, model_kwargs["attn_implementation"]
        )
        self.truncate_dim = truncate_dim

    def get_embedding_dimension(self):
        return 2

    def tokenize(self, sentences):
        lengths = [len(text) + 2 for text in sentences]
        if getattr(self.input_module, "unpad_inputs", False) is not False:
            # FA2 packing removes attention_mask and merges the input rows.
            return {
                "input_ids": torch.ones((1, sum(lengths)), dtype=torch.long),
                "cu_seq_lens_q": torch.tensor([0, *torch.tensor(lengths).cumsum(0)]),
            }
        mask = torch.zeros((len(sentences), max(lengths)), dtype=torch.long)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        return {"attention_mask": mask, "input_ids": mask.clone()}

    def forward(self, features, **kwargs):
        counts = features["attention_mask"].sum(1).float()
        return {
            **features,
            "sentence_embedding": torch.stack((counts, counts + 1), dim=1),
            "token_embeddings": features["input_ids"].float().unsqueeze(-1),
        }


@pytest.fixture(params=[False, True], ids=["older-st", "st-auto-unpadding"])
def model(monkeypatch, request):
    encoder = type("Encoder", (_Encoder,), {"supports_unpadding": request.param})
    st = ModuleType("sentence_transformers")
    st.__version__ = "6.3.0" if request.param else "5.1.1"
    st.SentenceTransformer = encoder
    util = ModuleType("sentence_transformers.util")
    util.batch_to_device = lambda features, device: features
    util.truncate_embeddings = lambda values, dim: values[..., :dim]
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util)
    monkeypatch.setattr(core, "is_flash_attn_available", lambda: True)
    family = EmbeddingModelFamilyV2(
        version=2,
        model_name="Qwen3-Embedding-0.6B",
        dimensions=2,
        max_tokens=512,
        language=["en"],
        model_specs=[
            TransformersEmbeddingSpecV1(
                model_format="pytorch", model_id="unused", quantization="none"
            )
        ],
    )
    result = core.SentenceTransformerEmbeddingModel("test", "/unused", family)
    monkeypatch.setattr(result, "_clean_cache_if_needed", lambda _: None)
    result.load()
    assert result._model.input_module.attention == "flash_attention_2"
    return result


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("token_embeddings", [False, True])
def test_fa2_padding_preserves_usage_order_and_output_modes(
    model, batch_size, token_embeddings
):
    texts = ["short", "a much longer sentence", ""]
    kwargs = (
        {"output_value": "token_embeddings"} if token_embeddings else {"dimensions": 1}
    )
    result = model._create_embedding(
        texts, batch_size=batch_size, show_progress_bar=False, **kwargs
    )
    assert result["usage"] == {"prompt_tokens": 33, "total_tokens": 33}
    assert [item["index"] for item in result["data"]] == [0, 1, 2]
    if token_embeddings:
        assert [len(item["embedding"]) for item in result["data"]] == [7, 24, 2]
    else:
        assert [item["embedding"] for item in result["data"]] == [[1.0]] * 3


async def test_fa2_padding_keeps_concurrent_request_token_usage(model):
    try:
        first, second = await asyncio.gather(
            model.create_embedding(["a", "bb"], show_progress_bar=False),
            model.create_embedding(["long text"], show_progress_bar=False),
        )
    finally:
        model._process_batch_task.cancel()
        await asyncio.gather(model._process_batch_task, return_exceptions=True)
    assert first["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
    assert second["usage"] == {"prompt_tokens": 11, "total_tokens": 11}
    assert [item["index"] for item in first["data"]] == [0, 1]
    assert [item["index"] for item in second["data"]] == [0]
