import shutil
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from .....client import Client
from ...cache_manager import RerankCacheManager
from ...core import RerankModelFamilyV2, TransformersRerankSpecV1
from ..core import VLLMRerankModel

TEST_MODEL_SPEC = RerankModelFamilyV2(
    version=2,
    model_name="bge-reranker-base",
    type="normal",
    max_tokens=512,
    language=["en", "zh"],
    model_specs=[
        TransformersRerankSpecV1(
            model_id="BAAI/bge-reranker-base",
            model_revision="465b4b7ddf2be0a020c8ad6e525b9bb1dbb708ae",
            model_format="pytorch",
        )
    ],
)


def test_qwen3_vl_load_configures_vllm(monkeypatch):
    from .. import core

    llm = MagicMock()
    llm.return_value.get_tokenizer.return_value = MagicMock()
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = llm
    fake_vllm.__version__ = "0.14.0"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(core, "is_vacc_available", lambda: False)

    model = object.__new__(VLLMRerankModel)
    model._kwargs = {}
    model._model_path = "/model"
    model.model_family = SimpleNamespace(model_name="Qwen3-VL-Reranker-2B")
    model.load()

    kwargs = llm.call_args.kwargs
    assert kwargs["runner"] == "pooling"
    assert kwargs["hf_overrides"] == {
        "architectures": ["Qwen3VLForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    assert "<|im_start|>system" in model._qwen3_vl_reranker_template


def test_qwen3_vl_rerank_converts_multimodal_inputs():
    model = object.__new__(VLLMRerankModel)
    model.model_family = SimpleNamespace(model_name="Qwen3-VL-Reranker-2B")
    model._model = MagicMock()
    model._counter = 0
    model._qwen3_vl_reranker_template = "template"

    first_output, second_output = MagicMock(), MagicMock()
    model._model.score.side_effect = [[first_output], [second_output]]

    outputs = model._rerank(
        documents=[
            {"image": "https://example.com/image.jpg"},
            {"video": "https://example.com/second-video.mp4"},
        ],
        query="query",
    )

    query = "query"
    first_document = {
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]
    }
    second_document = {
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/second-video.mp4"},
            }
        ]
    }
    score_kwargs = {"use_tqdm": False, "chat_template": "template"}
    assert model._model.score.call_args_list == [
        call(query, first_document, **score_kwargs),
        call(query, second_document, **score_kwargs),
    ]
    assert outputs == [first_output, second_output]


def test_qwen3_vl_rerank_rejects_unsupported_multimodal_pairs():
    model = object.__new__(VLLMRerankModel)
    model.model_family = SimpleNamespace(model_name="Qwen3-VL-Reranker-2B")
    model._model = MagicMock()
    model._counter = 0
    model._qwen3_vl_reranker_template = "template"

    with pytest.raises(ValueError, match="one content item"):
        model._rerank(
            documents=[{"text": "document", "image": "https://example.com/image.jpg"}],
            query="query",
        )

    with pytest.raises(ValueError, match="media in both query and document"):
        model._rerank(
            documents=[{"image": "https://example.com/image.jpg"}],
            query={"video": "https://example.com/video.mp4"},
        )


@pytest.mark.skipif(VLLMRerankModel.check_lib() != True, reason="vllm not installed")
def test_qwen3_vl_score_wrapper_is_unwrapped_by_vllm(monkeypatch):
    from vllm import LLM

    query = {
        "content": [
            {
                "type": "text",
                "text": "query",
            }
        ]
    }
    document = {
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            }
        ]
    }
    llm = object.__new__(LLM)
    llm.model_config = SimpleNamespace(
        runner_type="pooling",
        is_cross_encoder=True,
        hf_config=SimpleNamespace(num_labels=1),
        is_multimodal_model=True,
    )
    llm.get_tokenizer = MagicMock()

    def cross_encoding_score(_, __, data_1, data_2, *args, **kwargs):
        assert data_1 == query["content"]
        assert data_2 == document["content"]
        return []

    monkeypatch.setattr(LLM, "_cross_encoding_score", cross_encoding_score)
    assert LLM.score(llm, query, document, use_tqdm=False) == []


@pytest.mark.skipif(VLLMRerankModel.check_lib() != True, reason="vllm not installed")
def test_model():
    model_path = None
    try:
        model_path = RerankCacheManager(TEST_MODEL_SPEC).cache()
        model = VLLMRerankModel("mock", model_path, TEST_MODEL_SPEC, "none")

        query = "A man is eating pasta."
        # With all sentences in the corpus
        corpus = [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin.",
            "Two men pushed carts through the woods.",
            "A man is riding a white horse on an enclosed ground.",
            "A monkey is playing drums.",
            "A cheetah is running behind its prey.",
        ]
        model.load()
        scores = model.rerank(corpus, query, None, None, True, True)
        assert scores["results"][0]["index"] == 0
        assert scores["results"][0]["document"]["text"] == corpus[0]

    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


@pytest.mark.skipif(VLLMRerankModel.check_lib() != True, reason="vllm not installed")
def test_qwen3_vllm(setup):
    endpoint, _ = setup
    client = Client(endpoint)
    model_uid = client.launch_model(
        model_name="Qwen3-Reranker-0.6B",
        model_type="rerank",
        model_engine="vllm",
    )

    model = client.get_model(model_uid)
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]

    scores = model.rerank(corpus, query, return_documents=True)
    assert scores["results"][0]["index"] == 1
