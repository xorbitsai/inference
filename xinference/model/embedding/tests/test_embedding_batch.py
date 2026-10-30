# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CI: linux

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from xoscar.batch import _ExtensibleWrapper

from ....types import Embedding, EmbeddingData, EmbeddingUsage
from ...utils import virtualenv_discovery_var
from .. import core as embedding_core
from .. import generate_engine_config_by_model_name
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, TransformersEmbeddingSpecV1
from ..embed_family import match_embedding

_STUB_SPEC = EmbeddingModelFamilyV2(
    version=2,
    model_name="stub-embedding",
    dimensions=8,
    max_tokens=512,
    language=["en"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="stub/stub-embedding",
            model_revision="0000000000000000000000000000000000000000",
            quantization="none",
        )
    ],
)


class _StubEmbeddingModel(EmbeddingModel):
    """EmbeddingModel whose ``_create_embedding`` needs no model download.

    It mimics the real engines (sentence_transformers, flag, vllm, ...) by
    numbering ``EmbeddingData.index`` with ``enumerate`` over the *entire* input
    list it receives. Under batching that list is the concatenation of every
    caller's inputs, so the produced indices are global to the batch — exactly
    the behaviour the splitter in ``EmbeddingModel.create_embedding.batch`` must
    re-normalize back to ``[0, n)`` per caller.
    """

    @classmethod
    def check_lib(cls):
        return True

    @classmethod
    def match_json(cls, model_family, model_spec, quantization):
        return True

    def load(self):
        pass

    def _create_embedding(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        data = [
            EmbeddingData(index=i, object="embedding", embedding=[float(i)])
            for i in range(len(sentences))
        ]
        return Embedding(
            object="list",
            model=self._model_uid,
            model_replica=self._model_uid,
            data=data,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )


def _make_stub_model() -> _StubEmbeddingModel:
    return _StubEmbeddingModel("stub-uid", "/tmp/stub-embedding", _STUB_SPEC)


def test_create_embedding_batch_normalizes_index_per_request():
    """Regression for intermittent invalid embedding indices under concurrency.

    ``create_embedding.batch`` concatenates inputs from concurrent calls, runs
    ``_create_embedding`` once over the full list, then slices the result back
    per call. Each caller must receive indices covering ``[0, n)``; a call whose
    inputs land at a non-zero offset would otherwise leak the global batch index
    (e.g. ``[1, 2]`` instead of ``[0, 1]``) — the production defect captured in
    ``optimize/report``.
    """
    model = _make_stub_model()
    results = model.create_embedding.batch(
        _ExtensibleWrapper.delay(["a"]),  # call 0: 1 input, offset 0
        _ExtensibleWrapper.delay(["b", "c"]),  # call 1: 2 inputs, offset 1
    )

    assert [d["index"] for d in results[0]["data"]] == [0]
    # call 1's slice is data[1:3]; without re-normalization this is [1, 2]
    assert [d["index"] for d in results[1]["data"]] == [0, 1]
    # the vectors themselves must be unchanged — only the index label is fixed
    assert [d["embedding"] for d in results[1]["data"]] == [[1.0], [2.0]]


def test_create_embedding_batch_single_input_at_nonzero_offset():
    """A single-input call batched behind another request sits at offset > 0.

    Its response must still report ``index == [0]`` rather than the global
    ``[1]``. This guards against a fix that only handles multi-input calls.
    """
    model = _make_stub_model()
    results = model.create_embedding.batch(
        _ExtensibleWrapper.delay(["first"]),  # offset 0
        _ExtensibleWrapper.delay(["second"]),  # offset 1, single input
    )

    assert [d["index"] for d in results[0]["data"]] == [0]
    assert [d["index"] for d in results[1]["data"]] == [0]


def test_create_embedding_batch_treats_dict_as_one_multimodal_input():
    model = _make_stub_model()
    results = model.create_embedding.batch(
        _ExtensibleWrapper.delay({"image": "one.jpg", "text": "first"}),
        _ExtensibleWrapper.delay({"video": "two.mp4", "text": "second"}),
    )

    assert len(results) == 2
    assert [len(result["data"]) for result in results] == [1, 1]
    assert [result["data"][0]["index"] for result in results] == [0, 0]


class _CountingEncoder:
    device = "cpu"
    default_prompt_name = None
    prompts = {"query": "Q: "}
    truncate_dim = None

    def __init__(self):
        self.max_length = 512
        self.tokenized = []
        self.forwarded = []

    def eval(self):
        return self

    def tokenize(self, texts):
        self.tokenized.append(texts)
        lengths = [min(len(text), self.max_length) for text in texts]
        mask = torch.zeros((len(texts), max(lengths)), dtype=torch.long)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        return {"input_ids": mask.clone(), "attention_mask": mask}

    def forward(self, features, **kwargs):
        counts = features["attention_mask"].sum(1).float()
        self.forwarded.append(counts.tolist())
        return {
            **features,
            "sentence_embedding": torch.stack((counts, counts + 1), dim=1),
        }


def _select_model(
    engine, model_name="bge-small-en-v1.5", model_format="pytorch", quantization="none"
):
    family = match_embedding(model_name, model_format, quantization)
    engines = {}
    token = virtualenv_discovery_var.set(True)
    try:
        generate_engine_config_by_model_name(family, engines)
    finally:
        virtualenv_discovery_var.reset(token)
    cls = engines[model_name][engine][0]["embedding_class"]
    model = cls("replica", "/unused", family)
    model.batch_interval = 0
    model._clean_cache_if_needed = Mock()
    return model


@pytest.fixture
def counting_model(monkeypatch):
    st = ModuleType("sentence_transformers")
    st.SentenceTransformer = _CountingEncoder
    util = ModuleType("sentence_transformers.util")
    util.batch_to_device = lambda features, device: features
    util.truncate_embeddings = lambda values, dim: values[..., :dim]
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util)
    model = _select_model("sentence_transformers")
    model._model = _CountingEncoder()
    return model


async def _assert_request_usage(
    model, requests, expected, concurrent_expected=None, **kwargs
):
    kwargs = {"model_uid": "public", **kwargs}
    try:
        isolated = [await model.create_embedding(texts, **kwargs) for texts in requests]
        concurrent = await asyncio.gather(
            *(model.create_embedding(texts, **kwargs) for texts in requests)
        )
    finally:
        task = model._process_batch_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert [r["usage"] for r in isolated] == [
        {"prompt_tokens": count, "total_tokens": count} for count in expected
    ]
    # Check the vectors, local indices, and public response fields before usage.
    for texts, alone, together in zip(requests, isolated, concurrent):
        n = 1 if isinstance(texts, str) else len(texts)
        assert type(alone) is type(together) is dict
        assert together["data"] == alone["data"]
        assert [item["index"] for item in together["data"]] == list(range(n))
        assert set(together) == {"object", "model", "model_replica", "data", "usage"}
        assert together["object"] == "list"
        assert together["model"] == "public"
        assert together["model_replica"] == "replica"
    if concurrent_expected is None:
        concurrent_expected = expected
    assert [r["usage"] for r in concurrent] == [
        {"prompt_tokens": count, "total_tokens": count} for count in concurrent_expected
    ]


@pytest.mark.parametrize(
    "requests,expected",
    [
        (["a", "b" * 9], [1, 9]),
        ([["a", "b" * 9], "ccc"], [10, 3]),
        (["ccc", ["b" * 9, "a"]], [3, 10]),
    ],
)
async def test_concurrent_embedding_usage_is_request_local(
    counting_model, requests, expected
):
    await _assert_request_usage(
        counting_model, requests, expected, batch_size=2, normalize_embeddings=False
    )
    # A shared native forward proves the concurrent callers entered one batch.
    if expected == [1, 9]:
        assert counting_model._model.forwarded[-1] == [9.0, 1.0]
    else:
        assert counting_model._model.forwarded[-2:] == [[9.0, 3.0], [1.0]]


@pytest.mark.parametrize(
    "prompt_kwargs", [{"prompt": "Q: "}, {"prompt_name": "query"}, {}]
)
async def test_embedding_usage_counts_encoded_prompt_and_truncation(
    counting_model, prompt_kwargs
):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(text[: kwargs["max_length"]])}

        def decode(self, ids, **kwargs):
            return "".join(ids)

    counting_model._model.max_length = 6
    counting_model._model.default_prompt_name = "query"
    counting_model._tokenizer = Tokenizer()
    await _assert_request_usage(
        counting_model,
        ["a", "b" * 9],
        [4, 6],
        truncate_prompt_tokens=4,
        normalize_embeddings=False,
        **prompt_kwargs,
    )
    assert counting_model._model.forwarded[-1] == [6.0, 4.0]
    # One prompt and one input pass per batch; usage needs no extra tokenization.
    assert len(counting_model._model.tokenized) == 6


@pytest.mark.parametrize("model_name", ["jina-clip-v2", "jina-embeddings-v4"])
async def test_clip_text_usage_preserves_isolated_aggregate(counting_model, model_name):
    model = _select_model("sentence_transformers", model_name)
    model._model = _CountingEncoder()
    await _assert_request_usage(
        model,
        [["a", "b" * 9], "ccc"],
        [18, 3],
        concurrent_expected=[-1, -1],
        batch_size=2,
        normalize_embeddings=False,
    )


def test_mixed_unknown_and_known_input_counts_stay_unknown(counting_model):
    model = _select_model("sentence_transformers", "jina-clip-v2")
    encoder = _CountingEncoder()
    tokenize = encoder.tokenize

    def tokenize_with_media(texts):
        features = tokenize(texts)
        # An opaque media tensor cannot be mapped to both rows of this batch.
        features["pixel_values"] = torch.ones(1, 3, 2, 2)
        return features

    encoder.tokenize = tokenize_with_media
    model._model = encoder
    results = model._func.batch(
        _ExtensibleWrapper.delay(["a", "b" * 9], batch_size=2),
        _ExtensibleWrapper.delay("ccc", batch_size=2),
    )
    assert [r["usage"] for r in results] == [
        {"prompt_tokens": -1, "total_tokens": -1}
    ] * 2
    assert encoder.forwarded == [[9.0, 3.0], [1.0]]
    alone = model._func.batch(_ExtensibleWrapper.delay(["a", "b" * 9], batch_size=2))[0]
    assert alone["usage"] == {"prompt_tokens": 30, "total_tokens": 30}
    for together in results:
        assert type(alone) is type(together) is dict


@pytest.mark.parametrize(
    "model_name", ["WeMM-Embedding-2B", "jina-embeddings-v5-omni-nano"]
)
async def test_native_multimodal_item_count_usage_is_request_local(
    counting_model, monkeypatch, model_name
):
    from ..sentence_transformers import core as st_core

    monkeypatch.setattr(st_core, "ensure_wemm_video_reader", lambda: None)
    model = _select_model("sentence_transformers", model_name)

    def encode(inputs, **kwargs):
        texts = [
            sample if isinstance(sample, str) else sample[0]["content"][0]["text"]
            for sample in inputs
        ]
        return torch.tensor([[float(len(text))] for text in texts]).numpy()

    model._model = SimpleNamespace(encode=encode)
    # These native wrappers already report input items, rather than token counts.
    await _assert_request_usage(model, [["a", "b" * 9], "ccc"], [2, 1])


async def test_legacy_vllm_batch_usage_is_request_local(monkeypatch):
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.18.0"
    vllm.PoolingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    model = _select_model("vllm")

    def embed(texts, **kwargs):
        return [
            SimpleNamespace(
                prompt_token_ids=list(range(len(text))),
                outputs=SimpleNamespace(embedding=[float(len(text))]),
            )
            for text in texts
        ]

    model._model = SimpleNamespace(embed=embed)
    await _assert_request_usage(model, [["a", "b" * 9], "ccc"], [10, 3])


async def test_transformers_gemma_usage_counts_each_processed_input():
    model = _select_model("transformers", "embeddinggemma-2")

    def process(sample, **kwargs):
        count = min(len(sample[0]["content"][0]["text"]), 5)
        features = {"attention_mask": torch.ones(1, count)}
        return SimpleNamespace(to=lambda **kwargs: features)

    model._processor = SimpleNamespace(apply_chat_template=process)
    model._model = Mock(
        dtype=torch.float32,
        side_effect=lambda **features: SimpleNamespace(
            last_hidden_state=torch.full(
                (1, features["attention_mask"].shape[1], 768),
                float(features["attention_mask"].sum()),
            )
        ),
    )
    await _assert_request_usage(
        model, [["a", "b" * 9], "ccc"], [6, 3], normalize_embeddings=False
    )


async def test_native_sentence_transformers_gemma_aggregate_is_unknown_per_slice():
    model = _select_model("sentence_transformers", "embeddinggemma-2")
    callbacks = []
    model._model = Mock()
    model._model.register_forward_pre_hook.side_effect = (
        lambda callback: callbacks.append(callback) or Mock()
    )

    def encode(messages, **kwargs):
        counts = [len(sample[0]["content"][0]["text"]) for sample in messages]
        callbacks[-1](None, ({"attention_mask": torch.tensor([counts])},))
        return torch.tensor(counts).float().unsqueeze(1).expand(-1, 768)

    model._model.encode.side_effect = encode
    await _assert_request_usage(
        model,
        [["a", "b" * 9], "ccc"],
        [10, 3],
        concurrent_expected=[-1, -1],
        normalize_embeddings=False,
    )


async def test_flag_keeps_unknown_usage_sentinels(monkeypatch):
    from ..flag import core as flag_core

    flag = ModuleType("FlagEmbedding")
    flag.BGEM3FlagModel = object
    monkeypatch.setitem(sys.modules, "FlagEmbedding", flag)
    monkeypatch.setattr(flag_core, "FLAG_EMBEDDER_MODEL_LIST", ["bge-m3"])
    model = _select_model("flag", "bge-m3")
    model._model = SimpleNamespace(
        encode=lambda texts, **kwargs: {
            "dense_vecs": torch.tensor([[float(len(text))] for text in texts])
        }
    )
    await _assert_request_usage(model, [["a", "b" * 9], "ccc"], [-1, -1])


async def test_llama_cpp_aggregate_is_unknown_per_slice():
    model = _select_model("llama.cpp", "Qwen3-Embedding-0.6B", "ggufv2", "Q8_0")

    def embed(data):
        counts = [len(text) for text in data["input"]]
        return Embedding(
            object="list",
            model=data.get("model"),
            model_replica="replica",
            data=[
                EmbeddingData(index=i, object="embedding", embedding=[float(count)])
                for i, count in enumerate(counts)
            ],
            usage=EmbeddingUsage(prompt_tokens=sum(counts), total_tokens=sum(counts)),
        )

    model._llm = SimpleNamespace(handle_embeddings=embed)
    model._executor = SimpleNamespace(submit=lambda fn: fn())
    await _assert_request_usage(
        model, [["a", "b" * 9], "ccc"], [10, 3], concurrent_expected=[-1, -1]
    )


@pytest.mark.parametrize("tokens", [-7, -1, 0, 13])
def test_batch_without_per_input_counts_preserves_unknown_usage(monkeypatch, tokens):
    model = _make_stub_model()
    create = model._create_embedding

    def create_with_usage(sentences, **kwargs):
        result = create(sentences, **kwargs)
        result["usage"] = EmbeddingUsage(prompt_tokens=tokens, total_tokens=tokens)
        return result

    monkeypatch.setattr(model, "_create_embedding", create_with_usage)
    # An isolated aggregate remains valid, including engines reporting sentinels.
    alone = model.create_embedding.batch(_ExtensibleWrapper.delay(["a", "bb"]))
    assert alone[0]["usage"] == {"prompt_tokens": tokens, "total_tokens": tokens}
    results = model.create_embedding.batch(
        _ExtensibleWrapper.delay(["a", "bb"]), _ExtensibleWrapper.delay(["ccc"])
    )
    expected = tokens if tokens < 0 else -1
    assert [r["usage"] for r in results] == [
        {"prompt_tokens": expected, "total_tokens": expected}
    ] * 2


def test_cache_cleanup_keeps_count_and_token_triggers_while_rate_limiting_gc(
    monkeypatch,
):
    model = _make_stub_model()
    clock = [100.0]
    events = []
    monkeypatch.setattr(embedding_core, "EMBEDDING_EMPTY_CACHE_COUNT", 3)
    monkeypatch.setattr(embedding_core, "EMBEDDING_EMPTY_CACHE_TOKENS", 10)
    monkeypatch.setattr(embedding_core, "EMBEDDING_GC_INTERVAL", 60)
    # Keep process-wide clocks and GC available to concurrent runtime threads.
    monkeypatch.setattr(
        embedding_core, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(
        embedding_core, "gc", SimpleNamespace(collect=lambda: events.append("gc"))
    )
    monkeypatch.setattr(embedding_core, "empty_cache", lambda: events.append("cache"))

    model._clean_cache_if_needed(0)
    assert events == []
    model._clean_cache_if_needed(10)  # Token trigger, first full collection.
    model._clean_cache_if_needed(0)  # Count trigger still releases GPU cache.
    clock[0] = 159.9
    model._clean_cache_if_needed(10)
    assert events == ["gc", "cache", "cache", "cache"]

    clock[0] = 160.0
    model._clean_cache_if_needed(0)  # The GC interval alone is not a cache trigger.
    assert events == ["gc", "cache", "cache", "cache"]
    model._clean_cache_if_needed(10)
    assert events == ["gc", "cache", "cache", "cache", "gc", "cache"]


def test_gc_interval_zero_restores_collection_on_every_cleanup(monkeypatch):
    model = _make_stub_model()
    collect = Mock()
    release = Mock()
    monkeypatch.setattr(embedding_core, "EMBEDDING_GC_INTERVAL", 0)
    monkeypatch.setattr(embedding_core, "gc", SimpleNamespace(collect=collect))
    monkeypatch.setattr(embedding_core, "empty_cache", release)
    for _ in range(5):
        model._clean_cache_if_needed(embedding_core.EMBEDDING_EMPTY_CACHE_TOKENS)
    assert collect.call_count == release.call_count == 5


def test_gc_cooldown_is_per_model(monkeypatch):
    collect = Mock()
    release = Mock()
    monkeypatch.setattr(embedding_core, "EMBEDDING_GC_INTERVAL", 60)
    monkeypatch.setattr(embedding_core, "gc", SimpleNamespace(collect=collect))
    monkeypatch.setattr(embedding_core, "empty_cache", release)
    first, second = _make_stub_model(), _make_stub_model()
    for model in (first, first, second):
        model._clean_cache_if_needed(embedding_core.EMBEDDING_EMPTY_CACHE_TOKENS)
    assert collect.call_count == 2
    assert release.call_count == 3


def test_consecutive_large_batches_keep_gc_cooldown_and_release_gpu_cache(
    monkeypatch,
):
    model = _make_stub_model()
    collect = Mock()
    release = Mock()
    monkeypatch.setattr(embedding_core, "EMBEDDING_GC_INTERVAL", 60)
    monkeypatch.setattr(
        embedding_core, "time", SimpleNamespace(monotonic=lambda: 100.0)
    )
    monkeypatch.setattr(embedding_core, "gc", SimpleNamespace(collect=collect))
    monkeypatch.setattr(embedding_core, "empty_cache", release)
    for _ in range(100):
        model._clean_cache_if_needed(embedding_core.EMBEDDING_EMPTY_CACHE_TOKENS)
    assert collect.call_count == 1
    assert release.call_count == 100
