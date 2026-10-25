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

from unittest.mock import Mock

from xoscar.batch import _ExtensibleWrapper

from ....types import Embedding, EmbeddingData, EmbeddingUsage
from .. import core as embedding_core
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, TransformersEmbeddingSpecV1

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


def test_cache_cleanup_keeps_count_and_token_triggers_while_rate_limiting_gc(
    monkeypatch,
):
    model = _make_stub_model()
    clock = [100.0]
    events = []
    monkeypatch.setattr(embedding_core, "EMBEDDING_EMPTY_CACHE_COUNT", 3)
    monkeypatch.setattr(embedding_core, "EMBEDDING_EMPTY_CACHE_TOKENS", 10)
    monkeypatch.setattr(embedding_core, "EMBEDDING_GC_INTERVAL", 60)
    monkeypatch.setattr(embedding_core.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(embedding_core.gc, "collect", lambda: events.append("gc"))
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
    monkeypatch.setattr(embedding_core.gc, "collect", collect)
    monkeypatch.setattr(embedding_core, "empty_cache", release)
    for _ in range(5):
        model._clean_cache_if_needed(embedding_core.EMBEDDING_EMPTY_CACHE_TOKENS)
    assert collect.call_count == release.call_count == 5


def test_gc_cooldown_is_per_model(monkeypatch):
    collect = Mock()
    release = Mock()
    monkeypatch.setattr(embedding_core, "EMBEDDING_GC_INTERVAL", 60)
    monkeypatch.setattr(embedding_core.gc, "collect", collect)
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
    monkeypatch.setattr(embedding_core.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(embedding_core.gc, "collect", collect)
    monkeypatch.setattr(embedding_core, "empty_cache", release)
    for _ in range(100):
        model._clean_cache_if_needed(embedding_core.EMBEDDING_EMPTY_CACHE_TOKENS)
    assert collect.call_count == 1
    assert release.call_count == 100
