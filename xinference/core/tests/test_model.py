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

import asyncio
import json
import types

import pytest
import pytest_asyncio
import xoscar as xo
from xoscar import create_actor_pool

from ..model import ModelActor

TEST_EVENT = None
TEST_VALUE = None


class MockModelFamily:
    def to_description(self) -> dict:
        return {}


class MockModel:
    def __init__(self):
        self.model_family = MockModelFamily()

    async def generate(self, prompt, **kwargs):
        global TEST_VALUE
        TEST_VALUE = True
        assert isinstance(TEST_EVENT, asyncio.Event)
        await TEST_EVENT.wait()
        yield {"test1": prompt}
        yield {"test2": prompt}


class _DescribedModelFamily:
    def __init__(self, model_type: str):
        self._model_type = model_type

    def to_description(self) -> dict:
        return {"model_type": self._model_type, "model_name": "test-video"}


class _DescribedModel:
    def __init__(self, model_type: str):
        self.model_family = _DescribedModelFamily(model_type)


def test_video_metrics_keep_selected_engine_label():
    actor = ModelActor(
        supervisor_address="test:123",
        worker_address="test:345",
        model=_DescribedModel("video"),  # type: ignore
        replica_model_uid="test-video-0",
        model_engine="MLX",
    )

    assert actor._metrics_labels["engine"] == "MLX"


class MockModelActor(ModelActor):
    def __init__(
        self,
        supervisor_address: str,
        worker_address: str,
        replica_model_uid: str,
    ):
        super().__init__(
            supervisor_address=supervisor_address,
            worker_address=worker_address,
            model=MockModel(),  # type: ignore
            replica_model_uid=replica_model_uid,
        )
        # This actor is constructed directly with a ready-to-use MockModel
        # (no launch/load path), so mark it ready to bypass the
        # _require_ready() guard added by the model state machine.
        self._model_state = "ready"
        self._lock = asyncio.locks.Lock()

    async def __pre_destroy__(self):
        pass

    async def record_metrics(self, name, op, kwargs):
        pass


@pytest_asyncio.fixture
async def setup_pool():
    pool = await create_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}", n_process=0
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_concurrent_call(setup_pool):
    pool = setup_pool
    addr = pool.external_address

    global TEST_EVENT
    TEST_EVENT = asyncio.Event()

    worker: xo.ActorRefType[MockModelActor] = await xo.create_actor(  # type: ignore
        MockModelActor,
        address=addr,
        uid=MockModelActor.default_uid(),
        supervisor_address="test:123",
        worker_address="test:345",
        replica_model_uid="test_model",
    )

    await worker.generate("test_prompt1")
    assert TEST_VALUE is not None
    # This request is waiting for the TEST_EVENT, so the queue is empty.
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 0
    await worker.generate("test_prompt3")
    # This request is waiting in the queue because the previous request is waiting for TEST_EVENT.
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 1

    async def _check():
        gen = await worker.generate("test_prompt2")
        result = []
        async for g in gen:
            result.append(g)
        assert result == [
            b'data: {"test1": "test_prompt2"}\r\n\r\n',
            b'data: {"test2": "test_prompt2"}\r\n\r\n',
        ]

    check_task = asyncio.create_task(_check())
    await asyncio.sleep(2)
    assert not check_task.done()
    # Pending 2 requests: test_prompt3 and test_prompt2
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 2
    TEST_EVENT.set()
    await check_task
    pending_count = await worker.get_pending_requests_count()
    assert pending_count == 0


def _gguf_spec_fields():
    return {
        "quantization": "Q8_0",
        "model_file_name_template": "m.gguf",
        "model_file_name_split_template": None,
        "quantization_parts": None,
        "model_id": None,
        "model_uri": None,
        "model_revision": None,
    }


def _make_rerank_model(model_format: str):
    """Build an unloaded RerankModel with the given spec format (issue #5156)."""
    from typing import Any, Dict

    from ...model.rerank.core import RerankModel, RerankModelFamilyV2

    spec: Dict[str, Any] = {
        "model_format": model_format,
        "model_hub": "huggingface",
        "quantization": "none",
    }
    if model_format == "ggufv2":
        spec.update(_gguf_spec_fields())
    family = RerankModelFamilyV2(
        version=2,
        model_name="fake-reranker",
        model_specs=[spec],
        language=["en"],
        # non-"unknown" type avoids _auto_detect_type reading a real model path
        type="normal",
        max_tokens=512,
        virtualenv=None,
    )

    class _FakeRerankModel(RerankModel):
        @classmethod
        def check_lib(cls):
            return True

        @classmethod
        def match_json(cls, model_family, model_spec, quantization):
            return True

        def load(self):
            pass

        def rerank(self, *args, **kwargs):
            raise NotImplementedError

    model = _FakeRerankModel("fake-reranker-0", "/tmp/fake", family, "none")
    # emulate a loaded model holding GPU tensors
    model._model = object()  # type: ignore[assignment]
    return model


def _make_embedding_model(model_format: str):
    """Build an unloaded EmbeddingModel with the given spec format (issue #5156)."""
    from typing import Any, Dict

    from ...model.embedding.core import EmbeddingModel, EmbeddingModelFamilyV2

    spec: Dict[str, Any] = {
        "model_format": model_format,
        "model_hub": "huggingface",
        "model_id": None,
        "model_uri": None,
        "model_revision": None,
        "quantization": "none",
    }
    if model_format == "ggufv2":
        spec.update(_gguf_spec_fields())
    family = EmbeddingModelFamilyV2(
        version=2,
        model_name="fake-embedding",
        dimensions=8,
        max_tokens=512,
        language=["en"],
        model_specs=[spec],
        cache_config=None,
        virtualenv=None,
    )

    class _FakeEmbeddingModel(EmbeddingModel):
        @classmethod
        def check_lib(cls):
            return True

        @classmethod
        def match_json(cls, model_family, model_spec, quantization):
            return True

        def load(self):
            pass

        def _create_embedding(self, *args, **kwargs):
            raise NotImplementedError

    model = _FakeEmbeddingModel("fake-embedding-0", "/tmp/fake", family)
    # emulate a loaded model holding GPU tensors
    model._model = object()  # type: ignore[assignment]
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_model, model_format, expect_freed",
    [
        (_make_rerank_model, "pytorch", True),
        (_make_rerank_model, "ggufv2", False),
        (_make_embedding_model, "pytorch", True),
        (_make_embedding_model, "ggufv2", False),
    ],
)
async def test_pre_destroy_frees_gpu_memory(
    monkeypatch, make_model, model_format, expect_freed
):
    # Regression test for issue #5156: stopping a pytorch (sentence_transformers)
    # rerank/embedding model must run del + empty_cache so its VRAM is
    # released; the torch-free llama.cpp (ggufv2) path must be skipped.
    calls = []
    monkeypatch.setattr(
        "xinference.core.model.empty_cache",
        lambda: calls.append(1),
    )

    # __pre_destroy__ only touches self._model and (vLLM-only) self._transfer_ref,
    # so exercise it on a lightweight stand-in to avoid the xoscar actor context.
    stub = types.SimpleNamespace(
        _model=make_model(model_format),
        _transfer_ref=None,
        address="test:0",
    )

    await ModelActor.__pre_destroy__(stub)

    if expect_freed:
        assert len(calls) == 1
        assert not hasattr(stub, "_model")
    else:
        assert len(calls) == 0
        assert stub._model is not None


# ---------------------------------------------------------------------------
# Regression test for the non-stream chat metrics guard (candidate #2).
# A backend whose non-stream chat completion lacks a "usage" field (or sets
# it to None) must not crash ModelActor.chat inside its `finally` metrics
# block. The streaming path already guards with `if final_usage is not None`;
# this keeps the non-stream path consistent with that intent.
# ---------------------------------------------------------------------------
class _NoUsageChatModel:
    def __init__(self):
        self.model_family = MockModelFamily()

    async def chat(self, messages, **kwargs):
        # Intentionally omit "usage" to mimic backends that don't report tokens.
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-no-usage",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        }


class _NoUsageChatModelActor(MockModelActor):
    def __init__(self, supervisor_address, worker_address, replica_model_uid):
        super().__init__(supervisor_address, worker_address, replica_model_uid)
        # Replace the placeholder MockModel with one that returns no usage.
        self._model = _NoUsageChatModel()


@pytest_asyncio.fixture
async def _chat_pool():
    pool = await create_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}", n_process=0
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_chat_without_usage_does_not_crash(_chat_pool):
    # Before the fix, `await actor.chat(...)` raised KeyError/TypeError from
    # the `finally` block because it indexed `record["usage"]` unconditionally.
    pool = _chat_pool
    addr = pool.external_address
    actor = await xo.create_actor(
        _NoUsageChatModelActor,
        address=addr,
        uid=_NoUsageChatModelActor.default_uid(),
        supervisor_address="test:123",
        worker_address="test:345",
        replica_model_uid="test_chat_no_usage",
    )
    result = await actor.chat([{"role": "user", "content": "hi"}])
    # Tolerate both a raw bytes return and an xoscar generator wrapper.
    if isinstance(result, (bytes, str)):
        parsed = json.loads(result)
    else:
        collected = []
        async for chunk in result:
            collected.append(chunk)
        first = collected[0]
        parsed = json.loads(
            first if isinstance(first, (bytes, str)) else first.decode()
        )
    assert parsed["choices"][0]["message"]["content"] == "hello"


class _NonDictUsageChatModel:
    """A model whose non-stream chat returns usage as a non-dict truthy value
    (e.g. a string). The original `if usage:` guard would have crashed on it
    via AttributeError when calling .get()."""

    def __init__(self):
        self.model_family = MockModelFamily()

    async def chat(self, messages, **kwargs):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-non-dict-usage",
            "usage": "unexpected-string-usage",  # truthy but NOT a dict
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        }


class _NonDictUsageChatModelActor(MockModelActor):
    def __init__(self, supervisor_address, worker_address, replica_model_uid):
        super().__init__(supervisor_address, worker_address, replica_model_uid)
        self._model = _NonDictUsageChatModel()


@pytest.mark.asyncio
async def test_chat_with_non_dict_usage_does_not_crash(_chat_pool):
    # Regression test for the isinstance(usage, dict) guard: a non-dict truthy
    # usage (string/list) must not raise AttributeError from .get().
    pool = _chat_pool
    addr = pool.external_address
    actor = await xo.create_actor(
        _NonDictUsageChatModelActor,
        address=addr,
        uid=_NonDictUsageChatModelActor.default_uid(),
        supervisor_address="test:123",
        worker_address="test:345",
        replica_model_uid="test_chat_non_dict_usage",
    )
    result = await actor.chat([{"role": "user", "content": "hi"}])
    if isinstance(result, (bytes, str)):
        parsed = json.loads(result)
    else:
        collected = []
        async for chunk in result:
            collected.append(chunk)
        first = collected[0]
        parsed = json.loads(
            first if isinstance(first, (bytes, str)) else first.decode()
        )
    assert parsed["choices"][0]["message"]["content"] == "hi"
