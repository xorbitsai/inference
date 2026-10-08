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
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from ..core import VLLMEmbeddingModel


@pytest.fixture
def model(monkeypatch):
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.19.0"
    vllm.PoolingParams = SimpleNamespace
    vllm.LLM = Mock()
    outputs = ModuleType("vllm.outputs")
    outputs.EmbeddingRequestOutput = SimpleNamespace(from_base=lambda output: output)
    engine_module = ModuleType("vllm.v1.engine.async_llm")
    engine = Mock()
    engine_module.AsyncLLM = SimpleNamespace(from_engine_args=Mock(return_value=engine))
    args_module = ModuleType("vllm.engine.arg_utils")
    args_module.AsyncEngineArgs = SimpleNamespace
    for name, module in {
        "vllm": vllm,
        "vllm.outputs": outputs,
        "vllm.v1.engine.async_llm": engine_module,
        "vllm.engine.arg_utils": args_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "xinference.model.embedding.vllm.core.is_vacc_available", lambda: False
    )
    family = SimpleNamespace(model_name="bge-m3", model_specs=[None])
    model = VLLMEmbeddingModel(
        "replica", "/model", family, batch_size=8, batch_interval=0.1
    )
    model.load()
    model._clean_cache_if_needed = Mock(
        side_effect=AssertionError("native worker owns cache")
    )
    return model


def output(prompt):
    return SimpleNamespace(
        outputs=SimpleNamespace(embedding=[float(len(prompt))]),
        prompt_token_ids=list(range(len(prompt))),
    )


@pytest.mark.asyncio
async def test_native_requests_overlap_and_keep_caller_order_and_usage(model):
    started = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def encode(prompt, params, request_id):
        started.append((prompt, params, request_id))
        if len(started) == 3:
            all_started.set()
        await release.wait()
        if prompt == "long":
            await asyncio.sleep(0)
        yield output(prompt)

    model._model.encode = encode
    first = asyncio.create_task(
        model.create_embedding(
            ["long", "a"], model_uid="public", dimensions=1, normalize_embedding=False
        )
    )
    second = asyncio.create_task(model.create_embedding("bb", model_uid="public"))
    try:
        await asyncio.wait_for(all_started.wait(), 1)
        assert model._process_batch_task is None
        release.set()
        one, two = await asyncio.gather(first, second)
    finally:
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
    assert [x["embedding"] for x in one["data"]] == [[4.0], [1.0]]
    assert [x["index"] for x in one["data"]] == [0, 1]
    assert two["data"][0]["index"] == 0
    assert one["usage"]["prompt_tokens"] == 5
    assert two["usage"]["prompt_tokens"] == 2
    assert one["model"] == "public" and one["model_replica"] == "replica"
    assert len({x[2] for x in started}) == 3
    params = {prompt: params for prompt, params, _ in started}
    assert params["long"].dimensions == 1
    assert params["long"].use_activation is False
    assert params["bb"].use_activation is True
    assert all(p.task == "embed" for p in params.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_native_cancellation_and_failure_stop_sibling_inputs(model, fail):
    active = set()
    ready = asyncio.Event()
    cancelled = set()

    async def encode(prompt, params, request_id):
        active.add(prompt)
        if len(active) == 2:
            ready.set()
        try:
            await ready.wait()
            if fail and prompt == "bad":
                raise ValueError("invalid input")
            await asyncio.Event().wait()
            yield output(prompt)
        except asyncio.CancelledError:
            cancelled.add(prompt)
            raise
        finally:
            active.remove(prompt)

    model._model.encode = encode
    task = asyncio.create_task(model.create_embedding(["bad", "pending"]))
    await asyncio.wait_for(ready.wait(), 1)
    if not fail:
        task.cancel()
    with pytest.raises(ValueError if fail else asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert not active
    assert "pending" in cancelled


@pytest.mark.asyncio
async def test_native_truncation_and_stop(model):
    model._truncate_sentences = Mock(return_value=["cut"])

    async def encode(prompt, params, request_id):
        assert prompt == "cut"
        yield output(prompt)

    model._model.encode = encode
    result = await model.create_embedding(["long input"], truncate_prompt_tokens=3)
    model._truncate_sentences.assert_called_once_with(["long input"], 3)
    assert result["usage"]["total_tokens"] == 3
    model.stop()
    model._model.shutdown.assert_called_once()


def test_native_load_removes_xinference_batch_options(model):
    args = sys.modules[
        "vllm.v1.engine.async_llm"
    ].AsyncLLM.from_engine_args.call_args.args[0]
    assert args.model == "/model"
    assert args.runner == "pooling"
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "batch_interval")
    model.wait_for_load()
    sys.modules["vllm"].LLM.assert_not_called()


def test_older_vllm_keeps_existing_batch_path(model):
    sys.modules["vllm"].__version__ = "0.18.0"
    legacy = VLLMEmbeddingModel("legacy", "/model", model.model_family)
    legacy.load()
    assert not legacy._native_pooling
    assert legacy.create_embedding.__func__ != legacy._async_create_embedding.__func__
    sys.modules["vllm"].LLM.assert_called_once_with(model="/model", runner="pooling")
