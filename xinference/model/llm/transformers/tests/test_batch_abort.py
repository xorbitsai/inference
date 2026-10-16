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
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .....api import restful_api
from .....api.tests.test_llm_request_abort import (
    _create_streaming_response,
    _new_api,
    _Request,
)
from .....core.model import ModelActor
from ....scheduler.batch import BatchScheduler
from ....scheduler.core import XINFERENCE_NON_STREAMING_ABORT_FLAG
from ....scheduler.core import XINFERENCE_STREAMING_ABORT_FLAG as STREAM_ABORT
from ....scheduler.core import XINFERENCE_STREAMING_DONE_FLAG as STREAM_DONE
from ..core import PytorchChatModel, PytorchModel
from ..utils import get_batch_size_and_seq_len_from_kv_cache, get_kv_cache_layer


class _Tokenizer:
    padding_side = "left"
    eos_token_id = -1

    def __call__(self, prompts, **kwargs):
        return SimpleNamespace(input_ids=[[int(t) for t in p.split()] for p in prompts])

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


@pytest.fixture
def runtime(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = object.__new__(PytorchModel)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        model._model = LlamaForCausalLM(config).eval()
    model._device, model._context_length, model.model_uid = "cpu", 128, "tiny"
    model._pytorch_model_config = {"max_num_seqs": 6}
    model._tokenizer = _Tokenizer()
    model.model_family = SimpleNamespace(stop_token_ids=[], to_description=lambda: {})
    model.model_path = "/unused"
    model._batch_scheduler = BatchScheduler(model)
    model._ensure_scheduler_started = AsyncMock()
    model.chat = PytorchChatModel.chat.__get__(model)
    monkeypatch.setattr(BatchScheduler, "_empty_cache", staticmethod(lambda: None))
    monkeypatch.setattr(
        asyncio, "to_thread", AsyncMock(side_effect=lambda f, *a, **k: f(*a, **k))
    )
    yield model
    torch.set_num_threads(previous_threads)


@pytest.fixture
def actor(runtime):
    actor = ModelActor("supervisor", "worker", runtime, "tiny-0")
    actor._model_state, actor.record_metrics = "ready", AsyncMock()
    return actor


async def _add(scheduler, rid, stream=False, **config):
    sink = asyncio.Queue() if stream else asyncio.get_running_loop().create_future()
    config = dict(max_tokens=40, temperature=0.0, **config)
    config.update(request_id=rid, stream=stream, stream_options={"include_usage": True})
    await scheduler.add_request(
        f"{1 + 3 * (len(rid) % 3)} 2 3", sink, "generate", config
    )
    return scheduler._id_to_req[rid]


async def _finish(scheduler):
    for _ in range(4):
        await scheduler.step()
        if not scheduler._id_to_req:
            break
    assert not scheduler._id_to_req and not scheduler._abort_req_ids
    assert not scheduler._running_queue and not scheduler._waiting_queue


def _result(req):
    if not req.stream:
        return req.future_or_queue.result()
    items = [
        req.future_or_queue.get_nowait() for _ in range(req.future_or_queue.qsize())
    ]
    terminal = STREAM_ABORT if req.aborted else STREAM_DONE
    assert items.pop() == terminal and terminal not in items
    return items[-1] if items else None


@pytest.mark.parametrize(
    "cached,aborts,fresh,complete,stream,expected",
    [
        (True, (0,), False, False, False, [{0}]),
        (True, (1,), False, False, True, [{1}]),
        (True, (2,), False, False, False, [{2}]),
        (True, (0, 1, 2), False, False, True, []),
        (True, (0,), True, False, False, [{0}]),
        (True, (0,), False, True, True, [{0}, {0}]),
        (True, (), True, True, False, [{2}]),
        (True, (), False, False, False, []),
        (True, (), False, False, True, []),
        (False, (0, 1, 2), False, False, True, []),
    ],
    ids="first middle last all fresh completion invalid normal usage queued".split(),
)
async def test_cache_retirement(
    runtime, monkeypatch, cached, aborts, fresh, complete, stream, expected
):
    scheduler = runtime._batch_scheduler
    requests = [await _add(scheduler, rid, stream) for rid in ("a", "bb", "ccc")]
    if complete:
        requests[1].generate_config["max_tokens"] = 18
    cached_rows = {}
    if cached:
        await scheduler.step()
        key, value = get_kv_cache_layer(requests[0].kv_cache, 0)
        cached_rows = {
            r: (key[i].clone(), value[i].clone()) for i, r in enumerate(requests)
        }
    inference = Mock(wraps=runtime.batch_inference)
    monkeypatch.setattr(runtime, "batch_inference", inference)
    reduce = Mock(
        side_effect=lambda cache, indexes: PytorchModel.build_reduced_kv_cache(
            runtime, deepcopy(cache), indexes
        )
    )
    monkeypatch.setattr(runtime, "build_reduced_kv_cache", reduce)
    for idx in aborts:
        assert await scheduler.abort_request(requests[idx].request_id) == "DONE"
        assert await scheduler.abort_request(requests[idx].request_id) == "DONE"
    added = [await _add(scheduler, "fresh", stream)] if fresh else []
    bad = None
    if fresh and complete:
        bad = await _add(scheduler, "invalid", stream_interval=0)
    await scheduler.step()
    if len(aborts) == 3:
        inference.assert_not_called()
    assert [c.args[1] for c in reduce.call_args_list] == expected
    active = added + [r for r in requests if not r.stopped]
    for row, req in enumerate(active):
        assert req.kv_cache is active[0].kv_cache
        size, _ = get_batch_size_and_seq_len_from_kv_cache(req.kv_cache, runtime)
        assert size == len(active)
        if req in cached_rows:
            key, value = get_kv_cache_layer(req.kv_cache, 0)
            old_key, old_value = cached_rows[req]
            prefix_len = old_key.shape[-2]
            torch.testing.assert_close(key[row, :, :prefix_len], old_key)
            torch.testing.assert_close(value[row, :, :prefix_len], old_value)
    await _finish(scheduler)
    for req in requests + added:
        assert req.kv_cache is None and req.error_msg is None
        assert await scheduler.abort_request(req.request_id) == "NOT_FOUND"
        result = _result(req)
        if req.aborted:
            if not stream:
                assert result == XINFERENCE_NON_STREAMING_ABORT_FLAG
            continue
        limit = req.generate_config["max_tokens"]
        assert result["usage"]["completion_tokens"] == limit
        if not stream:
            assert result["choices"][0]["finish_reason"] == "length"
        # Compare generated token IDs with an independent run of the same prompt.
        control = BatchScheduler(runtime)
        reference = await _add(control, req.request_id)
        await _finish(control)
        assert req.new_tokens[:limit] == reference.new_tokens[:limit]
    if bad:
        assert bad.kv_cache is None
        with pytest.raises(ValueError, match="stream_interval"):
            bad.future_or_queue.result()


async def test_failure_recovery(runtime, monkeypatch):
    scheduler = runtime._batch_scheduler
    requests = [await _add(scheduler, rid) for rid in ("a", "b")]
    await scheduler.step()
    await scheduler.abort_request("a")
    with monkeypatch.context() as patch:
        patch.setattr(
            runtime,
            "build_reduced_kv_cache",
            Mock(side_effect=RuntimeError("injected failure")),
        )
        await scheduler.step()
    assert not scheduler._id_to_req and not scheduler._abort_req_ids
    for req in requests:
        assert req.kv_cache is None
        with pytest.raises(RuntimeError, match="injected failure"):
            req.future_or_queue.result()
    recovery = await _add(scheduler, "recovery")
    await _finish(scheduler)
    assert _result(recovery)["usage"]["completion_tokens"] == 40


@pytest.mark.parametrize("placement", ["positional", "keyword", "none"])
async def test_actor_chat_identity(actor, monkeypatch, placement):
    invoke = AsyncMock(return_value={"choices": []})
    monkeypatch.setattr(actor._model, "chat", invoke)
    config = {"request_id": "config"}
    args = (config,) if placement == "positional" else ()
    kwargs = {"generate_config": config} if placement == "keyword" else {}
    await actor.chat("input", *args, request_id=17, **kwargs)
    captured = (
        invoke.call_args.args[1] if args else invoke.call_args.kwargs["generate_config"]
    )
    assert captured == dict(config if placement != "none" else {}, request_id="17")
    assert captured is not config and config == {"request_id": "config"}
    assert "request_id" not in invoke.call_args.kwargs


def test_batching_capability():
    for attributes, expected in [
        ({}, False),
        ({"_batch_scheduler": None}, False),
        ({"_batch_scheduler": object()}, True),
    ]:
        actor = SimpleNamespace(_model=SimpleNamespace(**attributes))
        assert ModelActor.is_transformers_batching_enabled(actor) is expected


@pytest.mark.parametrize("kind", ["completion", "chat_completion"])
@pytest.mark.parametrize("client_id", ["missing", None, "", "explicit"])
async def test_rest_scheduler_identity(actor, runtime, monkeypatch, kind, client_id):
    scheduler, registered = runtime._batch_scheduler, asyncio.Event()
    add = scheduler.add_request

    async def register(*args, **kwargs):
        await add(*args, **kwargs)
        registered.set()

    monkeypatch.setattr(scheduler, "add_request", register)
    method = "generate" if kind == "completion" else "chat"

    async def invoke(*args, **kwargs):
        result = await getattr(actor, method)(*args, **kwargs)
        result._actor_ref, result._gc_destroy = actor, False
        return result

    model = SimpleNamespace(
        uid=b"tiny",
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=True),
        abort_request=AsyncMock(side_effect=runtime.abort_request),
        decrease_serve_count=AsyncMock(),
        **{method: invoke},
    )
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(
        return_value=SimpleNamespace(describe_model=AsyncMock(return_value={}))
    )
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    monkeypatch.setattr(
        restful_api.uuid, "uuid4", lambda: SimpleNamespace(hex="generated")
    )
    body = dict(model="tiny", stream=True)
    if kind == "completion":
        body["prompt"] = "1 2 3"
    else:
        body["messages"] = [{"role": "user", "content": "1 2 3"}]
    if client_id != "missing":
        body["request_id"] = client_id
    create = getattr(api, f"create_{kind}")
    response = await create(_Request(body))
    task = asyncio.create_task(anext(response.body_iterator))
    effective_id = "explicit" if client_id == "explicit" else "generated"
    try:
        await asyncio.wait_for(registered.wait(), 5)
        assert list(scheduler._id_to_req) == [effective_id]
        req = scheduler._id_to_req[effective_id]
        req.future_or_queue.put_nowait({"choices": []})
        await task
        await response.body_iterator.aclose()
        model.abort_request.assert_awaited_once_with(effective_id)
        assert scheduler._abort_req_ids == {effective_id}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_vllm_identity_branch(monkeypatch, kind):
    async def source():
        yield b'{"choices": []}'

    response, model, _ = await _create_streaming_response(monkeypatch, kind, source())
    await anext(response.body_iterator)
    call = (model.generate if kind == "completion" else model.chat).await_args
    assert call.kwargs["request_id"] == f"{kind}-request"
    assert (
        "request_id" not in call.args[1]
        and "request_id" not in call.kwargs["raw_params"]
    )
    model.is_transformers_batching_enabled.assert_not_awaited()
    await response.body_iterator.aclose()
