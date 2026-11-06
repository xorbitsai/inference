# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
from concurrent.futures import Future

import pytest


@pytest.mark.parametrize("capacity", [1, 4, 16])
def test_transformers_capacity_tracks_actual_batching(capacity):
    from ..transformers.core import PytorchModel

    model = object.__new__(PytorchModel)
    model._pytorch_model_config = {"max_num_seqs": capacity}
    model._batch_scheduler = None
    assert model.get_request_priority_capacity() == 1
    model._batch_scheduler = object()
    assert model.get_request_priority_capacity() == capacity


def test_transformers_multimodal_capacity_without_batch_scheduler():
    from ..transformers.multimodal.core import PytorchMultiModalModel

    class VisionModel(PytorchMultiModalModel):
        def decide_device(self):
            pass

        def load_processor(self):
            pass

        def load_multimodal_model(self):
            pass

    model = object.__new__(VisionModel)
    model._pytorch_model_config = {"max_num_seqs": 16}
    assert not hasattr(model, "_batch_scheduler")
    assert model.get_request_priority_capacity() == 1


@pytest.mark.parametrize("vision", [False, True])
def test_mlx_capacity_tracks_batching_and_serial_vision(vision):
    from ..mlx.core import MLXChatModel, MLXVisionModel

    model = object.__new__(MLXVisionModel if vision else MLXChatModel)
    model._model_config = {"batch_size": 8}
    model._batch_model = None
    assert model.get_request_priority_capacity() == 1
    if not vision:
        model._batch_model = object()
        assert model.get_request_priority_capacity() == 8


def test_llama_cpp_capacity_uses_loaded_native_slots():
    from ..llama_cpp.core import XllamaCppModel

    model = object.__new__(XllamaCppModel)
    model._request_priority_capacity = 8
    assert model.get_request_priority_capacity() == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("read", [False, True])
async def test_llama_cpp_stream_close_waits_for_native_worker(read):
    from ..llama_cpp.core import _owned_native_stream

    native = Future()
    stream = _owned_native_stream(iter(["chunk", "next"]), native)
    if read:
        assert next(stream) == "chunk"
    close = asyncio.create_task(asyncio.to_thread(stream.close))
    await asyncio.sleep(0)
    assert not close.done()
    native.set_result(None)
    await asyncio.wait_for(close, 3)


@pytest.fixture
def mlx_priority_model():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from ....core.model import ModelActor
    from ..mlx.core import MLXChatModel

    family = SimpleNamespace(
        model_specs=[SimpleNamespace(model_revision="test", quantization="4bit")],
        model_name="priority-test",
        model_family="priority-test",
        model_ability=["chat"],
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
        stop=None,
        stop_token_ids=None,
        to_description=lambda: {"model_type": "LLM"},
    )
    model = MLXChatModel("priority-test-0", family, "/unused", {"batch_size": 4})
    model._model = object()
    model._context_length = 8192
    model._tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    model._batch_model = SimpleNamespace(generate=AsyncMock())
    actor = ModelActor("test:123", "test:456", model, "priority-test-0", None)
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    yield model, actor
    model.stop()


@pytest.mark.asyncio
async def test_mlx_constrained_requests_wait_in_priority_order(mlx_priority_model):
    from ..utils import generate_completion_chunk

    model, actor = mlx_priority_model
    started = []

    def generate(prompt, config):
        started.append(prompt)
        yield generate_completion_chunk(
            "ok", None, "chunk", model.model_uid, 3, 1, 4
        ), None

    model._generate_stream = generate
    config = {"response_format": {"type": "json_object"}, "max_tokens": 4}
    stream = await actor.chat(
        [{"role": "user", "content": "A"}], {**config, "stream": True}
    )
    stream._actor_ref = actor
    await stream.__anext__()
    low = asyncio.create_task(
        actor.chat([{"role": "user", "content": "B"}], {**config, "priority": 10})
    )
    high = asyncio.create_task(
        actor.chat([{"role": "user", "content": "C"}], {**config, "priority": -10})
    )
    try:
        for _ in range(100):
            if actor._serve_count == 3:
                break
            await asyncio.sleep(0)
        assert actor._serve_count == 3
        assert started == ["A"]
        stream._gc_destroy = False
        await stream.destroy()
        await asyncio.wait_for(asyncio.gather(low, high), 3)
        assert started == ["A", "C", "B"]
        model._batch_model.generate.assert_not_awaited()
    finally:
        stream._gc_destroy = False
        await stream.destroy()
        await asyncio.gather(low, high, return_exceptions=True)
        model.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("outer_workers", [4, 32])
async def test_llama_cpp_callback_capacity_keeps_waiters_in_priority_queue(
    monkeypatch, outer_workers
):
    import contextvars
    import functools
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from ....core.model import ModelActor
    from ..llama_cpp.core import XllamaCppModel

    # A small shared executor must not create an earlier hidden FIFO either.
    outer = ThreadPoolExecutor(max_workers=outer_workers)

    async def shared_thread(fn, *args, **kwargs):
        context = contextvars.copy_context()
        call = functools.partial(context.run, fn, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(outer, call)

    monkeypatch.setattr(asyncio, "to_thread", shared_thread)
    entered = []
    releases = {f"low-{i}": threading.Event() for i in range(16)}
    releases["high"] = threading.Event()
    lock = threading.Lock()

    def handle(data, callback):
        label = data["prompt"]
        with lock:
            entered.append(label)
        assert releases[label].wait(10)
        callback({"choices": []})

    params = SimpleNamespace(
        model="",
        mmproj=SimpleNamespace(path=""),
        cpuparams=SimpleNamespace(),
        cpuparams_batch=SimpleNamespace(),
        n_gpu_layers=0,
    )
    monkeypatch.setitem(
        sys.modules,
        "xllamacpp",
        SimpleNamespace(
            CommonParams=lambda: params,
            Server=lambda params: SimpleNamespace(handle_completions=handle),
            __version__="0.2.0",
            estimate_gpu_layers=Mock(),
            get_device_info=Mock(),
            ggml_backend_dev_type=Mock(),
        ),
    )
    monkeypatch.setattr(
        "xinference.model.llm.gguf_model_metadata.resolve_gguf_model_path",
        lambda *args: "/unused.gguf",
    )
    model = object.__new__(XllamaCppModel)
    model.model_uid = "priority-test"
    model.model_family = SimpleNamespace(
        chat_template=None,
        model_name="priority-test",
        to_description=lambda: {"model_type": "LLM"},
    )
    model.model_spec = SimpleNamespace()
    model.model_path = "/unused"
    model._llamacpp_model_config = {
        "reasoning_content": False,
        "n_parallel": 16,
        "n_threads": 4,
        "n_gpu_layers": 0,
    }
    model.prepare_parse_reasoning_content = Mock()
    model.prepare_parse_tool_calls = Mock()
    model._draft_options = lambda: ()
    model._apply_draft_model = Mock()
    model.load()
    actor = ModelActor("test:123", "test:456", model, "priority-test-0", None)
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    tasks = []

    async def until(predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), 3)

    try:
        for i in range(16):
            tasks.append(
                asyncio.create_task(actor.generate(f"low-{i}", {"priority": 10}))
            )
        await until(lambda: len(entered) == 10 and actor._serve_count == 16)
        tasks.append(asyncio.create_task(actor.generate("high", {"priority": -10})))
        await until(lambda: actor._serve_count == 17)
        releases[entered[0]].set()
        await until(lambda: len(entered) > 10)
        assert entered[10] == "high"
    finally:
        for release in releases.values():
            release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        model._executor.shutdown(wait=True)
        outer.shutdown(wait=True)
        executor = actor.__dict__.get("_request_priority_executor")
        if executor is not None:
            executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("serial_active", [False, True])
async def test_mlx_batch_capacity_and_shared_limit(mlx_priority_model, serial_active):
    from unittest.mock import AsyncMock

    from ..utils import generate_completion_chunk

    model, actor = mlx_priority_model
    entered = []
    release = asyncio.Event()
    stream = None

    def serial(prompt, config):
        yield generate_completion_chunk(
            "ok", None, "serial", model.model_uid, 3, 1, 4
        ), None

    async def batch(**kwargs):
        entered.append(kwargs["prompt"])
        await release.wait()
        return "ok", {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}

    model._generate_stream = serial
    model._batch_model.generate = AsyncMock(side_effect=batch)
    if serial_active:
        stream = await actor.chat(
            [{"role": "user", "content": "serial"}],
            {"stream": True, "response_format": {"type": "json_object"}},
        )
        stream._actor_ref = actor
        await stream.__anext__()
    tasks = [
        asyncio.create_task(actor.chat([{"role": "user", "content": f"batch-{i}"}], {}))
        for i in range(4)
    ]

    async def until(count):
        async def wait():
            while len(entered) < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), 3)

    try:
        await until(3 if serial_active else 4)
        if serial_active:
            assert len(entered) == 3
            stream._gc_destroy = False
            await stream.destroy()
            await until(4)
        assert len(entered) == 4
        assert all(not task.done() for task in tasks)
    finally:
        release.set()
        if stream is not None:
            stream._gc_destroy = False
            await stream.destroy()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("json_encoded", [False, True])
@pytest.mark.parametrize(
    "context,capacity",
    [
        ({}, 4),
        ({"thinking_budget": None}, 4),
        ({"thinking_budget": 0}, 1),
        ({"thinking_budget": -1}, 1),
    ],
)
def test_mlx_capacity_normalizes_chat_options_without_mutating(
    mlx_priority_model, json_encoded, context, capacity
):
    import copy
    import json

    model, _ = mlx_priority_model
    config = {"chat_template_kwargs": json.dumps(context) if json_encoded else context}
    original = copy.deepcopy(config)
    assert model.get_request_priority_capacity(config) == capacity
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("priority", [None, 0])
async def test_mlx_api_accepts_json_string_chat_template_kwargs(
    mlx_priority_model, monkeypatch, endpoint, priority
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from ....api import restful_api

    model, actor = mlx_priority_model
    model._batch_model.generate.return_value = (
        "ok",
        {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )
    proxy = SimpleNamespace(
        uid=b"priority-test-0",
        chat=actor.chat,
        generate=actor.generate,
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
    )
    supervisor = SimpleNamespace(
        get_model=AsyncMock(return_value=proxy),
        describe_model=AsyncMock(
            return_value={"model_family": "priority-test", "model_engine": "MLX"}
        ),
    )
    api = object.__new__(restful_api.RESTfulAPI)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._set_trace_model = api._set_trace_model_type = MagicMock()
    api._check_model_access = MagicMock()
    api._report_error_event = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    body = {
        "model": "priority-test",
        "messages": [{"role": "user", "content": "hello"}],
        "prompt": "hello",
        "max_tokens": 4,
        "chat_template_kwargs": "{}",
    }
    if priority is not None:
        body["priority"] = priority
    request = SimpleNamespace(
        json=AsyncMock(return_value=body), headers={}, state=SimpleNamespace()
    )
    handler = (
        api.create_chat_completion if endpoint == "chat" else api.create_completion
    )
    response = await handler(request)
    assert response.status_code == 200
    choice = json.loads(response.body)["choices"][0]
    assert (
        choice["message"]["content"] if endpoint == "chat" else choice["text"]
    ) == "ok"
    model._batch_model.generate.assert_awaited_once()
    assert model.get_request_priority_capacity({"chat_template_kwargs": "{}"}) == 4
    api._report_error_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["not-json", "null", "[]", None, [], True])
async def test_mlx_chat_still_rejects_invalid_chat_options(mlx_priority_model, context):
    model, actor = mlx_priority_model
    with pytest.raises(TypeError, match="chat_template_kwargs"):
        await actor.chat(
            [{"role": "user", "content": "hello"}], {"chat_template_kwargs": context}
        )
    model._batch_model.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_template_kwargs",
    [
        {"enable_thinking": True, "thinking_budget": -1},
        '{"enable_thinking": true, "thinking_budget": -1}',
        "not-json",
        "null",
        "[]",
        None,
        [],
        True,
    ],
)
async def test_mlx_completion_capacity_probe_preserves_ignored_chat_options(
    mlx_priority_model,
    chat_template_kwargs,
):
    import copy
    from unittest.mock import AsyncMock

    from ..reasoning_parser import ReasoningParser

    model, actor = mlx_priority_model
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    model._batch_model.generate = AsyncMock(
        return_value=(
            "ok",
            {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )
    )
    config = {
        "max_tokens": 4,
        "chat_template_kwargs": chat_template_kwargs,
    }
    original = copy.deepcopy(config)
    assert model.get_request_priority_capacity(config) == 1
    assert config == original
    # Completion generation has never applied chat-only budget controls.
    await actor.generate("hello", config)
    model._batch_model.generate.assert_awaited_once()
    assert config == original
