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
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_mlx_batch_abort_cancels_generation_and_waits_for_removal(
    mlx_priority_model, monkeypatch, method, cancel_during_cleanup, cleanup_error
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ..mlx.core import MLXBatchModel

    model, actor = mlx_priority_model
    model._model_config["batch_size"] = 1
    monkeypatch.setattr(
        model._tokenizer, "decode", lambda tokens, **kw: "", raising=False
    )
    cleanup_started, cleanup_release, cleanup_finished = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    state = {
        "generator": SimpleNamespace(remove=Mock()),
        "queues": {},
        "pending": {},
        "active": set(),
        "cache_boundaries": {},
        "cache_salts": {},
        "xavier_prompts": {},
        "xavier_writes": {},
    }
    inserted = []

    async def flush(writes):
        if len(inserted) == 1:
            cleanup_started.set()
            try:
                await cleanup_release.wait()
                if cleanup_error:
                    raise RuntimeError("cache publication failed after removal")
            finally:
                cleanup_finished.set()

    batch = object.__new__(MLXBatchModel)
    batch._xavier = SimpleNamespace(flush=flush)
    batch._get_or_create_generator = lambda *args: state
    batch._ensure_background_worker = lambda *args: None

    def insert(*args, **kwargs):
        uid = len(inserted) + 1
        inserted.append(uid)
        if uid > 1 or cancel_during_cleanup:
            state["pending"][uid] = [SimpleNamespace(token=None, finish_reason="stop")]
        return uid, 0, None

    batch._insert_request = insert
    monkeypatch.setattr(MLXBatchModel, "_tokenizer_ref", model._tokenizer)
    monkeypatch.setattr(MLXBatchModel, "_lock", None)
    model._batch_model = batch
    handler = getattr(actor, method)
    prompt = [{"role": "user", "content": "hello"}] if method == "chat" else "hello"

    async def until(predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), 2)

    busy = asyncio.create_task(
        handler(prompt, {"request_id": "busy", "max_tokens": 1000})
    )
    waiting = None
    try:
        await until(lambda: bool(inserted))
        if cancel_during_cleanup:
            await asyncio.wait_for(cleanup_started.wait(), 2)
        waiting = asyncio.create_task(
            handler(prompt, {"request_id": "next", "priority": -5})
        )
        await until(lambda: bool(actor._request_priority_scheduler._waiting))
        assert await actor.abort_request("busy", block_duration=0) == "DONE"
        # Active-generation cases receive no EOS or max_tokens result.
        await asyncio.wait_for(cleanup_started.wait(), 2)
        state["generator"].remove.assert_called_once_with([1])
        assert not state["active"] and not state["queues"]
        busy.cancel()  # Repeated cancellation must not interrupt acknowledgement.
        for _ in range(5):
            await asyncio.sleep(0)
        assert not cleanup_finished.is_set()
        assert not busy.done() and not waiting.done()
        assert inserted == [1]
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(busy), 2)
        await asyncio.wait_for(waiting, 2)
        assert inserted == [1, 2]
        assert actor._request_priority_scheduler.idle and actor._serve_count == 0
    finally:
        cleanup_release.set()
        # Let the pre-fix shielded task exit after a failed assertion.
        for queue in list(state["queues"].values()):
            queue.put_nowait(SimpleNamespace(token=None, finish_reason="stop"))
        await asyncio.wait_for(
            asyncio.gather(
                busy, *([waiting] if waiting else []), return_exceptions=True
            ),
            3,
        )


@pytest.mark.asyncio
async def test_mlx_batch_cancel_before_registration_creates_no_native_uid(
    mlx_priority_model, monkeypatch
):
    from unittest.mock import Mock

    from ..mlx.core import MLXBatchModel

    waiting = asyncio.Event()

    class HeldLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                waiting.set()
            return await super().acquire()

    lock = HeldLock()
    await lock.acquire()
    model, _ = mlx_priority_model
    batch = object.__new__(MLXBatchModel)
    batch._get_or_create_generator = lambda *args: {"generator": object()}
    batch._ensure_background_worker = lambda *args: None
    batch._insert_request = Mock(return_value=(1, 0, None))
    monkeypatch.setattr(MLXBatchModel, "_tokenizer_ref", model._tokenizer)
    monkeypatch.setattr(MLXBatchModel, "_lock", lock)
    stream = batch.generate_stream("hello", max_tokens=1000)
    task = asyncio.create_task(stream.__anext__())
    try:
        await asyncio.wait_for(waiting.wait(), 2)
        batch._insert_request.assert_not_called()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        batch._insert_request.assert_not_called()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        lock.release()
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
async def test_mlx_serial_abort_stops_decode_and_joins_cleanup(
    mlx_priority_model, monkeypatch, method, cancel_during_cleanup
):
    import threading

    from ..utils import generate_completion_chunk

    model, actor = mlx_priority_model
    started, step_release, closing, close_release, closed = (
        threading.Event() for _ in range(5)
    )
    close_queued = asyncio.Event()
    advanced_after_abort = []
    entered = []

    def generate(prompt, config):
        entered.append(config["request_id"])
        if config["request_id"] != "busy":
            yield generate_completion_chunk(
                "ok", "stop", "next", model.model_uid, 3, 1, 4
            ), {}
            return
        try:
            started.set()
            assert step_release.wait(5)
            yield generate_completion_chunk(
                "x", None, "busy", model.model_uid, 3, 1, 4
            ), {}
            if not cancel_during_cleanup:
                advanced_after_abort.append(True)
        finally:
            closing.set()
            assert close_release.wait(5)
            closed.set()

    async def until(predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait(), 2)

    model._generate_stream = generate
    handler = getattr(actor, method)
    prompt = [{"role": "user", "content": "hello"}] if method == "chat" else "hello"
    config = {"response_format": {"type": "json_object"}, "max_tokens": 1000}
    busy = asyncio.create_task(handler(prompt, {**config, "request_id": "busy"}))
    waiting = None
    try:
        await until(started.is_set)
        submit = model._fallback_executor.submit

        def record_submit(fn, *args, **kwargs):
            if getattr(fn, "__name__", None) == "close":
                close_queued.set()
            return submit(fn, *args, **kwargs)

        monkeypatch.setattr(model._fallback_executor, "submit", record_submit)
        if cancel_during_cleanup:
            step_release.set()
            await until(closing.is_set)
        waiting = asyncio.create_task(handler(prompt, {**config, "request_id": "next"}))
        await until(lambda: bool(actor._request_priority_scheduler._waiting))
        assert await actor.abort_request("busy", block_duration=0) == "DONE"
        await asyncio.wait_for(close_queued.wait(), 2)
        step_release.set()
        await until(closing.is_set)
        busy.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not closed.is_set() and not busy.done() and not waiting.done()
        assert entered == ["busy"] and not advanced_after_abort
        assert model._fallback_active and model._fallback_lock.locked()
        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(busy), 2)
        await asyncio.wait_for(waiting, 2)
        assert closed.is_set() and entered == ["busy", "next"]
        assert not model._fallback_active and not model._fallback_lock.locked()
        assert actor._request_priority_scheduler.idle and actor._serve_count == 0
    finally:
        step_release.set()
        close_release.set()
        await asyncio.wait_for(
            asyncio.gather(
                busy, *([waiting] if waiting else []), return_exceptions=True
            ),
            3,
        )


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
