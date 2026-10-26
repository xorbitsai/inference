# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import Request

from ...core.model import ModelActor
from .. import restful_api
from .test_anthropic_messages import make_app, make_rest_api

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(10)]


@pytest_asyncio.fixture
async def calls(monkeypatch):
    # Newer sse-starlette starts a per-loop shutdown watcher that outlives any
    # one response. Initialize that lifespan before tracking request tasks.
    async def startup():
        yield {"data": "startup"}
        await asyncio.sleep(0)

    async def receive():
        await asyncio.Event().wait()

    await restful_api.EventSourceResponse(startup(), ping=3600)(
        {"type": "http", "method": "GET"}, receive, AsyncMock()
    )
    baseline, entries = asyncio.all_tasks(), {}

    async def generate(messages, config, **kwargs):
        call = entries[messages[0]["content"]]
        call.engine_id = kwargs["request_id"]
        if call.startup_error is not None:
            raise call.startup_error
        if call.wait_creation:
            call.started.set()
            try:
                await call.release.wait()
            finally:
                call.closed.set()

        async def result():
            call.started.set()
            try:
                if config["stream"]:
                    yield {
                        "choices": [
                            {"delta": {"content": "hello"}, "finish_reason": None}
                        ]
                    }
                    call.next_item.set()
                await call.release.wait()
                if call.error:
                    raise RuntimeError(call.error)
                yield {
                    "choices": [
                        {
                            "message": {"content": "hello"},
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            finally:
                await asyncio.sleep(0)
                call.closed.set()

        if config["stream"]:
            return result()
        source = result()
        try:
            return await anext(source)
        finally:
            await source.aclose()

    backend = SimpleNamespace(
        async_chat=generate,
        allow_batch=True,
        model_family=SimpleNamespace(to_description=lambda: {}),
        prepare_anthropic_request=lambda messages, config: {
            "messages": messages,
            "generate_config": config,
            "input_tokens": 7,
            "cache_control_applied": True,
        },
    )
    actor = ModelActor("supervisor", "worker", backend, "test-model-0")
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    actor.abort_request = AsyncMock(side_effect=actor.abort_request)
    actor.is_vllm_backend = AsyncMock(return_value=True)
    chat = actor.chat

    async def local_chat(*args, **kwargs):
        from xoscar.api import IteratorWrapper

        result = await chat(*args, **kwargs)
        if isinstance(result, IteratorWrapper):
            # Preserve actor admission/serialization while consuming the
            # registered generator locally, without opening an actor pool.
            result._gc_destroy = False
            call = entries[args[0][0]["content"]]
            if call.rpc_iterator:
                result._actor_ref = actor
                result.destroy = AsyncMock(side_effect=result.destroy)
                call.iterator = result
            else:
                call.iterator = actor._generators.pop(result._uid)
            call.acquired.set()
            return call.iterator
        return result

    actor.chat = AsyncMock(side_effect=local_chat)
    decrease = actor.decrease_serve_count

    async def decrease_serve_count():
        await asyncio.sleep(0)
        await decrease()

    actor.decrease_serve_count = AsyncMock(side_effect=decrease_serve_count)
    api = make_rest_api()
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    api._report_error_event = AsyncMock()
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=actor))
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    monkeypatch.setattr(
        asyncio, "to_thread", AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    )

    def make(
        stream=False,
        key="operation",
        block_send=False,
        wait_creation=False,
        block_start=False,
        rpc_iterator=False,
    ):
        c = SimpleNamespace(
            model=actor,
            api=api,
            error=None,
            startup_error=None,
            engine_id=None,
            sent=[],
            wait_creation=wait_creation,
            iterator=None,
            rpc_iterator=rpc_iterator,
        )
        c.started, c.next_item, c.closed, c.release, c.disconnected, c.body_sent = (
            asyncio.Event() for _ in range(6)
        )
        c.acquired, c.response_started = asyncio.Event(), asyncio.Event()
        entries[key] = c
        body = json.dumps(
            {
                "model": "test-model",
                "max_tokens": 32,
                "stream": stream,
                "messages": [{"role": "user", "content": key}],
                "cache_control": {"type": "ephemeral"},
            }
        ).encode()
        received = False

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await c.disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            c.sent.append(message)
            if message["type"] == "http.response.start":
                c.response_started.set()
                if block_start:
                    await asyncio.Event().wait()
            if message["type"] == "http.response.body" and message.get("body"):
                c.body_sent.set()
                if block_send:
                    await asyncio.Event().wait()

        c.scope = dict(
            type="http",
            method="POST",
            path="/v1/messages",
            query_string=b"",
            headers=[
                (b"anthropic-version", b"2023-06-01"),
                (b"request-id", b"shared-correlation"),
            ],
        )
        c.request = Request(c.scope, receive)
        app = make_app(api)
        c.start = lambda: asyncio.create_task(app(c.scope, receive, send))
        return c

    yield make
    timers = {t for t in actor._running_tasks.values() if t.get_name() == "abort_block"}
    pending = asyncio.all_tasks() - baseline - {asyncio.current_task()}
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    assert not pending - timers, "request left owned tasks running"
    assert actor._serve_count == 0
    assert not actor._generators


@pytest.mark.parametrize("phase", ["nonstream", "creation", "send", "next_item"])
async def test_disconnect_aborts_same_engine_request_and_releases_actor(calls, phase):
    c = calls(
        stream=phase != "nonstream",
        block_send=phase == "send",
        wait_creation=phase == "creation",
    )
    task = c.start()
    await c.started.wait()
    if phase == "send":
        await c.body_sent.wait()
    elif phase == "next_item":
        await c.next_item.wait()
    c.disconnected.set()
    await asyncio.wait_for(task, 5)
    await asyncio.wait_for(c.closed.wait(), 1)
    c.model.abort_request.assert_awaited_once_with(c.engine_id)
    assert c.engine_id != "shared-correlation"
    if phase in {"nonstream", "creation"}:
        assert c.sent[0]["status"] == 499
        c.model.decrease_serve_count.assert_not_awaited()
    else:
        c.model.decrease_serve_count.assert_awaited_once()


@pytest.mark.parametrize("stream", [False, True])
async def test_connected_response_keeps_protocol_and_headers(calls, stream):
    c = calls(stream=stream)
    c.release.set()
    await asyncio.wait_for(c.start(), 5)
    assert c.sent[0]["status"] == 200
    assert (b"request-id", b"shared-correlation") in c.sent[0]["headers"]
    assert (b"x-xinference-cache-control", b"breakpoint") in c.sent[0]["headers"]
    body = b"".join(m.get("body", b"") for m in c.sent)
    if stream:
        assert b"event: message_stop" in body
        c.model.decrease_serve_count.assert_awaited_once()
    else:
        message = json.loads(body)
        assert message["content"] == [{"type": "text", "text": "hello"}]
        assert message["usage"] == {"input_tokens": 7, "output_tokens": 3}
        c.model.decrease_serve_count.assert_not_awaited()
    c.model.abort_request.assert_not_awaited()


@pytest.mark.parametrize("rpc_iterator", [False, True])
async def test_disconnect_releases_stream_returned_during_startup_abort(
    calls, rpc_iterator
):
    c = calls(stream=True, wait_creation=True, rpc_iterator=rpc_iterator)
    abort = c.model.abort_request.side_effect

    async def finish_startup_during_abort(request_id):
        c.release.set()
        await c.acquired.wait()
        await abort(request_id)

    c.model.abort_request.side_effect = finish_startup_during_abort
    task = c.start()
    await c.started.wait()
    c.disconnected.set()
    await asyncio.wait_for(task, 5)
    assert c.sent[0]["status"] == 499
    c.model.abort_request.assert_awaited_once_with(c.engine_id)
    c.model.decrease_serve_count.assert_awaited_once()
    if rpc_iterator:
        c.iterator.destroy.assert_awaited_once()
    else:
        assert c.iterator.ag_frame is None


@pytest.mark.parametrize("rpc_iterator", [False, True])
async def test_disconnect_before_first_body_pull_releases_stream(calls, rpc_iterator):
    c = calls(stream=True, block_start=True, rpc_iterator=rpc_iterator)
    task = c.start()
    await c.response_started.wait()
    assert c.acquired.is_set()
    assert not c.started.is_set()
    c.disconnected.set()
    await asyncio.wait_for(task, 5)
    assert c.sent[0]["status"] == 200
    assert not c.body_sent.is_set()
    c.model.abort_request.assert_awaited_once_with(c.engine_id)
    c.model.decrease_serve_count.assert_awaited_once()
    if rpc_iterator:
        c.iterator.destroy.assert_awaited_once()
    else:
        assert c.iterator.ag_frame is None


@pytest.mark.parametrize("rpc_iterator", [False, True])
async def test_handler_cancelled_after_stream_acquisition(calls, rpc_iterator):
    c = calls(stream=True, rpc_iterator=rpc_iterator)
    task = c.start()
    await c.acquired.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert not c.sent
    c.model.abort_request.assert_awaited_once_with(c.engine_id)
    c.model.decrease_serve_count.assert_awaited_once()
    if rpc_iterator:
        c.iterator.destroy.assert_awaited_once()
    else:
        assert c.iterator.ag_frame is None


@pytest.mark.parametrize("rpc_iterator", [False, True])
async def test_abort_timeout_still_releases_acquired_stream(
    calls, monkeypatch, rpc_iterator
):
    c = calls(stream=True, block_start=True, rpc_iterator=rpc_iterator)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def abort(request_id):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    wait_for = asyncio.wait_for

    async def expire(operation, timeout):
        if timeout != 5:
            return await wait_for(operation, timeout)
        task = asyncio.create_task(operation)
        await entered.wait()
        return await wait_for(task, 0)

    c.model.abort_request.side_effect = abort
    monkeypatch.setattr(asyncio, "wait_for", expire)
    task = c.start()
    await c.response_started.wait()
    c.disconnected.set()
    await wait_for(task, 5)
    assert cancelled.is_set()
    c.model.abort_request.assert_awaited_once_with(c.engine_id)
    c.model.decrease_serve_count.assert_awaited_once()
    if rpc_iterator:
        c.iterator.destroy.assert_awaited_once()
    else:
        assert c.iterator.ag_frame is None


@pytest.mark.parametrize(
    "error,status", [("backend failed", 500), ("Rate limit reached", 429)]
)
async def test_nonstream_errors_keep_anthropic_envelope(calls, error, status):
    c = calls()
    c.error = error
    c.release.set()
    await asyncio.wait_for(c.start(), 5)
    assert c.sent[0]["status"] == status
    body = json.loads(c.sent[1]["body"])
    assert body["type"] == "error" and body["error"]["message"] == error
    c.model.abort_request.assert_not_awaited()


@pytest.mark.parametrize(
    "failure,status,error_type",
    [
        ("limit", 429, "rate_limit_error"),
        ("reloading", 500, "api_error"),
        ("unsupported", 501, "api_error"),
    ],
)
async def test_stream_startup_errors_keep_http_status(
    calls, failure, status, error_type
):
    c = calls(stream=True)
    if failure in {"limit", "reloading"}:
        chat = c.model.chat

        async def fail_after_preparation(*args, **kwargs):
            if failure == "limit":
                c.model._request_limits = 0
            else:
                c.model._model_state = "reloading"
            return await chat(*args, **kwargs)

        c.model.chat = AsyncMock(side_effect=fail_after_preparation)
    else:
        c.startup_error = NotImplementedError("unsupported structured output")
    await asyncio.wait_for(c.start(), 5)
    assert c.sent[0]["status"] == status
    assert (b"content-type", b"application/json") in c.sent[0]["headers"]
    assert (b"request-id", b"shared-correlation") in c.sent[0]["headers"]
    body = json.loads(b"".join(m.get("body", b"") for m in c.sent))
    assert body["type"] == "error" and body["error"]["type"] == error_type
    c.model.abort_request.assert_not_awaited()
    c.model.decrease_serve_count.assert_not_awaited()


async def test_shared_correlation_id_does_not_cancel_other_request(calls):
    first, second = calls(key="first"), calls(key="second")
    tasks = first.start(), second.start()
    await first.started.wait()
    await second.started.wait()
    assert first.engine_id != second.engine_id
    first.disconnected.set()
    await asyncio.wait_for(tasks[0], 5)
    assert not second.closed.is_set()
    second.release.set()
    await asyncio.wait_for(tasks[1], 5)
    first.model.abort_request.assert_awaited_once_with(first.engine_id)
