# CI: linux
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
import subprocess
import sys
import textwrap
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import Request

from ...core.model import ModelActor
from ...core.pd_model import PDModelActor
from ...model.llm.vllm.tests.test_abort import _new_model
from .. import restful_api
from ..streaming_outcome import get_stream_outcome_reporter
from .test_llm_request_abort import _new_api, _Request
from .test_model_request_logging import _app, _enable_capture

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(10)]
RESULT = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
DATA = json.dumps(RESULT, separators=(",", ":")).encode()


@pytest_asyncio.fixture
async def calls(monkeypatch):
    baseline, entries = asyncio.all_tasks(), {}
    events = _enable_capture(monkeypatch)

    async def generate(_input, config, **kwargs):
        call = entries[kwargs["request_id"]]
        call.started.set()
        try:
            await call.release.wait()
            if call.error:
                raise RuntimeError(call.error)
            return RESULT
        finally:
            call.closed = True

    backend = SimpleNamespace(
        async_generate=generate,
        async_chat=generate,
        allow_batch=True,
        model_family=SimpleNamespace(to_description=lambda: {}),
    )
    actor = ModelActor("supervisor", "worker", backend, "test-model-0")
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    actor.decrease_serve_count = AsyncMock()
    actor.abort_request = AsyncMock(side_effect=actor.abort_request)
    actor.is_vllm_backend = AsyncMock(return_value=True)
    actor.get_limit_mm_per_prompt = AsyncMock(return_value=None)
    api = _new_api()
    api._advanced_auth_service, api._running_tasks = None, {}
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    api.handle_request_limit_error = restful_api.RESTfulAPI.handle_request_limit_error
    api._get_supervisor_ref = AsyncMock()
    supervisor = api._get_supervisor_ref.return_value
    supervisor.describe_model.return_value = {"model_family": "test"}
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=actor))
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    monkeypatch.setattr(
        asyncio, "to_thread", AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    )

    def make(kind="completion", audit=False, key="operation"):
        c = SimpleNamespace(model=actor, api=api, events=events)
        c.started, c.release, c.disconnected = (asyncio.Event() for _ in range(3))
        c.error, c.messages, c.closed = None, [], False
        entries[key] = c
        path = "/v1/completions" if kind == "completion" else "/v1/chat/completions"
        body = dict(model="test-model", request_id=key, stream=False, prompt="hello")
        if kind != "completion":
            body["messages"] = [{"role": "user", "content": body.pop("prompt")}]
        raw = json.dumps(body).encode()
        parts = iter((raw[:12], raw[12:]))

        async def receive():
            part = next(parts, None)
            if part is not None:
                return dict(type="http.request", body=part, more_body=len(part) == 12)
            await c.disconnected.wait()
            return {"type": "http.disconnect"}

        c.scope = dict(type="http", method="POST", path=path, query_string=b"")
        c.scope["headers"] = [(b"x-request-id", b"http-id")]
        c.request = Request(c.scope, receive)
        c.receive, c.send = receive, AsyncMock(side_effect=c.messages.append)
        c.app = _app(path, getattr(api, f"create_{kind}"))
        if audit:
            c.app.middleware("http")(api._audit_middleware)
        c.start = lambda: asyncio.create_task(c.app(c.scope, receive, c.send))
        return c

    yield make
    timers = {t for t in actor._running_tasks.values() if t.get_name() == "abort_block"}
    pending = asyncio.all_tasks() - baseline - {asyncio.current_task()}
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    assert not pending - timers, "request left owned tasks running"
    assert all(c.closed for c in entries.values() if c.started.is_set())
    assert actor._serve_count == 0
    actor.decrease_serve_count.assert_not_awaited()
    assert len(events) == 2 * sum("status_code" in e for e in events)


@pytest.mark.parametrize("kind", ["completion", "chat_completion"])
@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("status", [499, 200, 500, 429])
async def test_routes(calls, kind, audit, status):
    c = calls(kind, audit)
    task = c.start()
    await c.started.wait()
    if status == 499:
        c.disconnected.set()
    else:
        c.error = {500: "backend failed", 429: "Rate limit reached for the model"}.get(
            status
        )
        c.release.set()
    await asyncio.wait_for(task, 5)
    assert [e["status_code"] for e in c.events if "status_code" in e] == [status]
    if status == 499:
        c.model.abort_request.assert_awaited_once()
        assert c.model.abort_request.await_args.args == ("operation",)
        assert all(not m.get("body") for m in c.messages)
    else:
        c.model.abort_request.assert_not_awaited()
        assert c.messages[0]["status"] == status
    if status == 200:
        assert c.messages[1]["body"] == DATA
        assert (b"content-type", b"application/json") in c.messages[0]["headers"]
        assert (b"x-request-id", b"http-id") in c.messages[0]["headers"]
        records = c.model.record_metrics.await_args_list
        metrics = [(a.args[0], a.args[2]["value"]) for a in records]
        names = ("input_tokens_total_counter", "output_tokens_total_counter")
        assert [[v for k, v in metrics if k == n] for n in names] == [[7], [3]]
        assert [v for k, v in metrics if k == "generate_tokens_total"] == [3]
    assert c.api._report_error_event.await_count == (status in (429, 500))


async def test_pd_router_disconnect(calls, monkeypatch):
    c = calls()
    router = SimpleNamespace(generate=c.model.generate, abort_request=AsyncMock())
    router.is_vllm_backend = PDModelActor.is_vllm_backend.__get__(router)
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=router))
    task = c.start()
    await c.started.wait()
    c.disconnected.set()
    await asyncio.wait_for(task, 5)
    router.abort_request.assert_awaited_once_with("operation")
    assert [e["status_code"] for e in c.events if "status_code" in e] == [499]


def _raises(error, message):
    # Python 3.10 drops a CancelledError's message when it crosses a task boundary.
    if error is asyncio.CancelledError and sys.version_info < (3, 11):
        return pytest.raises(error)
    return pytest.raises(error, match=message)


@pytest.mark.parametrize("error", [None, ValueError, asyncio.CancelledError])
async def test_generation_wins_ties(calls, monkeypatch, error):
    c = calls()
    await c.request.json()
    c.disconnected.set()
    wait = asyncio.wait
    monkeypatch.setattr(
        asyncio,
        "wait",
        lambda tasks, **kw: wait(tasks, return_when=asyncio.ALL_COMPLETED),
    )

    async def generate():
        if error:
            raise error("backend")
        return b"exact bytes"

    with _raises(error, "backend") if error else nullcontext():
        response = await restful_api._run_non_streaming_request(
            c.request, c.model, "operation", "generate", generate()
        )
        assert response.body == b"exact bytes"
    c.model.abort_request.assert_not_awaited()


@pytest.mark.parametrize(
    "mode,error,state",
    [
        ("raise", None, "client_disconnected"),
        ("timeout", None, "client_disconnected"),
        ("handler", asyncio.CancelledError, "cancelled"),
        ("disconnect", asyncio.CancelledError, "client_disconnected"),
        ("receive", ValueError, "pending"),
    ],
)
async def test_interruption_cleanup(calls, monkeypatch, mode, error, state):
    c = calls()
    await c.request.json()
    entered, release = asyncio.Event(), asyncio.Event()

    async def abort(key):
        entered.set()
        if mode == "raise":
            raise RuntimeError("abort failed")
        await release.wait()

    async def receive():
        await c.started.wait()
        raise ValueError("receive")

    wait_for = asyncio.wait_for

    async def expire(operation, timeout):
        if timeout != 5 or mode != "timeout":
            return await wait_for(operation, timeout)
        task = asyncio.create_task(operation)
        await entered.wait()
        return await wait_for(task, 0)

    monkeypatch.setattr(asyncio, "wait_for", expire)
    c.model.abort_request.side_effect = abort
    if mode == "receive":
        c.request._receive = receive
    operation = c.model.generate("hello", {}, request_id="operation")
    task = asyncio.create_task(
        restful_api._run_non_streaming_request(
            c.request, c.model, "operation", "generate", operation
        )
    )
    await c.started.wait()
    if mode == "handler":
        task.cancel(mode)
    else:
        c.disconnected.set()
    await entered.wait()
    if mode in ("handler", "disconnect"):
        task.cancel("again" if mode == "handler" else mode)
    if mode != "timeout":
        release.set()
    with _raises(error, mode) if error else nullcontext():
        response = await wait_for(task, 5)
        assert response.status_code == 499
    c.model.abort_request.assert_awaited_once_with("operation")
    assert get_stream_outcome_reporter(c.request).outcome.state.value == state


@pytest.mark.timeout(660)
async def test_cleanup_cancelled_while_joining_generation():
    # Contain the shutdown busy loop in a subprocess: async timeouts cannot stop it.
    # Importing the REST API can take minutes on slow Windows runners, so only the
    # scenario after "ready" has the short deadline.
    script = textwrap.dedent(
        """
        import asyncio
        from unittest.mock import AsyncMock
        from xinference.api.restful_api import Request, _run_non_streaming_request
        print("ready", flush=True)

        async def main():
            started, finalizing = asyncio.Event(), asyncio.Event()
            async def operation():
                try:
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    finalizing.set()
                    await asyncio.Event().wait()
            async def receive():
                await started.wait()
                return {"type": "http.disconnect"}
            request = Request({"type": "http", "headers": [], "client": ("test", 1)}, receive)
            asyncio.create_task(_run_non_streaming_request(
                request, AsyncMock(), "id", "generate", operation(),
            ))
            await finalizing.wait()
            print("cleanup joining generation", flush=True)

        asyncio.run(main())
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output, ready = [], threading.Event()

    def read():
        for line in proc.stdout:
            output.append(line.rstrip("\n"))
            if output[-1] == "ready":
                ready.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert ready.wait(600), output
        assert proc.wait(timeout=20) == 0, output
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        reader.join(5)
    assert output[-1] == "cleanup joining generation", output


async def test_concurrent_ids_and_explicit_abort(calls):
    first, second = calls(key="first"), calls(key="second")
    entered, release = asyncio.Event(), asyncio.Event()
    abort_actor = first.model.abort_request.side_effect

    async def abort(key):
        entered.set()
        await release.wait()
        return await abort_actor(key)

    first.model.abort_request.side_effect = abort
    tasks = [first.start(), second.start()]
    await asyncio.gather(first.started.wait(), second.started.wait())
    first.disconnected.set()
    engine = _new_model(SimpleNamespace(abort=AsyncMock()))
    engine._active_request_ids.add("first")
    first.model._model.abort_request = engine.abort_request
    await entered.wait()

    async def abort_api(uid, key, duration, **kw):
        return {"msg": await abort_actor(key, duration)}

    first.api._get_supervisor_ref.return_value.abort_request = abort_api
    response = await first.api.abort_request(
        _Request({"block_duration": 0}), "test-model", "first"
    )
    assert response.status_code == 200
    assert json.loads(response.body) == {"msg": "DONE"}
    release.set()
    await asyncio.wait_for(tasks[0], 5)
    assert first.model._serve_count == 1 and not tasks[1].done()
    engine._engine.abort.assert_awaited_once_with("first")
    second.release.set()
    await tasks[1]
    first.model.abort_request.assert_awaited_once_with("first")


async def test_h11_connection_lost(calls):
    from uvicorn import Config
    from uvicorn.protocols.http.h11_impl import H11Protocol
    from uvicorn.server import ServerState

    c = calls()
    transport = MagicMock()
    transport.get_extra_info.return_value = None
    transport.is_closing.return_value = False
    state = ServerState()
    config = Config(c.app, lifespan="off", log_config=None, ws="none")
    server = H11Protocol(config, state, {})
    server.connection_made(transport)
    body = await c.request.body()
    server.data_received(
        b"POST /v1/completions HTTP/1.1\r\nHost: test\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
    )
    await c.started.wait()
    server.connection_lost(None)
    await asyncio.wait_for(asyncio.gather(*state.tasks), 5)
    c.model.abort_request.assert_awaited_once()
