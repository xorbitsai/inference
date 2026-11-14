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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from .. import restful_api


class _Request:
    client = "test-client"

    def __init__(self, body, headers=None):
        self._body = body
        self.headers = headers or {}
        self.state = SimpleNamespace()

    async def json(self):
        return self._body


def _new_api():
    api = object.__new__(restful_api.RESTfulAPI)
    api._set_trace_model = MagicMock()
    api._set_trace_model_type = MagicMock()
    api._check_model_access = MagicMock()
    api._report_error_event = AsyncMock()
    api.handle_request_limit_error = MagicMock()
    return api


@pytest.mark.asyncio
async def test_chat_routing_key_is_scoped_to_the_credential(monkeypatch):
    model = SimpleNamespace(
        uid=b"affinity-model-rep1",
        chat=AsyncMock(return_value=b'{"choices": []}'),
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
    )
    supervisor = SimpleNamespace(
        get_model=AsyncMock(return_value=model),
        get_chat_model=AsyncMock(return_value=(model, True)),
        describe_model=AsyncMock(return_value={"model_family": "test-family"}),
    )
    model.release_chat_reservation = AsyncMock()
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    body = {
        "model": "affinity-model",
        "messages": [{"role": "user", "content": "hello"}],
    }

    for key, credential in (("task-1", "a"), ("task-1", "a"), ("task-1", "b")):
        response = await api.create_chat_completion(
            _Request(
                dict(body, routing_key=key), {"authorization": f"Bearer {credential}"}
            )
        )
        assert response.body == b'{"choices": []}'
        call = model.chat.await_args
        assert "routing_key" not in call.args[1]
        assert "routing_key" not in call.kwargs["raw_params"]
        reservation_id = supervisor.get_chat_model.await_args.args[2]
        assert call.kwargs["_chat_reservation_id"] == reservation_id
        model.release_chat_reservation.assert_not_awaited()
        supervisor.describe_model.assert_awaited_with(
            "affinity-model", selected_replica_uid="affinity-model-rep1"
        )
    keys = [call.args[1] for call in supervisor.get_chat_model.await_args_list]
    assert keys[0] == keys[1] != keys[2]
    assert "task-1" not in keys

    await api.create_chat_completion(_Request(body))
    supervisor.get_model.assert_awaited_with("affinity-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "kind,routing_key", [("chat", None), ("completion", None), ("chat", "task")]
)
async def test_llm_describes_selected_replica(monkeypatch, kind, routing_key, stream):
    from ...core.exceptions import ModelNotReadyError
    from ...core.supervisor import SupervisorActor
    from ...core.utils import build_replica_model_uid

    chunk = {"data": '{"choices": []}'}

    async def chunks():
        yield chunk

    operation = AsyncMock(return_value=chunks() if stream else b'{"choices": []}')
    model = SimpleNamespace(
        uid=b"metadata-model-rep1",
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        reserve_chat_request=AsyncMock(return_value=True),
        release_chat_reservation=AsyncMock(),
        decrease_serve_count=AsyncMock(),
        **{"chat" if kind == "chat" else "generate": operation},
    )
    recovering = SimpleNamespace(
        get_model=AsyncMock(side_effect=ModelNotReadyError("replica 0 is recovering")),
        describe_model=AsyncMock(
            side_effect=ModelNotReadyError("replica 0 is recovering")
        ),
    )
    healthy = SimpleNamespace(
        get_model=AsyncMock(return_value=model),
        describe_model=AsyncMock(return_value={"model_family": "test-family"}),
    )
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(2)
    supervisor._model_uid_to_replica_info = {"metadata-model": info}
    supervisor._replica_model_uid_to_worker = {
        build_replica_model_uid("metadata-model", 0): recovering,
        build_replica_model_uid("metadata-model", 1): healthy,
    }
    # Keep replica 0 active while round robin selects healthy replica 1.
    assert next(info.scheduler) == 0
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    body = {"model": "metadata-model", "stream": stream}
    if kind == "chat":
        body["messages"] = [{"role": "user", "content": "hello"}]
        if routing_key is not None:
            body["routing_key"] = routing_key
        response = await api.create_chat_completion(_Request(body))
    else:
        body["prompt"] = "hello"
        response = await api.create_completion(_Request(body))
    assert response.status_code == 200
    if stream:
        expected = [chunk, "[DONE]"] if kind == "chat" else [chunk]
        assert [item async for item in response.body_iterator] == expected
        if response.background is not None:
            await response.background()
    else:
        assert response.body == b'{"choices": []}'
    operation.assert_awaited_once()
    healthy.get_model.assert_awaited_once_with(model_uid="metadata-model-rep1")
    healthy.describe_model.assert_awaited_once_with(model_uid="metadata-model-rep1")
    recovering.get_model.assert_not_awaited()
    recovering.describe_model.assert_not_awaited()
    api._report_error_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_ready", [True, False])
async def test_chat_describes_live_peer_when_selected_replica_was_removed(
    monkeypatch, peer_ready
):
    from ...core.supervisor import SupervisorActor
    from ...core.utils import build_replica_model_uid

    model = SimpleNamespace(
        uid=b"affinity-race-rep1",
        chat=AsyncMock(return_value=b'{"choices": []}'),
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
    )
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(2)
    supervisor._model_uid_to_replica_info = {"affinity-race": info}
    worker = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": "test-family"})
    )
    supervisor._replica_model_uid_to_worker = (
        {build_replica_model_uid("affinity-race", 0): worker} if peer_ready else {}
    )

    async def get_model(*args, **kwargs):
        info.active_replica_ids.remove(1)
        return model, True

    supervisor.get_chat_model = AsyncMock(side_effect=get_model)
    model.release_chat_reservation = AsyncMock()
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    request = _Request(
        {
            "model": "affinity-race",
            "messages": [{"role": "user", "content": "hello"}],
            "routing_key": "task",
        }
    )
    if peer_ready:
        response = await api.create_chat_completion(request)
        assert response.body == b'{"choices": []}'
        worker.describe_model.assert_awaited_once_with(model_uid="affinity-race-rep0")
        model.chat.assert_awaited_once()
        model.release_chat_reservation.assert_not_awaited()
    else:
        with pytest.raises(HTTPException) as caught:
            await api.create_chat_completion(request)
        assert caught.value.status_code == 503
        assert caught.value.headers["Retry-After"] == "30"
        model.chat.assert_not_awaited()
        model.release_chat_reservation.assert_awaited_once_with(
            supervisor.get_chat_model.await_args.args[2]
        )
    api._report_error_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [False, 42, "", " ", "a" * 129])
async def test_chat_rejects_invalid_routing_keys_before_dispatch(key):
    api = _new_api()
    api._get_supervisor_ref = AsyncMock()
    with pytest.raises(HTTPException, match="routing_key") as error:
        await api.create_chat_completion(
            _Request(
                {
                    "model": "affinity-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "routing_key": key,
                }
            )
        )
    assert error.value.status_code == 400
    api._get_supervisor_ref.assert_not_awaited()


async def _create_streaming_response(
    monkeypatch, kind, source, extra_body=None, is_vllm=True, limit_getter=None
):
    method = "generate" if kind == "completion" else "chat"
    decrease_completed = asyncio.Event()

    async def decrease_serve_count():
        await asyncio.sleep(0)
        decrease_completed.set()

    model = SimpleNamespace(
        uid=f"{kind}-model".encode(),
        abort_request=AsyncMock(return_value="DONE"),
        release_chat_reservation=AsyncMock(),
        decrease_serve_count=AsyncMock(side_effect=decrease_serve_count),
        is_vllm_backend=AsyncMock(return_value=is_vllm),
        get_limit_mm_per_prompt=limit_getter or AsyncMock(return_value=None),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        **{method: AsyncMock(return_value=source)},
    )

    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    monkeypatch.setattr(
        restful_api, "require_chat_model", AsyncMock(return_value=(model, True))
    )
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": "test-family"})
    )
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    if kind == "completion":
        body = {
            "model": "completion-model",
            "prompt": "hello",
            "stream": True,
            "request_id": "completion-request",
        }
    else:
        body = {
            "model": "chat-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "request_id": "chat-request",
        }
    body.update(extra_body or {})
    create = (
        api.create_completion if kind == "completion" else api.create_chat_completion
    )
    response = await create(_Request(body))
    return response, model, decrease_completed


@pytest.mark.asyncio
async def test_chat_releases_reservation_if_stream_never_starts(monkeypatch):
    response, model, _ = await _create_streaming_response(
        monkeypatch, "chat", None, extra_body={"routing_key": "task"}
    )
    reservation_id = restful_api.require_chat_model.await_args.kwargs["reservation_id"]
    await response.background()
    model.chat.assert_not_awaited()
    model.release_chat_reservation.assert_awaited_once_with(reservation_id)


@pytest.mark.asyncio
async def test_failed_chat_releases_reservation(monkeypatch):
    error = RuntimeError("chat failed")
    model = SimpleNamespace(
        uid=b"model-rep0",
        is_vllm_backend=AsyncMock(return_value=False),
        chat=AsyncMock(side_effect=error),
        release_chat_reservation=AsyncMock(),
    )
    supervisor = SimpleNamespace(
        get_chat_model=AsyncMock(return_value=(model, True)),
        describe_model=AsyncMock(return_value={}),
    )
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._get_model_last_error = AsyncMock(return_value=error)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    request = _Request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "routing_key": "task",
        }
    )
    with pytest.raises(HTTPException) as caught:
        await api.create_chat_completion(request)
    assert caught.value.status_code == 500
    model.release_chat_reservation.assert_awaited_once_with(
        model.chat.await_args.kwargs["_chat_reservation_id"]
    )


@pytest.mark.asyncio
async def test_vllm_non_streaming_disconnect_releases_pending_reservation(monkeypatch):
    dispatched = asyncio.Event()
    cancelled = asyncio.Event()

    async def chat(*args, **kwargs):
        dispatched.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def receive():
        await dispatched.wait()
        return {"type": "http.disconnect"}

    model = SimpleNamespace(
        uid=b"model-rep0",
        is_vllm_backend=AsyncMock(return_value=True),
        get_limit_mm_per_prompt=AsyncMock(return_value=None),
        chat=AsyncMock(side_effect=chat),
        abort_request=AsyncMock(return_value="DONE"),
        release_chat_reservation=AsyncMock(),
    )
    supervisor = SimpleNamespace(
        get_chat_model=AsyncMock(return_value=(model, True)),
        describe_model=AsyncMock(return_value={}),
    )
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    request = _Request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "routing_key": "task",
            "request_id": "chat-request",
        }
    )
    request.receive = receive

    response = await asyncio.wait_for(api.create_chat_completion(request), 5)
    assert response.status_code == 499
    assert cancelled.is_set()
    model.abort_request.assert_awaited_once_with("chat-request")
    model.release_chat_reservation.assert_awaited_once_with(
        model.chat.await_args.kwargs["_chat_reservation_id"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("is_vllm", [False, True])
@pytest.mark.parametrize("ignore_eos", [None, False, True])
async def test_generation_routes_preserve_explicit_ignore_eos(
    monkeypatch, kind, is_vllm, ignore_eos
):
    source, _, _ = _tracked_stream(wait_after_first=False)
    extra_body = {} if ignore_eos is None else {"ignore_eos": ignore_eos}
    response, model, _ = await _create_streaming_response(
        monkeypatch, kind, source, extra_body=extra_body, is_vllm=is_vllm
    )
    async for _ in response.body_iterator:
        pass
    call = (model.generate if kind == "completion" else model.chat).await_args
    config = call.args[1]
    if ignore_eos is None:
        assert "ignore_eos" not in config
    else:
        assert config["ignore_eos"] is ignore_eos


async def _run_asgi_disconnect(response, phase, next_item_started):
    first_body_started = asyncio.Event()

    async def receive():
        await first_body_started.wait()
        if phase == "next_item":
            await next_item_started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_body_started.set()
            if phase == "send":
                await asyncio.Event().wait()

    await asyncio.wait_for(
        response(
            {"type": "http", "method": "POST", "asgi": {"version": "3.0"}},
            receive,
            send,
        ),
        timeout=5,
    )


def _tracked_stream(*, wait_after_first):
    closed = asyncio.Event()
    next_item_started = asyncio.Event()

    async def generate():
        try:
            yield b'{"choices": []}'
            if wait_after_first:
                next_item_started.set()
                await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            closed.set()

    return generate(), closed, next_item_started


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", [True, -1.0, "-1", 0, -2])
async def test_chat_does_not_forward_invalid_priority_hint(monkeypatch, hint):
    stream, _, _ = _tracked_stream(wait_after_first=False)
    response, model, _ = await _create_streaming_response(
        monkeypatch, "chat", stream, extra_body={"priority_hint": hint}
    )
    assert [item async for item in response.body_iterator]
    model.chat.assert_awaited_once()
    assert "priority_hint" not in model.chat.await_args.kwargs["raw_params"]


@pytest.mark.asyncio
async def test_completion_disconnect_aborts_propagated_request_id(monkeypatch):
    async def disconnected_stream():
        yield b'{"choices": []}'
        raise asyncio.CancelledError

    model = SimpleNamespace(
        uid=b"completion-model",
        generate=AsyncMock(return_value=disconnected_stream()),
        abort_request=AsyncMock(return_value="DONE"),
        decrease_serve_count=AsyncMock(),
        is_vllm_backend=AsyncMock(return_value=True),
    )

    async def require_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(restful_api, "require_model", require_model)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock()
    response = await api.create_completion(
        _Request(
            {
                "model": "completion-model",
                "prompt": "hello",
                "stream": True,
                "request_id": "completion-request",
            }
        )
    )

    assert await anext(response.body_iterator) == b'{"choices": []}'
    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)

    model.generate.assert_awaited_once()
    call = model.generate.await_args
    assert call.kwargs["request_id"] == "completion-request"
    assert "request_id" not in call.args[1]
    assert "request_id" not in call.kwargs["raw_params"]
    model.abort_request.assert_awaited_once_with("completion-request")
    model.decrease_serve_count.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("phase", ["send", "next_item"])
async def test_real_sse_disconnect_aborts_and_cleans_stream(monkeypatch, kind, phase):
    source, closed, next_item_started = _tracked_stream(wait_after_first=True)
    response, model, decrease_completed = await _create_streaming_response(
        monkeypatch,
        kind,
        source,
        extra_body={"routing_key": "task"} if kind == "chat" else None,
    )

    await _run_asgi_disconnect(response, phase, next_item_started)

    model.abort_request.assert_awaited_once_with(f"{kind}-request")
    model.decrease_serve_count.assert_awaited_once()
    assert closed.is_set()
    assert decrease_completed.is_set()
    if kind == "chat":
        assert (
            model.chat.await_args.kwargs["_chat_reservation_id"]
            == restful_api.require_chat_model.await_args.kwargs["reservation_id"]
        )
        model.release_chat_reservation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
async def test_pd_sse_disconnect_destroys_router_stream(monkeypatch, kind, engine):
    import xoscar as xo

    from ...core.pd_model import PDModelActor
    from ...core.tests.test_pd_model import FakeStage

    released = asyncio.Event()
    original_release = PDModelActor.decrease_serve_count

    async def release_router(self):
        await original_release(self)
        released.set()

    monkeypatch.setattr(PDModelActor, "decrease_serve_count", release_router)
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        prefill = await xo.create_actor(
            FakeStage, address=pool.external_address, uid="p"
        )
        decode = await xo.create_actor(
            FakeStage, wait_after_first=True, address=pool.external_address, uid="d"
        )
        router = await xo.create_actor(
            PDModelActor,
            "pd",
            model_engine=engine,
            transport_backend="nixl",
            address=pool.external_address,
        )
        await router.add_prefill_actor("p", prefill)
        await router.add_decode_actor("d", decode)
        response, model, decreased = await _create_streaming_response(
            monkeypatch, kind, None, limit_getter=router.get_limit_mm_per_prompt
        )
        method = "generate" if kind == "completion" else "chat"
        streams = []

        async def infer(*args, **kwargs):
            stream = await getattr(router, method)(*args, **kwargs)
            streams.append(stream)
            return stream

        async def abort(request_id):
            return await router.abort_request(request_id)

        getattr(model, method).side_effect = infer
        model.abort_request.side_effect = abort
        await _run_asgi_disconnect(response, "send", None)
        assert await decode.get_stream_counts() == (0, 1)
        model.abort_request.assert_awaited_once_with(f"{kind}-request")
        assert released.is_set()
        assert not decreased.is_set()
        model.decrease_serve_count.assert_not_called()
        assert len(streams) == 1
        # API cleanup owns destroy, so GC must not start a blocking retry.
        assert streams[0]._gc_destroy is False
        with patch.object(streams[0], "destroy", new_callable=AsyncMock) as destroy:
            streams[0].__del__()
        destroy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_explicit_stream_close_aborts_and_cleans_stream(monkeypatch, kind):
    source, closed, _ = _tracked_stream(wait_after_first=True)
    response, model, decrease_completed = await _create_streaming_response(
        monkeypatch, kind, source
    )

    assert await anext(response.body_iterator) == b'{"choices": []}'
    await response.body_iterator.aclose()

    model.abort_request.assert_awaited_once_with(f"{kind}-request")
    model.decrease_serve_count.assert_awaited_once()
    assert closed.is_set()
    assert decrease_completed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_normal_stream_completion_does_not_abort(monkeypatch, kind):
    source, closed, _ = _tracked_stream(wait_after_first=False)
    response, model, decrease_completed = await _create_streaming_response(
        monkeypatch, kind, source
    )

    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks[0] == b'{"choices": []}'
    if kind == "chat":
        assert chunks[-1] == "[DONE]"
    model.abort_request.assert_not_awaited()
    model.decrease_serve_count.assert_awaited_once()
    assert closed.is_set()
    assert decrease_completed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("is_vllm", [False, True])
async def test_disconnect_after_stream_completion_does_not_abort(
    monkeypatch, kind, is_vllm
):
    source, closed, _ = _tracked_stream(wait_after_first=False)
    response, model, decrease_completed = await _create_streaming_response(
        monkeypatch, kind, source, is_vllm=is_vllm
    )
    completed = asyncio.Event()

    async def receive():
        await completed.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] != "http.response.body":
            return
        if b"[DONE]" in message.get("body", b"") or not message.get("more_body", False):
            completed.set()
            # Let the ASGI disconnect listener run after generation finished.
            await asyncio.Event().wait()

    await asyncio.wait_for(
        response(
            {"type": "http", "method": "POST", "asgi": {"version": "3.0"}},
            receive,
            send,
        ),
        timeout=5,
    )
    model.abort_request.assert_not_awaited()
    model.decrease_serve_count.assert_awaited_once()
    assert closed.is_set() and decrease_completed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_iterator_close_failure_still_releases_serve_count(monkeypatch, kind):
    close_attempted = asyncio.Event()

    async def failing_close_stream():
        try:
            yield b'{"choices": []}'
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            close_attempted.set()
            raise RuntimeError("close failed")

    response, model, decrease_completed = await _create_streaming_response(
        monkeypatch, kind, failing_close_stream()
    )

    assert await anext(response.body_iterator) == b'{"choices": []}'
    await response.body_iterator.aclose()

    model.abort_request.assert_awaited_once_with(f"{kind}-request")
    model.decrease_serve_count.assert_awaited_once()
    assert close_attempted.is_set()
    assert decrease_completed.is_set()


@pytest.mark.asyncio
async def test_chat_passes_request_id_outside_generation_config(monkeypatch):
    model = SimpleNamespace(
        uid=b"chat-model",
        chat=AsyncMock(return_value=b'{"choices": []}'),
        is_vllm_backend=AsyncMock(return_value=True),
        get_limit_mm_per_prompt=AsyncMock(return_value=None),
    )
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": "test-family"})
    )

    async def require_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(restful_api, "require_model", require_model)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    response = await api.create_chat_completion(
        _Request(
            {
                "model": "chat-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "request_id": "chat-request",
            }
        )
    )

    assert response.body == b'{"choices": []}'
    model.chat.assert_awaited_once()
    call = model.chat.await_args
    assert call.kwargs["request_id"] == "chat-request"
    assert "request_id" not in call.args[1]
    assert "request_id" not in call.kwargs["raw_params"]


@pytest.mark.asyncio
async def test_non_vllm_chat_keeps_request_id_in_generation_config(monkeypatch):
    model = SimpleNamespace(
        uid=b"chat-model",
        chat=AsyncMock(return_value=b'{"choices": []}'),
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
    )
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": "test-family"})
    )

    async def require_model(*_args, **_kwargs):
        return model

    monkeypatch.setattr(restful_api, "require_model", require_model)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    response = await api.create_chat_completion(
        _Request(
            {
                "model": "chat-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "request_id": "chat-request",
            }
        )
    )

    assert response.body == b'{"choices": []}'
    call = model.chat.await_args
    assert call.args[1]["request_id"] == "chat-request"
    assert call.kwargs["request_id"] == "chat-request"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_vllm_terminal_error_reaches_http_or_sse_error(monkeypatch, stream):
    import json

    from fastapi import HTTPException

    from xinference.model.llm.vllm.core import VLLMModel

    engine_model = object.__new__(VLLMModel)
    engine_model._engine = SimpleNamespace(abort=AsyncMock())
    engine_model._active_request_ids = set()

    async def failed_outputs():
        yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason="error")])

    async def chat(messages, config, **kwargs):
        outputs = engine_model._track_engine_request(
            kwargs["request_id"], failed_outputs()
        )
        if config.get("stream"):
            return outputs
        async for _ in outputs:
            pytest.fail("terminal error was returned as a successful result")

    model = SimpleNamespace(
        uid=b"chat-model",
        chat=chat,
        is_vllm_backend=AsyncMock(return_value=True),
        get_limit_mm_per_prompt=AsyncMock(return_value=None),
        abort_request=AsyncMock(),
        decrease_serve_count=AsyncMock(),
    )
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(
        return_value=SimpleNamespace(
            describe_model=AsyncMock(return_value={"model_family": "test-family"})
        )
    )
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    req = _Request(
        {
            "model": "chat-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "request_id": "failed",
        }
    )
    if stream:
        response = await api.create_chat_completion(req)
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 1
        assert (
            json.loads(chunks[0]["data"])["error"]
            == api._chat_completion_error_detail()
        )
    else:
        with pytest.raises(HTTPException) as exc:
            await api.create_chat_completion(req)
        assert exc.value.status_code == 500
        assert exc.value.detail == api._chat_completion_error_detail()
    assert any(
        "finish_reason=error" in str(call.args[-1])
        for call in api._report_error_event.await_args_list
    )
    assert not engine_model._active_request_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_completion_reports_the_model_error_when_generation_cannot_start(
    monkeypatch, stream
):
    import json

    import httpx
    from fastapi import FastAPI

    from ...core.model import ModelActor

    backend = SimpleNamespace(
        generate=MagicMock(), model_family=SimpleNamespace(to_description=dict)
    )
    model = ModelActor("supervisor", "worker", backend, "completion-model-0")
    model.uid = b"completion-model-rep0"
    # A failed reload leaves the actor here; _require_ready raises RuntimeError.
    model._model_state = "error"
    model.record_metrics = AsyncMock()
    model.is_vllm_backend = AsyncMock(return_value=False)
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    api = _new_api()
    api.handle_request_limit_error = restful_api.RESTfulAPI.handle_request_limit_error
    api._get_supervisor_ref = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda uid, error: error)
    app = FastAPI()
    app.add_api_route("/v1/completions", api.create_completion, methods=["POST"])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "completion-model", "prompt": "hello", "stream": stream},
        )

    if stream:
        assert response.status_code == 200
        events = [
            json.loads(line[len("data:") :])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events == [{"error": "Model is in error state"}]
    else:
        assert response.status_code == 500
        assert response.json() == {"detail": "Model is in error state"}
    backend.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "completion"])
@pytest.mark.parametrize("failure", ["duplicate", "limit"])
async def test_rejected_stream_setup_does_not_abort_request_owner(
    monkeypatch, kind, failure
):
    from ...core.utils import CancelMixin

    owner = CancelMixin()
    request_id = "rejected-request"
    running_id = request_id if failure == "duplicate" else "other-request"
    entered, release = asyncio.Event(), asyncio.Event()
    disconnect_handlers = []

    async def running_request():
        owner._add_running_task(running_id)
        entered.set()
        await release.wait()

    async def last_error(self, model_uid, error):
        return error

    async def reject(*args, **kwargs):
        if failure == "duplicate":
            owner._add_running_task(request_id)
            pytest.fail("The duplicate request must be rejected")
        raise RuntimeError("Rate limit reached for the model")

    async def abort(rejected_id):
        owner._cancel_running_task(rejected_id, block_duration=30)
        return "DONE"

    monkeypatch.setattr(restful_api.RESTfulAPI, "_get_model_last_error", last_error)
    monkeypatch.setattr(
        restful_api,
        "_observe_sse_disconnect",
        lambda response, handler: disconnect_handlers.append(handler),
    )

    first = asyncio.create_task(running_request())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        response, model, _ = await _create_streaming_response(
            monkeypatch,
            kind,
            None,
            extra_body={"request_id": request_id},
        )
        method = "generate" if kind == "completion" else "chat"
        getattr(model, method).side_effect = reject
        model.abort_request.side_effect = abort

        event = await anext(response.body_iterator)
        assert "error" in json.loads(event["data"])
        model.abort_request.assert_not_awaited()

        # Disconnect while the error is yielded, before the generator's finally.
        assert len(disconnect_handlers) == 1
        await disconnect_handlers[0]({"type": "http.disconnect"})
        model.abort_request.assert_not_awaited()
        model.decrease_serve_count.assert_not_awaited()
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
        assert not first.done()
        assert owner._running_tasks[running_id] is first

        if failure == "limit":
            # Rejected setup must not install an abort block for this fresh ID.
            owner._add_running_task(request_id)
            assert owner._running_tasks[request_id] is asyncio.current_task()
    finally:
        release.set()
        tasks = {first, *list(owner._running_tasks.values())}
        tasks.discard(asyncio.current_task())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_explicit_zero_block_abort_ends_sse_without_reblocking(monkeypatch, kind):
    import httpx
    from fastapi import FastAPI

    from ...core.tests.test_request_priority import _close
    from ...core.tests.test_request_tracking import _streaming_actor
    from ...core.utils import CancelMixin

    method = "generate" if kind == "completion" else "chat"
    actor, _, entered = _streaming_actor(method, native=True)
    response, model, _ = await _create_streaming_response(
        monkeypatch, kind, None, extra_body={"request_id": "r"}
    )

    async def infer(*args, **kwargs):
        stream = await asyncio.create_task(getattr(actor, method)(*args, **kwargs))
        stream._actor_ref, stream._gc_destroy = actor, False
        return stream

    getattr(model, method).side_effect = infer
    model.abort_request.side_effect = actor.abort_request
    model.decrease_serve_count.side_effect = actor.decrease_serve_count

    async def abort(model_uid, request_id, block_duration, **kwargs):
        return {"msg": await actor.abort_request(request_id, block_duration)}

    api = _new_api()
    CancelMixin.__init__(api)
    api._get_supervisor_ref = AsyncMock(
        return_value=SimpleNamespace(abort_request=abort)
    )
    app = FastAPI()
    app.add_api_route(
        "/v1/models/{model_uid}/requests/{request_id}/abort",
        api.abort_request,
        methods=["POST"],
    )
    pending = replacement = None
    try:
        await anext(response.body_iterator)
        pending = asyncio.create_task(anext(response.body_iterator))
        await entered.wait()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.post(
                f"/v1/models/{kind}-model/requests/r/abort", json={"block_duration": 0}
            )
        assert result.status_code == 200 and result.json() == {"msg": "DONE"}
        # Chat emits [DONE]; completions finish their pending read with EOF.
        await asyncio.gather(pending, return_exceptions=True)
        async for _ in response.body_iterator:
            pass
        replacement = await infer("hello", {"stream": True}, request_id="r")
        model.abort_request.assert_not_awaited()
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await response.body_iterator.aclose()
        if replacement is not None:
            await _close(replacement)
        blocks = [
            task
            for owner in (actor, api)
            for task in owner._running_tasks.values()
            if task.get_name() == owner._CANCEL_TASK_NAME
        ]
        for task in blocks:
            task.cancel()
        await asyncio.gather(*blocks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("native", [False, True])
async def test_late_sse_disconnect_does_not_abort_replacement(
    monkeypatch, kind, native
):
    import httpx
    from fastapi import FastAPI

    from ...core.tests.test_request_priority import _close
    from ...core.tests.test_request_tracking import _streaming_actor
    from ...core.utils import CancelMixin

    method = "generate" if kind == "completion" else "chat"
    actor, _, _ = _streaming_actor(method, native=native)

    async def chunks():
        yield {"choices": []}
        yield {"choices": []}
        await asyncio.Event().wait()

    async def generate(*args, **kwargs):
        return chunks()

    setattr(actor._model, f"async_{method}", generate)
    response, model, _ = await _create_streaming_response(
        monkeypatch, kind, None, extra_body={"request_id": "r"}
    )

    async def infer(*args, **kwargs):
        stream = await asyncio.create_task(getattr(actor, method)(*args, **kwargs))
        stream._actor_ref, stream._gc_destroy = actor, False
        return stream

    getattr(model, method).side_effect = infer
    model.abort_request.side_effect = actor.abort_request
    model.decrease_serve_count.side_effect = actor.decrease_serve_count

    async def abort(model_uid, request_id, block_duration, **kwargs):
        return {"msg": await actor.abort_request(request_id, block_duration)}

    api = _new_api()
    CancelMixin.__init__(api)
    api._get_supervisor_ref = AsyncMock(
        return_value=SimpleNamespace(abort_request=abort)
    )
    app = FastAPI()
    app.add_api_route(
        "/v1/models/{model_uid}/requests/{request_id}/abort",
        api.abort_request,
        methods=["POST"],
    )
    sending, disconnected = asyncio.Event(), asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            sending.set()
            await asyncio.Event().wait()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    task = asyncio.create_task(
        response({"type": "http", "method": "POST"}, receive, send)
    )
    replacement = None
    try:
        await asyncio.wait_for(sending.wait(), 3)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.post(
                f"/v1/models/{kind}-model/requests/r/abort", json={"block_duration": 0}
            )
        assert result.status_code == 200 and result.json() == {"msg": "DONE"}
        assert not task.done()  # The old response has not consumed EOF.
        replacement = await infer("hello", {"stream": True}, request_id="r")
        assert await anext(replacement)
        disconnected.set()
        await asyncio.wait_for(task, 3)
        assert await anext(replacement)
        if native:
            actor._model.abort_request.assert_awaited_once_with("r")
    finally:
        disconnected.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await response.body_iterator.aclose()
        if replacement is not None:
            await _close(replacement)
            await actor.decrease_serve_count()
        blocks = [
            t
            for owner in (actor, api)
            for t in owner._running_tasks.values()
            if t.get_name() == owner._CANCEL_TASK_NAME
        ]
        for t in blocks:
            t.cancel()
        await asyncio.gather(*blocks, return_exceptions=True)
