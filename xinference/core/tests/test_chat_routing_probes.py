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
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from ...api.restful_api import RESTfulAPI
from .test_chat_routing_capacity import _reservation_supervisor
from .test_request_priority import _actor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["lookup", "lookup_no_load", "load", "tokens", "prefixes"]
)
async def test_stalled_routing_probe_does_not_hold_up_chat(monkeypatch, stage):
    monkeypatch.setattr("xinference.core.supervisor._CHAT_ROUTING_PROBE_TIMEOUT", 0.02)
    monkeypatch.setattr("xinference.constants.XINFERENCE_GET_MODEL_RPC_TIMEOUT", 0.4)
    monkeypatch.setattr(
        "xinference.api.restful_api.XINFERENCE_TOKEN_ROUTER_ENABLED", False
    )
    entered, release = asyncio.Event(), asyncio.Event()
    probe_tasks = []

    async def stalled(*args, **kwargs):
        # Match an actor RPC whose CancelMessage cannot be acknowledged until
        # the destination recovers. asyncio.wait_for waits for that reply.
        probe_tasks.append(asyncio.current_task())
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    actors, refs = [], []
    for index in range(2):
        actor, model = _actor()
        model.async_chat = AsyncMock(return_value={"choices": [], "served_by": index})
        actors.append(actor)
        refs.append(
            SimpleNamespace(
                uid=f"demo-rep{index}".encode(),
                chat=actor.chat,
                get_chat_routing_load=actor.get_chat_routing_load,
                get_chat_routing_tokens=AsyncMock(return_value=[1, 2]),
                reserve_chat_request=actor.reserve_chat_request,
                release_chat_reservation=AsyncMock(
                    side_effect=actor.release_chat_reservation
                ),
                is_vllm_backend=AsyncMock(return_value=False),
                is_transformers_batching_enabled=AsyncMock(return_value=False),
            )
        )
    if stage == "load":
        refs[0].get_chat_routing_load = stalled
    elif stage == "tokens":
        refs[0].get_chat_routing_tokens = stalled
    supervisor, _ = _reservation_supervisor(*refs)
    if stage in ("lookup", "lookup_no_load"):
        supervisor._replica_model_uid_to_worker["demo-rep0"].get_model = stalled
    if stage == "lookup_no_load":
        del refs[1].get_chat_routing_load
    supervisor.describe_model = AsyncMock(return_value={"model_family": "test-family"})
    if stage in ("tokens", "prefixes"):
        supervisor._block_tracker_mapping = {
            "demo": SimpleNamespace(
                match_routing_prefixes=(
                    stalled if stage == "prefixes" else AsyncMock(return_value={})
                )
            )
        }
    api = object.__new__(RESTfulAPI)
    api._set_trace_model = Mock()
    api._set_trace_model_type = Mock()
    api._check_model_access = Mock()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._report_error_event = AsyncMock()
    app = FastAPI()
    app.add_api_route(
        "/v1/chat/completions", api.create_chat_completion, methods=["POST"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "routing_key": "session",
                },
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            done, _ = await asyncio.wait([request], timeout=0.3)
            assert request in done, (
                "chat is waiting for a routing probe cancellation; "
                f"backend calls: {[a._model.async_chat.await_count for a in actors]}; "
                f"serve counts: {[a._serve_count for a in actors]}"
            )
            response = request.result()
            assert response.status_code == 200, response.text
            assert response.json()["served_by"] == (
                1 if stage in ("lookup", "lookup_no_load", "load") else 0
            )
            assert len(probe_tasks) == 1
            assert sum(a._model.async_chat.await_count for a in actors) == 1
            assert all(actor._serve_count == 0 for actor in actors)
            assert all(not actor._chat_reservations for actor in actors)
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(request, *probe_tasks, return_exceptions=True), 2
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["missing", "error", "none"])
async def test_no_load_ranking_keeps_reachable_refs_in_order(signal):
    models = [SimpleNamespace() for _ in range(4)]
    for index in (1, 2):
        if signal != "missing":
            models[index].get_chat_routing_load = AsyncMock(
                return_value=None,
                side_effect=(
                    RuntimeError("telemetry unavailable") if signal == "error" else None
                ),
            )
    supervisor, info = _reservation_supervisor(*models)
    for index in (0, 3):
        supervisor._replica_model_uid_to_worker[
            f"demo-rep{index}"
        ].get_model.side_effect = asyncio.TimeoutError

    ranked, reachable = await supervisor._rank_chat_replicas(
        "demo", info, 0, {"interactive": True}
    )
    assert reachable == {1: models[1], 2: models[2]}
    assert ranked == [1, 2, 0, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("load", [None, 0])
async def test_ranked_fallback_keeps_busy_handling_after_ordinary_lookup(
    monkeypatch, load
):
    monkeypatch.setattr("xinference.core.supervisor._CHAT_ROUTING_PROBE_TIMEOUT", 0.02)
    monkeypatch.setattr("xinference.constants.XINFERENCE_GET_MODEL_RPC_TIMEOUT", 0.4)
    healthy = SimpleNamespace(
        get_chat_routing_load=AsyncMock(return_value=load),
        can_accept_chat_request=AsyncMock(return_value=False),
    )
    supervisor, _ = _reservation_supervisor(SimpleNamespace(), healthy)
    release = asyncio.Event()

    async def stalled(**kwargs):
        await release.wait()
        raise ConnectionError("worker unavailable")

    worker = supervisor._replica_model_uid_to_worker["demo-rep0"]
    worker.get_model.side_effect = stalled
    supervisor._get_replica_model = AsyncMock(wraps=supervisor._get_replica_model)
    request = asyncio.create_task(
        supervisor.get_chat_model(
            "demo", "task", None, routing_request={"interactive": True}
        )
    )
    try:
        done, _ = await asyncio.wait([request], timeout=0.8)
        assert request in done, "ranked fallback exceeded the ordinary lookup timeout"
        assert request.result() == (healthy, False)
        calls = supervisor._get_replica_model.await_args_list
        retries = [call for call in calls if call.args[1] == 0]
        assert len(retries) == 2
        assert [call.kwargs["rpc_timeout"] for call in retries] == [0.02, None]
        assert sum(call.args[1] == 1 for call in calls) == 1
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(request, return_exceptions=True), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stalled_peer", [False, True], ids=["slow_idle", "stalled_and_idle"]
)
async def test_full_fast_replica_routes_chat_to_slow_idle_replica(
    monkeypatch, stalled_peer
):
    monkeypatch.setattr("xinference.core.supervisor._CHAT_ROUTING_PROBE_TIMEOUT", 0.02)
    monkeypatch.setattr("xinference.constants.XINFERENCE_GET_MODEL_RPC_TIMEOUT", 0.4)
    monkeypatch.setattr(
        "xinference.api.restful_api.XINFERENCE_TOKEN_ROUTER_ENABLED", False
    )
    actors, refs = [], []
    for index in range(3 if stalled_peer else 2):
        actor, model = _actor(request_limits=1)
        model.async_chat = AsyncMock(return_value={"choices": [], "served_by": index})
        actors.append(actor)
        refs.append(
            SimpleNamespace(
                uid=f"demo-rep{index}".encode(),
                chat=actor.chat,
                get_chat_routing_load=actor.get_chat_routing_load,
                reserve_chat_request=actor.reserve_chat_request,
                release_chat_reservation=AsyncMock(
                    side_effect=actor.release_chat_reservation
                ),
                is_vllm_backend=AsyncMock(return_value=False),
                is_transformers_batching_enabled=AsyncMock(return_value=False),
            )
        )
    actors[0]._serve_count = 1
    supervisor, info = _reservation_supervisor(*refs)
    supervisor.get_chat_model = AsyncMock(wraps=supervisor.get_chat_model)
    supervisor.describe_model = AsyncMock(return_value={"model_family": "test-family"})
    idle_index = len(refs) - 1

    async def slow_lookup(**kwargs):
        await asyncio.sleep(0.06)
        return refs[idle_index]

    supervisor._replica_model_uid_to_worker[
        f"demo-rep{idle_index}"
    ].get_model.side_effect = slow_lookup
    stalled_tasks = []
    release = asyncio.Event()

    async def stalled_lookup(**kwargs):
        stalled_tasks.append(asyncio.current_task())
        await release.wait()
        return refs[1]

    if stalled_peer:
        supervisor._replica_model_uid_to_worker["demo-rep1"].get_model.side_effect = (
            stalled_lookup
        )
    api = object.__new__(RESTfulAPI)
    api._set_trace_model = Mock()
    api._set_trace_model_type = Mock()
    api._check_model_access = Mock()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    api._report_error_event = AsyncMock()
    app = FastAPI()
    app.add_api_route(
        "/v1/chat/completions", api.create_chat_completion, methods=["POST"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "routing_key": "task",
                },
            )
        )
        try:
            done, _ = await asyncio.wait([request], timeout=0.25)
            assert request in done, "idle replica waited for a stalled worker lookup"
            response = request.result()
            assert response.status_code == 200, response.text
            assert response.json()["served_by"] == idle_index
            actors[0]._model.async_chat.assert_not_awaited()
            actors[idle_index]._model.async_chat.assert_awaited_once()
            routing_key = supervisor.get_chat_model.await_args.args[1]
            assert info.routing_keys[routing_key] == idle_index
            assert all(not actor._chat_reservations for actor in actors)
            if stalled_peer:
                assert len(stalled_tasks) == 2  # Probe, then ordinary lookup.
                assert all(task.cancelled() for task in stalled_tasks)
                actors[1]._model.async_chat.assert_not_awaited()
        finally:
            release.set()
            await asyncio.gather(request, *stalled_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_ranked_fallback_keeps_ordinary_deadline_when_no_probe_answered(
    monkeypatch,
):
    monkeypatch.setattr("xinference.core.supervisor._CHAT_ROUTING_PROBE_TIMEOUT", 0.02)
    monkeypatch.setattr("xinference.constants.XINFERENCE_GET_MODEL_RPC_TIMEOUT", 0.4)
    models = [
        SimpleNamespace(
            get_chat_routing_load=AsyncMock(return_value=0),
            can_accept_chat_request=AsyncMock(return_value=True),
        )
        for _ in range(2)
    ]
    supervisor, _ = _reservation_supervisor(*models)
    for index, model in enumerate(models):

        async def slow_lookup(model=model, **kwargs):
            await asyncio.sleep(0.05)
            return model

        supervisor._replica_model_uid_to_worker[
            f"demo-rep{index}"
        ].get_model.side_effect = slow_lookup

    assert await supervisor.get_chat_model(
        "demo", "task", None, routing_request={"interactive": True}
    ) == (models[0], False)
