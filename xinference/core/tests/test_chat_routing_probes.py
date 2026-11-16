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
@pytest.mark.parametrize("stage", ["lookup", "load", "tokens", "prefixes"])
async def test_stalled_routing_probe_does_not_hold_up_chat(monkeypatch, stage):
    monkeypatch.setattr("xinference.core.supervisor._CHAT_ROUTING_PROBE_TIMEOUT", 0.02)
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
    if stage == "lookup":
        supervisor._replica_model_uid_to_worker["demo-rep0"].get_model = stalled
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
                1 if stage in ("lookup", "load") else 0
            )
            assert sum(a._model.async_chat.await_count for a in actors) == 1
            assert all(actor._serve_count == 0 for actor in actors)
            assert all(not actor._chat_reservations for actor in actors)
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(request, *probe_tasks, return_exceptions=True), 2
            )
