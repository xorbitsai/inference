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
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from ...core.exceptions import ModelReloadInProgress
from ..replica_drain import (
    ReplicaDrainMiddleware,
    ReplicaDrainRequests,
    track_model_lookup,
)
from ..restful_api import RESTfulAPI


@pytest.mark.asyncio
async def test_manual_drain_tracks_through_stream_completion():
    tracker = ReplicaDrainRequests()
    looked_up = asyncio.Event()
    selected = asyncio.Event()
    finished = asyncio.Event()

    async def app(_scope, _receive, _send):
        lookup = track_model_lookup("demo")
        looked_up.set()
        await selected.wait()
        lookup.selected(SimpleNamespace(uid=b"demo-rep1"))
        await finished.wait()

    middleware = ReplicaDrainMiddleware(app, tracker)
    task = asyncio.create_task(
        middleware({"type": "http", "method": "POST"}, AsyncMock(), AsyncMock())
    )
    await looked_up.wait()
    window = tracker.start("demo")
    window.collecting = False
    assert not tracker.drained(window, "demo", b"demo-rep1")
    selected.set()
    await asyncio.sleep(0)
    assert not tracker.drained(window, "demo", b"demo-rep1")
    assert tracker.drained(window, "demo", b"demo-rep0")
    finished.set()
    await task
    assert tracker.drained(window, "demo", b"demo-rep1")


@pytest.mark.asyncio
async def test_manual_drain_rejects_uncertain_stream_outcome():
    tracker = ReplicaDrainRequests()
    selected = asyncio.Event()
    disconnected = asyncio.Event()

    async def app(_scope, receive, _send):
        track_model_lookup("demo").selected(SimpleNamespace(uid=b"demo-rep1"))
        selected.set()
        await disconnected.wait()
        await receive()

    middleware = ReplicaDrainMiddleware(app, tracker)
    task = asyncio.create_task(
        middleware(
            {"type": "http", "method": "POST"},
            AsyncMock(return_value={"type": "http.disconnect"}),
            AsyncMock(),
        )
    )
    await selected.wait()
    window = tracker.start("demo")
    disconnected.set()
    await task
    with pytest.raises(RuntimeError, match="no confirmed outcome"):
        tracker.drained(window, "demo", b"demo-rep1")


@pytest.fixture
def replica_api():
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._uid_to_model_name = {}
    api._app = SimpleNamespace(router=SimpleNamespace(routes=[]))
    return api


@pytest.mark.asyncio
async def test_manual_drain_timeout_cancels_without_deleting(replica_api, monkeypatch):
    monkeypatch.delenv("XINFERENCE_API_HOST", raising=False)
    replica_api._replica_drain_requests = ReplicaDrainRequests()
    supervisor = AsyncMock()
    supervisor.begin_manual_replica_drain.return_value = {"status": "draining"}
    supervisor.confirm_manual_replica_drain.return_value = False
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    with pytest.raises(HTTPException) as raised:
        await replica_api.begin_model_replica_drain("demo", 1, "r1", timeout=0.02)
    assert raised.value.status_code == 408
    supervisor.cancel_manual_replica_drain.assert_awaited_once_with("demo", 1, "r1")
    supervisor.terminate_drained_model_replica.assert_not_awaited()
    assert not replica_api._replica_drain_requests.windows


@pytest.mark.asyncio
@pytest.mark.parametrize("status, expected", [("failed", 409), ("succeeded", 200)])
async def test_manual_drain_with_an_older_api_host(
    replica_api, monkeypatch, status, expected
):
    from ...deploy import api_host

    monkeypatch.setenv(api_host.API_HOST_ENV, "hosted")
    # An older host ignores the liveness option and returns its operation list.
    host = AsyncMock(return_value=[{"status": "failed"}, {"status": status}])
    monkeypatch.setattr(api_host, "host_request", host)
    replica_api._replica_drain_requests = ReplicaDrainRequests()
    supervisor = AsyncMock()
    supervisor.begin_manual_replica_drain.return_value = {"status": "draining"}
    supervisor.confirm_manual_replica_drain.return_value = True
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    if expected == 409:
        with pytest.raises(HTTPException) as raised:
            await replica_api.begin_model_replica_drain("demo", 1, "r1", timeout=1)
        assert raised.value.status_code == expected
        supervisor.begin_manual_replica_drain.assert_not_awaited()
    else:
        response = await replica_api.begin_model_replica_drain(
            "demo", 1, "r1", timeout=1
        )
        assert response.status_code == expected
        assert json.loads(response.body)["status"] == "drained"
    host.assert_awaited_once_with("list", require_single_api=True)
    assert not replica_api._replica_drain_requests.windows


@pytest.mark.asyncio
async def test_add_model_replica_accepts_empty_body(replica_api):
    supervisor = AsyncMock()
    supervisor.add_model_replicas.return_value = [
        {
            "replica_id": 1,
            "replica_model_uid": "demo-rep1",
            "worker_address": "127.0.0.1:9978",
        }
    ]
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    response = await replica_api.add_model_replica("demo")

    assert response.status_code == 200
    assert json.loads(response.body) == supervisor.add_model_replicas.return_value[0]
    supervisor.add_model_replicas.assert_awaited_once_with(
        model_uid="demo",
        replica=1,
        replica_configs=None,
        model_engine=None,
        n_gpu=None,
    )


@pytest.mark.asyncio
async def test_add_model_replicas_forwards_count_engine_and_device(replica_api):
    supervisor = AsyncMock()
    supervisor.add_model_replicas.return_value = [
        {
            "replica_id": replica_id,
            "replica_model_uid": f"demo-rep{replica_id}",
            "worker_address": "127.0.0.1:9978",
        }
        for replica_id in (1, 2)
    ]
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    response = await replica_api.add_model_replica(
        "demo",
        {
            "replica": 2,
            "model_engine": "vllm",
            "n_gpu": 1,
        },
    )

    assert json.loads(response.body) == {
        "replica": 2,
        "replicas": supervisor.add_model_replicas.return_value,
    }
    supervisor.add_model_replicas.assert_awaited_once_with(
        model_uid="demo",
        replica=2,
        replica_configs=None,
        model_engine="vllm",
        n_gpu=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"replica_config": []},
        {"replica_config": {"devices": []}},
        {"replica_config": {"devices": [{"gpu_idx": [0]}]}},
        {"replica": 0},
        {"replica": True},
        {"model_engine": "  "},
        {"n_gpu": -1},
        {"replica": 2, "replica_config": [{"devices": []}]},
    ],
)
async def test_add_model_replica_rejects_invalid_config(replica_api, payload):
    replica_api._get_supervisor_ref = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await replica_api.add_model_replica("demo", payload)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_terminate_last_replica_evicts_api_cache(replica_api, monkeypatch):
    supervisor = AsyncMock()
    supervisor.terminate_model_replica.return_value = 0
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    replica_api._uid_to_model_name["demo"] = "demo-model"
    evict_model_cache = Mock()
    monkeypatch.setattr(
        "xinference.api.oauth2.advanced.audit.evict_model_cache", evict_model_cache
    )

    response = await replica_api.terminate_model_replica("demo", 0)

    assert json.loads(response.body) == {"remaining_replicas": 0}
    assert "demo" not in replica_api._uid_to_model_name
    evict_model_cache.assert_called_once_with("demo")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "remove"])
async def test_resize_during_reload_returns_conflict(replica_api, operation):
    supervisor = AsyncMock()
    supervisor.add_model_replicas.side_effect = ModelReloadInProgress(
        "reload in progress"
    )
    supervisor.terminate_model_replica.side_effect = ModelReloadInProgress(
        "reload in progress"
    )
    replica_api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    with pytest.raises(HTTPException) as exc:
        if operation == "add":
            await replica_api.add_model_replica("demo")
        else:
            await replica_api.terminate_model_replica("demo", 0)
    assert exc.value.status_code == 409
