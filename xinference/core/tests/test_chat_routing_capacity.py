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
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from packaging.version import Version

from ...model.llm.vllm.core import VLLMModel
from ..exceptions import ModelNotReadyError
from ..model import ModelActor, request_limit
from ..rpc_context import actor_call
from ..supervisor import SupervisorActor
from ..utils import build_replica_model_uid


def _chat_actor(slots, active=0, request_limit=float("inf"), **engine_config):
    backend = VLLMModel.__new__(VLLMModel)
    backend._model_config = {"max_num_seqs": slots, **engine_config}
    actor = ModelActor.__new__(ModelActor)
    actor._model = backend
    actor._model_state = "ready"
    actor._serve_count = active
    actor._chat_reservations = {}
    actor._chat_preflight_reservations = set()
    actor._request_limits = request_limit
    actor._realtime_runtime = None
    return actor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capacity", [None, "32", "bad", 0, -1, True, False, 1.5, [], {}]
)
async def test_invalid_engine_capacity_preserves_normal_admission(capacity):
    actor = _chat_actor(capacity, active=3)
    assert actor._model.get_max_concurrent_requests() is None
    assert await actor.can_accept_chat_request()
    actor._request_limits = 3
    assert not await actor.can_accept_chat_request()


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"data_parallel_size": 2}, 8),
        ({"data_parallel_size": 4, "data_parallel_size_local": 2}, 16),
        ({"data_parallel_size": 4, "data_parallel_rank": 0}, 4),
        ({"data_parallel_size": 4, "data_parallel_external_lb": True}, 4),
        (
            {
                "data_parallel_size": 4,
                "data_parallel_hybrid_lb": True,
                "data_parallel_size_local": 2,
            },
            8,
        ),
        ({"data_parallel_size": 4, "data_parallel_hybrid_lb": True}, None),
        ({"data_parallel_size": "2"}, None),
        ({"data_parallel_size": 0}, None),
        ({"data_parallel_size": -1}, None),
        ({"data_parallel_size": True}, None),
        ({"data_parallel_size": []}, None),
    ],
)
def test_engine_capacity_counts_only_ranks_served_by_this_replica(options, expected):
    assert _chat_actor(4, **options)._model.get_max_concurrent_requests() == expected


def test_default_sanitized_model_config_exposes_sequence_capacity(monkeypatch):
    monkeypatch.setattr(
        "xinference.model.llm.vllm.core.VLLM_VERSION", Version("0.32.0")
    )
    backend = VLLMModel.__new__(VLLMModel)
    backend.model_family = SimpleNamespace(
        architectures=[], has_architecture=lambda _: False
    )
    backend.model_spec = SimpleNamespace(model_format="pytorch")
    backend._device_count = backend._n_worker = 1
    backend._model_config = backend._sanitize_model_config({})
    assert backend.get_max_concurrent_requests() == 256


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "slots,active,request_limit,expected",
    [
        (1, 0, float("inf"), True),
        (1, 1, float("inf"), False),
        (4, 1, float("inf"), True),
        (4, 3, float("inf"), True),
        (4, 4, float("inf"), False),
        (4, 5, float("inf"), False),
        (None, 4, float("inf"), True),
        (4, 1, 2, True),
        (4, 2, 2, False),
    ],
)
async def test_chat_capacity_observes_engine_and_request_limits(
    slots, active, request_limit, expected
):
    actor = _chat_actor(slots, active, request_limit)
    assert await actor.can_accept_chat_request() is expected


@pytest.mark.asyncio
async def test_chat_capacity_preserves_other_backends_and_readiness():
    actor = _chat_actor(1, active=3)
    actor._model = SimpleNamespace()
    assert await actor.can_accept_chat_request()
    actor._request_limits = 4
    actor._realtime_runtime = SimpleNamespace(active_count=1)
    assert not await actor.can_accept_chat_request()
    actor._realtime_runtime = None
    actor._model_state = "reloading"
    assert not await actor.can_accept_chat_request()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "slots,active,target,dp",
    [
        (1, 1, 1, 1),
        (4, 1, 0, 1),
        (4, 4, 1, 1),
        (4, 4, 0, 2),
        (4, 7, 0, 2),
        (4, 8, 1, 2),
    ],
)
async def test_affinity_avoids_saturated_engines_and_preserves_batching(
    slots, active, target, dp
):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(2)
    supervisor._model_uid_to_replica_info = {"demo": info}
    models = [
        _chat_actor(slots, data_parallel_size=dp),
        _chat_actor(slots, data_parallel_size=dp),
    ]
    supervisor._replica_model_uid_to_worker = {
        build_replica_model_uid("demo", index): SimpleNamespace(
            get_model=AsyncMock(return_value=model)
        )
        for index, model in enumerate(models)
    }
    assert await supervisor.get_model("demo", routing_key="task") is models[0]
    models[0]._serve_count = active
    assert await supervisor.get_model("demo", routing_key="task") is models[target]
    models[0]._serve_count = 0
    assert await supervisor.get_model("demo", routing_key="task") is models[target]
    assert info.routing_keys["task"] == target


@pytest.mark.asyncio
async def test_all_saturated_engines_keep_affinity_and_normal_admission():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(2)
    info.routing_keys["task"] = 1
    supervisor._model_uid_to_replica_info = {"demo": info}
    models = [_chat_actor(1, active=1), _chat_actor(1, active=1)]
    supervisor._replica_model_uid_to_worker = {
        build_replica_model_uid("demo", index): SimpleNamespace(
            get_model=AsyncMock(return_value=model)
        )
        for index, model in enumerate(models)
    }
    assert await supervisor.get_model("demo", routing_key="task") is models[1]
    assert info.routing_keys["task"] == 1


@pytest.mark.asyncio
async def test_concurrent_affinity_routes_reserve_the_last_slot():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(2)
    info.routing_keys.update({"a": 0, "b": 0})
    supervisor._model_uid_to_replica_info = {"demo": info}
    models = [_chat_actor(1), _chat_actor(1)]
    supervisor._replica_model_uid_to_worker = {
        build_replica_model_uid("demo", index): SimpleNamespace(
            get_model=AsyncMock(return_value=model)
        )
        for index, model in enumerate(models)
    }
    try:
        results = await asyncio.gather(
            supervisor.get_chat_model("demo", "a", "first"),
            actor_call(
                supervisor,
                "get_chat_model",
                "demo",
                "b",
                "second",
                _rpc_correlation_id="test-correlation",
            ),
        )
        assert all(reserved for _, reserved in results)
        assert {model for model, _ in results} == set(models)
        assert not await models[0].can_accept_chat_request()
        assert not await models[1].can_accept_chat_request()
    finally:
        for model in models:
            for reservation in ("first", "second"):
                model.release_chat_reservation(reservation)
    assert await models[0].can_accept_chat_request()
    assert await models[1].can_accept_chat_request()


@pytest.mark.asyncio
async def test_chat_admission_converts_reservation_to_active_request():
    actor = _chat_actor(1, request_limit=1)
    actor.model_uid = lambda: "demo"
    actor._metrics_labels = {}
    actor.record_metrics = AsyncMock()
    entered = asyncio.Event()

    @request_limit
    async def chat(model):
        entered.set()
        await asyncio.Event().wait()

    assert await actor.reserve_chat_request("chat")
    assert await actor.reserve_chat_request("chat")
    assert not await actor.reserve_chat_request("other")
    task = asyncio.create_task(chat(actor, _chat_reservation_id="chat"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert actor._serve_count == 1
        assert not actor._chat_reservations
        assert not await actor.can_accept_chat_request()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert await actor.can_accept_chat_request()


@pytest.mark.asyncio
async def test_pending_reservation_blocks_unreserved_request():
    actor = _chat_actor(1, request_limit=1)
    actor._replica_model_uid = "demo-rep0"
    actor._metrics_labels = {}
    actor.record_metrics = AsyncMock()
    operation = AsyncMock(return_value="done")

    @request_limit
    async def unreserved(model):
        return await operation()

    assert await actor.reserve_chat_request("chat")
    try:
        with pytest.raises(RuntimeError, match="Rate limit reached"):
            await unreserved(actor)
        operation.assert_not_awaited()
    finally:
        actor.release_chat_reservation("chat")
    assert await unreserved(actor) == "done"


@pytest.mark.asyncio
async def test_chat_preflights_share_reservation_at_request_limit():
    actor = _chat_actor(1, request_limit=1)
    actor._replica_model_uid = "demo-rep0"
    actor._metrics_labels = {}
    actor.record_metrics = AsyncMock()
    actor._model.async_preflight_multimodal_request = AsyncMock()
    actor._model.prepare_reasoning_request = AsyncMock()
    actor._call_wrapper_json = AsyncMock(return_value={})
    assert await actor.reserve_chat_request("chat")
    try:
        await actor.preflight_multimodal_request([], {}, _chat_reservation_id="chat")
        await actor.prepare_reasoning_request({}, _chat_reservation_id="chat")
        assert actor._call_wrapper_json.await_count == 2
        assert actor._serve_count == 0
        assert not await actor.reserve_chat_request("other")
    finally:
        actor.release_chat_reservation("chat")
    assert await actor.can_accept_chat_request()


@pytest.mark.asyncio
async def test_active_preflight_leaves_other_slot_available():
    actor = _chat_actor(2, request_limit=2)
    actor._replica_model_uid = "demo-rep0"
    actor._metrics_labels = {}
    actor.record_metrics = AsyncMock()
    actor._model.prepare_reasoning_request = AsyncMock()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def prepare(*args, **kwargs):
        entered.set()
        await finish.wait()

    actor._call_wrapper_json = prepare
    assert await actor.reserve_chat_request("first")
    task = asyncio.create_task(
        actor.prepare_reasoning_request({}, _chat_reservation_id="first")
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert await actor.reserve_chat_request("second")
        assert not await actor.reserve_chat_request("third")
        finish.set()
        await task
        assert not await actor.reserve_chat_request("third")
    finally:
        finish.set()
        await task
        for reservation in ("first", "second", "third"):
            actor.release_chat_reservation(reservation)


@pytest.mark.asyncio
async def test_chat_reservation_expires(monkeypatch):
    monkeypatch.setattr("xinference.core.model._CHAT_RESERVATION_TTL", 0)
    actor = _chat_actor(1)
    released = asyncio.Event()
    release = actor.release_chat_reservation

    def on_expire(reservation_id):
        release(reservation_id)
        released.set()

    actor.release_chat_reservation = on_expire
    assert await actor.reserve_chat_request("chat")
    assert not await actor.can_accept_chat_request()
    await asyncio.wait_for(released.wait(), 1)
    assert await actor.can_accept_chat_request()


def _reservation_supervisor(*models):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(len(models))
    info.routing_keys["task"] = 0
    supervisor._model_uid_to_replica_info = {"demo": info}
    supervisor._replica_model_uid_to_worker = {
        build_replica_model_uid("demo", index): SimpleNamespace(
            get_model=AsyncMock(return_value=model)
        )
        for index, model in enumerate(models)
    }
    return supervisor, info


@pytest.mark.asyncio
async def test_affinity_timeout_releases_reservation_before_busy_fallback(monkeypatch):
    monkeypatch.setattr("xinference.core.supervisor._AFFINITY_CAPACITY_TIMEOUT", 0.02)
    actor = _chat_actor(1)

    async def reserve(reservation_id):
        assert await actor.reserve_chat_request(reservation_id)
        await asyncio.Event().wait()

    preferred = SimpleNamespace(
        reserve_chat_request=reserve,
        release_chat_reservation=AsyncMock(side_effect=actor.release_chat_reservation),
    )
    busy = SimpleNamespace(reserve_chat_request=AsyncMock(return_value=False))
    supervisor, _ = _reservation_supervisor(preferred, busy)
    assert await supervisor.get_chat_model("demo", "task", "chat") == (preferred, False)
    preferred.release_chat_reservation.assert_awaited_once_with("chat")
    assert await actor.can_accept_chat_request()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_info", [False, True])
async def test_affinity_releases_reservation_when_selected_replica_changes(
    replace_info,
):
    preferred = SimpleNamespace(release_chat_reservation=AsyncMock())
    other = SimpleNamespace(reserve_chat_request=AsyncMock(return_value=True))
    supervisor, info = _reservation_supervisor(preferred, other)

    async def reserve(_reservation_id):
        if replace_info:
            supervisor._model_uid_to_replica_info["demo"] = (
                supervisor._build_replica_info(2)
            )
        else:
            info.active_replica_ids.remove(0)
        return True

    preferred.reserve_chat_request = reserve
    if replace_info:
        with pytest.raises(ModelNotReadyError, match="no ready replica"):
            await supervisor.get_chat_model("demo", "task", "chat")
        supervisor._replica_model_uid_to_worker[
            build_replica_model_uid("demo", 1)
        ].get_model.assert_not_awaited()
    else:
        assert await supervisor.get_chat_model("demo", "task", "chat") == (other, False)
    preferred.release_chat_reservation.assert_awaited_once_with("chat")


@pytest.mark.asyncio
async def test_affinity_probes_legacy_model_without_reservation():
    legacy = SimpleNamespace(can_accept_chat_request=AsyncMock(return_value=True))
    other = SimpleNamespace(reserve_chat_request=AsyncMock(return_value=True))
    supervisor, _ = _reservation_supervisor(legacy, other)
    assert await supervisor.get_chat_model("demo", "task", "chat") == (legacy, False)
    legacy.can_accept_chat_request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [False, True])
async def test_affinity_checks_all_replicas_during_concurrent_round_robin(unavailable):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(3)
    info.routing_keys["task"] = 0
    supervisor._model_uid_to_replica_info = {"demo": info}
    models = [_chat_actor(1, active=int(index < 2)) for index in range(3)]
    probing, release = asyncio.Event(), asyncio.Event()

    async def get_model(index, **kwargs):
        if index == 1:
            probing.set()
            await release.wait()
        if unavailable and index < 2:
            raise ModelNotReadyError("loading")
        return models[index]

    supervisor._replica_model_uid_to_worker = {
        build_replica_model_uid("demo", index): SimpleNamespace(
            get_model=AsyncMock(side_effect=partial(get_model, index))
        )
        for index in range(3)
    }
    sticky = asyncio.create_task(supervisor.get_model("demo", routing_key="task"))
    try:
        await asyncio.wait_for(probing.wait(), 1)
        assert await supervisor.get_model("demo") is models[2]
    finally:
        release.set()
    assert await sticky is models[2]
    assert info.routing_keys["task"] == 2
