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
from ..model import ModelActor
from ..supervisor import SupervisorActor
from ..utils import build_replica_model_uid


def _chat_actor(slots, active=0, request_limit=float("inf"), **engine_config):
    backend = VLLMModel.__new__(VLLMModel)
    backend._model_config = {"max_num_seqs": slots, **engine_config}
    actor = ModelActor.__new__(ModelActor)
    actor._model = backend
    actor._model_state = "ready"
    actor._serve_count = active
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
