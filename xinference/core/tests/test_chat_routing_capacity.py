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

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...model.llm.vllm.core import VLLMModel
from ..model import ModelActor
from ..supervisor import SupervisorActor
from ..utils import build_replica_model_uid


def _chat_actor(slots, active=0, request_limit=float("inf")):
    backend = VLLMModel.__new__(VLLMModel)
    backend._model_config = {"max_num_seqs": slots}
    actor = ModelActor.__new__(ModelActor)
    actor._model = backend
    actor._model_state = "ready"
    actor._serve_count = active
    actor._request_limits = request_limit
    actor._realtime_runtime = None
    return actor


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
@pytest.mark.parametrize("slots,active,target", [(1, 1, 1), (4, 1, 0), (4, 4, 1)])
async def test_affinity_avoids_saturated_engines_and_preserves_batching(
    slots, active, target
):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._pd_model_mapping = {}
    info = supervisor._build_replica_info(2)
    supervisor._model_uid_to_replica_info = {"demo": info}
    models = [_chat_actor(slots), _chat_actor(slots)]
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
