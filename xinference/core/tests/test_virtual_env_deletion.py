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

import pytest

from ..supervisor import SupervisorActor
from ..virtual_env_manager import VirtualEnvConflictError


class _WorkerWithVirtualEnv:
    def __init__(self, error=None):
        self._error = error
        self.remove_calls = 0

    async def list_virtual_envs(self, model_name, model_engine):
        return [{"model_name": model_name, "model_engine": model_engine}]

    async def remove_virtual_env(self, model_name, model_engine, python_version):
        self.remove_calls += 1
        if self._error is not None:
            raise self._error
        return True


@pytest.mark.asyncio
async def test_supervisor_preserves_virtual_env_conflict_from_worker():
    conflict = VirtualEnvConflictError("environment is used by qwen-rep0")
    busy_worker = _WorkerWithVirtualEnv(conflict)
    idle_worker = _WorkerWithVirtualEnv()
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._worker_address_to_worker = {
        "worker-0": busy_worker,
        "worker-1": idle_worker,
    }

    with pytest.raises(VirtualEnvConflictError, match="qwen-rep0"):
        await SupervisorActor.remove_virtual_env(
            supervisor,
            model_name="Qwen3.8-Flash-Next",
            model_engine="vllm",
            python_version="3.12",
        )

    assert busy_worker.remove_calls == 1
    assert idle_worker.remove_calls == 1
