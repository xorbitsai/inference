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

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ..supervisor import SupervisorActor


@pytest.fixture
def supervisor():
    actor = SupervisorActor.__new__(SupervisorActor)
    actor._worker_address_to_worker = {}
    return actor


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["LLM", "embedding", "rerank", "audio", "world"])
@pytest.mark.parametrize("reverse_workers", [False, True])
async def test_merge_heterogeneous_worker_engines(
    supervisor, model_type, reverse_workers
):
    pytorch = {"model_format": "pytorch", "quantizations": ["none"]}
    quantized = {"model_format": "pytorch", "quantizations": ["int8"]}
    gpu = SimpleNamespace(
        query_engines_by_model_name=AsyncMock(
            return_value={"transformers": [pytorch], "MindIE": "Not installed"}
        )
    )
    npu = SimpleNamespace(
        query_engines_by_model_name=AsyncMock(
            return_value={"MindIE": [pytorch], "transformers": [pytorch, quantized]}
        )
    )
    missing = SimpleNamespace(query_engines_by_model_name=AsyncMock(return_value=None))
    workers = [missing, gpu, npu]
    if reverse_workers:
        workers.reverse()
    supervisor._worker_address_to_worker = dict(enumerate(workers))

    with patch("xinference.core.supervisor.get_engine_params_by_name") as fallback:
        result = await supervisor.query_engines_by_model_name(
            "mixed-model", model_type=model_type, enable_virtual_env=False
        )
        fallback.assert_not_called()

    assert result == {"transformers": [pytorch, quantized], "MindIE": [pytorch]}
    assert list(result) == (
        ["MindIE", "transformers"] if reverse_workers else ["transformers", "MindIE"]
    )
    for worker in workers:
        worker.query_engines_by_model_name.assert_awaited_once_with(
            "mixed-model", model_type=model_type, enable_virtual_env=False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_results", [[], [None, None], [{}, None]])
@pytest.mark.parametrize("enable_virtual_env", [False, True])
async def test_empty_worker_engines_use_local_fallback(
    supervisor, worker_results, enable_virtual_env
):
    supervisor._worker_address_to_worker = {
        str(index): SimpleNamespace(
            query_engines_by_model_name=AsyncMock(return_value=result)
        )
        for index, result in enumerate(worker_results)
    }
    fallback_name = "get_engine_params_by_name"
    if enable_virtual_env:
        fallback_name += "_with_virtual_env"
    with patch(
        f"xinference.core.supervisor.{fallback_name}", return_value={"local": []}
    ) as fallback:
        result = await supervisor.query_engines_by_model_name(
            "missing-model", model_type="LLM", enable_virtual_env=enable_virtual_env
        )

    assert result == {"local": []}
    fallback.assert_called_once_with(
        "LLM", "missing-model", enable_virtual_env=enable_virtual_env
    )
    for worker in supervisor._worker_address_to_worker.values():
        worker.query_engines_by_model_name.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["LLM", "embedding", "rerank"])
async def test_worker_engine_error_is_not_hidden_by_partial_results(
    supervisor, model_type
):
    supervisor._worker_address_to_worker = {
        "healthy": SimpleNamespace(
            query_engines_by_model_name=AsyncMock(return_value={"transformers": []})
        ),
        "unavailable": SimpleNamespace(
            query_engines_by_model_name=AsyncMock(
                side_effect=RuntimeError("Worker offline")
            )
        ),
    }
    with pytest.raises(RuntimeError, match="Worker offline"):
        await supervisor.query_engines_by_model_name(
            "mixed-model", model_type=model_type, enable_virtual_env=False
        )
