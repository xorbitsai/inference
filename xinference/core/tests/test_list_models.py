import itertools
from typing import Any, Dict

import pytest

from xinference.core.supervisor import ReplicaInfo, SupervisorActor


class _DummyWorker:
    """Mimics WorkerActor.list_models() returning replica-model-uid keyed specs."""

    def __init__(self, models: Dict[str, Dict[str, Any]]):
        self._models = models

    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._models)


class DummySupervisor:
    """Borrow the unbound SupervisorActor.list_models with only the state it reads."""

    list_models = SupervisorActor.list_models

    def __init__(self, address: str, worker: _DummyWorker):
        self.address = address
        self._worker_address_to_worker = {address: worker}
        self._model_uid_to_replica_info: Dict[str, ReplicaInfo] = {}
        self._list_models_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._replica_gpu_cache: Dict[str, list] = {}


def _replica_info(replica: int) -> ReplicaInfo:
    active = list(range(replica))
    return ReplicaInfo(
        replica=replica,
        scheduler=itertools.cycle(active),
        active_replica_ids=active,
    )


@pytest.mark.asyncio
async def test_list_models_drops_stale_uid_without_replica_info():
    """A failed launch can leave a worker reporting a replica uid the supervisor
    no longer tracks. list_models must skip it instead of raising KeyError so
    that healthy models are still listed (issue #5167)."""
    address = "127.0.0.1:1234"
    worker = _DummyWorker(
        {
            # Healthy model, replica 0.
            "qwen3-0": {"model_name": "qwen3", "model_type": "LLM"},
            # Stale replica from a failed qwen3.5 launch; no replica info left.
            "qwen3.5-0": {"model_name": "qwen3.5", "model_type": "LLM"},
        }
    )
    supervisor = DummySupervisor(address, worker)
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(1)

    result = await supervisor.list_models()

    assert len(result) == 1
    assert "qwen3" in result
    assert result["qwen3"]["replica"] == 1
    # The stale, un-tracked model must be dropped, not raise.
    assert "qwen3.5" not in result


@pytest.mark.asyncio
async def test_list_models_returns_healthy_models():
    address = "127.0.0.1:1234"
    worker = _DummyWorker(
        {
            "qwen3-0": {"model_name": "qwen3", "model_type": "LLM"},
            "qwen3-1": {"model_name": "qwen3", "model_type": "LLM"},
        }
    )
    supervisor = DummySupervisor(address, worker)
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(2)

    result = await supervisor.list_models()

    assert set(result) == {"qwen3"}
    assert result["qwen3"]["replica"] == 2
