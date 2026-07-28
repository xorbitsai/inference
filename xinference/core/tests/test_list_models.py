import itertools
from typing import Any, Dict

import pytest

from xinference.core.supervisor import ReplicaInfo, SupervisorActor
from xinference.core.utils import build_replica_model_uid


class _DummyWorker:
    """Mimics WorkerActor.list_models() returning replica-model-uid keyed specs."""

    def __init__(self, models: Dict[str, Dict[str, Any]]):
        self._models = models

    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._models)


class _FailingWorker:
    """Mimics a worker whose list_models() raises (timeout/crash)."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        raise self._exc


class DummySupervisor:
    """Borrow the unbound SupervisorActor.list_models with only the state it reads."""

    list_models = SupervisorActor.list_models

    def __init__(self, address: str, workers: Dict[str, Any]):
        self.address = address
        self._worker_address_to_worker = dict(workers)
        self._model_uid_to_replica_info: Dict[str, ReplicaInfo] = {}
        self._list_models_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._replica_gpu_cache: Dict[str, list] = {}
        # worker_address -> replica_model_uid -> {gpu_idx -> bytes}
        self._worker_model_gpu_memory: Dict[str, Dict[str, Dict[int, int]]] = {}


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
            build_replica_model_uid("qwen3", 0): {
                "model_name": "qwen3",
                "model_type": "LLM",
            },
            # Stale replica from a failed qwen3.5 launch; no replica info left.
            build_replica_model_uid("qwen3.5", 0): {
                "model_name": "qwen3.5",
                "model_type": "LLM",
            },
        }
    )
    supervisor = DummySupervisor(address, {address: worker})
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
            build_replica_model_uid("qwen3", 0): {
                "model_name": "qwen3",
                "model_type": "LLM",
            },
            build_replica_model_uid("qwen3", 1): {
                "model_name": "qwen3",
                "model_type": "LLM",
            },
        }
    )
    supervisor = DummySupervisor(address, {address: worker})
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(2)

    result = await supervisor.list_models()

    assert set(result) == {"qwen3"}
    assert result["qwen3"]["replica"] == 2


@pytest.mark.asyncio
async def test_list_models_falls_back_to_cache_when_worker_fails():
    """When one worker's list_models() times out or crashes, the supervisor
    must fall back to that worker's last cached result instead of dropping its
    models, while still merging healthy workers that respond."""
    healthy_addr = "127.0.0.1:1234"
    failing_addr = "127.0.0.1:5678"
    healthy = _DummyWorker(
        {
            build_replica_model_uid("qwen3", 0): {
                "model_name": "qwen3",
                "model_type": "LLM",
            }
        }
    )
    failing = _FailingWorker(TimeoutError("worker unreachable"))

    supervisor = DummySupervisor(
        healthy_addr, {healthy_addr: healthy, failing_addr: failing}
    )
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(1)
    supervisor._model_uid_to_replica_info["llama3"] = _replica_info(1)
    # Last good result seen from the now-failing worker.
    supervisor._list_models_cache[failing_addr] = {
        build_replica_model_uid("llama3", 0): {
            "model_name": "llama3",
            "model_type": "LLM",
        }
    }

    result = await supervisor.list_models()

    assert set(result) == {"qwen3", "llama3"}
    assert result["qwen3"]["replica"] == 1
    assert result["llama3"]["replica"] == 1


@pytest.mark.asyncio
async def test_list_models_failing_worker_without_cache_is_skipped():
    """A worker that fails before ever caching a result contributes nothing,
    but must not break the healthy workers' listing."""
    healthy_addr = "127.0.0.1:1234"
    failing_addr = "127.0.0.1:5678"
    healthy = _DummyWorker(
        {
            build_replica_model_uid("qwen3", 0): {
                "model_name": "qwen3",
                "model_type": "LLM",
            }
        }
    )
    failing = _FailingWorker(RuntimeError("boom"))

    supervisor = DummySupervisor(
        healthy_addr, {healthy_addr: healthy, failing_addr: failing}
    )
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(1)

    result = await supervisor.list_models()

    assert set(result) == {"qwen3"}
    assert result["qwen3"]["replica"] == 1
