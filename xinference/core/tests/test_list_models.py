import asyncio
import itertools
import time as time_module
from types import SimpleNamespace
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
    """Borrow focused Supervisor methods with only the state they read."""

    list_models = SupervisorActor.list_models
    _invalidate_list_models_debounce_cache = (
        SupervisorActor._invalidate_list_models_debounce_cache
    )
    _clear_worker_model_gpu_memory = SupervisorActor._clear_worker_model_gpu_memory
    _process_model_gpu_memory_report = SupervisorActor._process_model_gpu_memory_report
    _expire_worker_model_gpu_memory = SupervisorActor._expire_worker_model_gpu_memory
    _is_valid_model_gpu_memory = staticmethod(
        SupervisorActor._is_valid_model_gpu_memory
    )

    def __init__(self, address: str, workers: Dict[str, Any]):
        self.address = address
        self._worker_address_to_worker = dict(workers)
        self._model_uid_to_replica_info: Dict[str, ReplicaInfo] = {}
        self._list_models_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._list_models_result_cache: Dict[str, Dict[str, Any]] = {}
        self._list_models_result_cache_time: float = 0.0
        self._list_models_cache_version: int = 0
        self._list_models_sweep_lock = asyncio.Lock()
        self._replica_gpu_cache: Dict[str, list] = {}
        # worker_address -> replica_model_uid -> {gpu_idx -> bytes}
        self._worker_model_gpu_memory: Dict[str, Dict[str, Dict[int, int]]] = {}
        self._worker_model_gpu_memory_update_time: Dict[str, float] = {}


def _replica_info(replica: int) -> ReplicaInfo:
    active = list(range(replica))
    return ReplicaInfo(
        replica=replica,
        scheduler=itertools.cycle(active),
        active_replica_ids=active,
    )


@pytest.mark.asyncio
async def test_get_replica_statuses_returns_per_replica_runtime_info():
    model_uid = "qwen3"
    replica_uids = [
        build_replica_model_uid(model_uid, 0),
        build_replica_model_uid(model_uid, 1),
    ]

    class _ReplicaWorker(_DummyWorker):
        def __init__(self, address: str, models: Dict[str, Dict[str, Any]]):
            super().__init__(models)
            self.address = address

    worker_one = _ReplicaWorker(
        "xinference-worker-4090-1:30001",
        {
            replica_uids[0]: {
                "address": "xinference-worker-4090-1:42135",
                "accelerators": [0],
            }
        },
    )
    worker_two = _ReplicaWorker(
        "xinference-worker-4090-2:30001",
        {
            replica_uids[1]: {
                "address": "xinference-worker-4090-2:42136",
                "accelerators": [1],
            }
        },
    )

    class _StatusGuard:
        async def get_replica_statuses(self, _model_uid: str):
            return [
                SimpleNamespace(
                    replica_id=0,
                    replica_model_uid=replica_uids[0],
                    worker_address=worker_one.address,
                    status="READY",
                    created_ts=1,
                    error_message=None,
                    replica_uid=None,
                    gpu_idx=None,
                ),
                SimpleNamespace(
                    replica_id=1,
                    replica_model_uid=replica_uids[1],
                    worker_address=worker_two.address,
                    status="READY",
                    created_ts=2,
                    error_message=None,
                    replica_uid=None,
                    gpu_idx=None,
                ),
            ]

    supervisor = SimpleNamespace(
        _status_guard_ref=_StatusGuard(),
        _replica_model_uid_to_worker={
            replica_uids[0]: worker_one,
            replica_uids[1]: worker_two,
        },
    )
    supervisor._get_replica_runtime_info = (
        SupervisorActor._get_replica_runtime_info.__get__(supervisor)
    )

    result = await SupervisorActor.get_replica_statuses(supervisor, model_uid)

    assert result[0]["worker_address"] == worker_one.address
    assert result[0]["model_address"] == "xinference-worker-4090-1:42135"
    assert result[0]["accelerators"] == ["0"]
    assert result[1]["worker_address"] == worker_two.address
    assert result[1]["model_address"] == "xinference-worker-4090-2:42136"
    assert result[1]["accelerators"] == ["1"]


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


@pytest.mark.asyncio
async def test_list_models_aggregates_gpu_memory_by_worker_and_replica():
    worker_one_address = "worker-1:30001"
    worker_two_address = "worker-2:30001"
    replica_zero = build_replica_model_uid("qwen3", 0)
    replica_one = build_replica_model_uid("qwen3", 1)
    supervisor = DummySupervisor(
        worker_one_address,
        {
            worker_one_address: _DummyWorker(
                {
                    replica_zero: {
                        "model_name": "qwen3",
                        "model_type": "LLM",
                    },
                    replica_one: {
                        "model_name": "qwen3",
                        "model_type": "LLM",
                    },
                }
            ),
            worker_two_address: _DummyWorker({}),
        },
    )
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(2)
    now = time_module.time()
    supervisor._worker_model_gpu_memory = {
        worker_one_address: {
            replica_zero: {0: 100},
            replica_one: {0: 50, 1: 200},
        },
        # Same local GPU index belongs to a different physical worker and must
        # stay under that worker's address instead of being merged.
        worker_two_address: {replica_zero: {0: 300}},
    }
    supervisor._worker_model_gpu_memory_update_time = {
        worker_one_address: now,
        worker_two_address: now,
    }

    result = await supervisor.list_models()

    assert result["qwen3"]["gpu_memory"] == {
        worker_one_address: {"0": 150, "1": 200},
        worker_two_address: {"0": 300},
    }


def test_model_gpu_memory_report_three_state_protocol():
    address = "worker-1:30001"
    supervisor = DummySupervisor(address, {})
    initial = {"qwen3-rep0": {0: 100}}

    status = {"model_gpu_memory": initial.copy(), "cpu": "ok"}
    supervisor._process_model_gpu_memory_report(address, status)
    first_timestamp = supervisor._worker_model_gpu_memory_update_time[address]
    assert supervisor._worker_model_gpu_memory[address] == initial
    assert "model_gpu_memory" not in status

    # Field absence represents a failed/skipped collection and preserves cache.
    supervisor._process_model_gpu_memory_report(address, {"cpu": "ok"})
    assert supervisor._worker_model_gpu_memory[address] == initial
    assert supervisor._worker_model_gpu_memory_update_time[address] == first_timestamp

    # A malformed RPC payload is also treated as a failed sample.
    supervisor._process_model_gpu_memory_report(address, None)  # type: ignore[arg-type]
    assert supervisor._worker_model_gpu_memory[address] == initial
    assert supervisor._worker_model_gpu_memory_update_time[address] == first_timestamp

    # Invalid structures are also treated as failed telemetry, not as clear.
    invalid_status = {"model_gpu_memory": {"qwen3-rep0": {"bad": 1}}}
    supervisor._process_model_gpu_memory_report(address, invalid_status)
    assert supervisor._worker_model_gpu_memory[address] == initial
    assert "model_gpu_memory" not in invalid_status

    # Explicit empty is a successful observation with no owned GPU process.
    supervisor._process_model_gpu_memory_report(address, {"model_gpu_memory": {}})
    assert address not in supervisor._worker_model_gpu_memory
    assert address not in supervisor._worker_model_gpu_memory_update_time


def test_clear_replica_gpu_memory_preserves_other_models_on_worker():
    address = "worker-1:30001"
    supervisor = DummySupervisor(address, {})
    supervisor._clear_replica_model_gpu_memory = (
        SupervisorActor._clear_replica_model_gpu_memory.__get__(supervisor)
    )
    supervisor._worker_model_gpu_memory = {
        address: {
            "qwen3-rep0": {0: 100},
            "llama3-rep0": {1: 200},
        }
    }
    supervisor._worker_model_gpu_memory_update_time = {address: time_module.time()}

    assert supervisor._clear_replica_model_gpu_memory("qwen3-rep0") is True
    assert supervisor._worker_model_gpu_memory == {address: {"llama3-rep0": {1: 200}}}
    assert address in supervisor._worker_model_gpu_memory_update_time


@pytest.mark.asyncio
async def test_list_models_expires_stale_gpu_memory(monkeypatch):
    address = "worker-1:30001"
    replica_uid = build_replica_model_uid("qwen3", 0)
    supervisor = DummySupervisor(
        address,
        {
            address: _DummyWorker(
                {
                    replica_uid: {
                        "model_name": "qwen3",
                        "model_type": "LLM",
                    }
                }
            )
        },
    )
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(1)
    supervisor._worker_model_gpu_memory = {address: {replica_uid: {0: 100}}}
    supervisor._worker_model_gpu_memory_update_time = {address: time_module.time() - 91}
    monkeypatch.setattr(
        "xinference.core.supervisor.XINFERENCE_MODEL_GPU_MEMORY_CACHE_TTL", 90
    )

    result = await supervisor.list_models()

    assert "gpu_memory" not in result["qwen3"]
    assert address not in supervisor._worker_model_gpu_memory
    assert address not in supervisor._worker_model_gpu_memory_update_time


@pytest.mark.asyncio
async def test_list_models_fallback_does_not_restore_expired_gpu_memory(monkeypatch):
    address = "worker-1:30001"
    replica_uid = build_replica_model_uid("qwen3", 0)
    worker = _DummyWorker(
        {
            replica_uid: {
                "model_name": "qwen3",
                "model_type": "LLM",
            }
        }
    )
    supervisor = DummySupervisor(address, {address: worker})
    supervisor._model_uid_to_replica_info["qwen3"] = _replica_info(1)
    supervisor._worker_model_gpu_memory = {address: {replica_uid: {0: 100}}}
    supervisor._worker_model_gpu_memory_update_time = {address: time_module.time()}
    monkeypatch.setattr(
        "xinference.core.supervisor.XINFERENCE_MODEL_GPU_MEMORY_CACHE_TTL", 90
    )

    first = await supervisor.list_models()

    assert "gpu_memory" in first["qwen3"]
    # The per-worker fallback cache must contain only the raw worker response.
    assert "replica" not in supervisor._list_models_cache[address][replica_uid]
    assert "gpu_memory" not in supervisor._list_models_cache[address][replica_uid]

    supervisor._worker_address_to_worker[address] = _FailingWorker(RuntimeError("boom"))
    supervisor._worker_model_gpu_memory_update_time[address] = time_module.time() - 91

    second = await supervisor.list_models()

    assert address not in supervisor._worker_model_gpu_memory
    assert "gpu_memory" not in second["qwen3"]
    assert second["qwen3"]["replica"] == 1


@pytest.mark.asyncio
async def test_list_models_debounce_cache_hit():
    """When called within the debounce window, list_models must return the
    cached whole-result without issuing any worker RPCs."""
    address = "127.0.0.1:1234"
    # Worker returns a model NOT in the cache — if the fast path is taken
    # we get the cached value, not what the worker would return.
    worker = _DummyWorker(
        {
            build_replica_model_uid("fresh-model", 0): {
                "model_name": "fresh-model",
                "model_type": "LLM",
            }
        }
    )
    supervisor = DummySupervisor(address, {address: worker})
    supervisor._model_uid_to_replica_info["fresh-model"] = _replica_info(1)
    # Pre-populate the debounce cache with a different model.
    supervisor._list_models_result_cache = {
        "cached-model": {
            "model_name": "cached-model",
            "model_type": "LLM",
            "replica": 1,
        }
    }
    supervisor._list_models_result_cache_time = time_module.time()

    result = await supervisor.list_models()

    # Must return the cached result (not the fresh worker result).
    assert set(result) == {"cached-model"}
    assert result["cached-model"]["model_name"] == "cached-model"


@pytest.mark.asyncio
async def test_list_models_debounce_cache_expiry():
    """When the debounce TTL has elapsed, list_models must perform a fresh
    RPC sweep and update the cache."""
    address = "127.0.0.1:1234"
    worker = _DummyWorker(
        {
            build_replica_model_uid("fresh-model", 0): {
                "model_name": "fresh-model",
                "model_type": "LLM",
            }
        }
    )
    supervisor = DummySupervisor(address, {address: worker})
    supervisor._model_uid_to_replica_info["fresh-model"] = _replica_info(1)
    # Pre-populate with stale cached data (well beyond the debounce TTL).
    supervisor._list_models_result_cache = {
        "stale-model": {
            "model_name": "stale-model",
            "model_type": "LLM",
            "replica": 1,
        }
    }
    supervisor._list_models_result_cache_time = time_module.time() - 3600

    result = await supervisor.list_models()

    # Cache is expired → must return the fresh worker result.
    assert set(result) == {"fresh-model"}
    assert result["fresh-model"]["model_name"] == "fresh-model"
    # The debounce cache must have been refreshed.
    assert supervisor._list_models_result_cache == result
    assert supervisor._list_models_result_cache_time > time_module.time() - 10


class _CountingWorker:
    """Worker that counts list_models() calls for single-flight verification."""

    def __init__(self, models: Dict[str, Dict[str, Any]]):
        self._models = models
        self.list_models_call_count = 0

    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        self.list_models_call_count += 1
        return dict(self._models)


@pytest.mark.asyncio
async def test_list_models_debounce_cache_empty_result():
    """An empty model list ({}) must also be cached so subsequent calls
    within the TTL return the cached empty dict without a new RPC sweep."""
    address = "127.0.0.1:1234"
    worker = _DummyWorker({})
    supervisor = DummySupervisor(address, {address: worker})

    # First call: fills the cache with an empty dict.
    result1 = await supervisor.list_models()
    assert result1 == {}
    assert supervisor._list_models_result_cache_time > 0

    # Second call within TTL: must return the cached empty dict.
    result2 = await supervisor.list_models()
    assert result2 == {}


@pytest.mark.asyncio
async def test_list_models_debounce_single_flight():
    """When the debounce cache is empty, concurrent list_models calls must
    coalesce into a single RPC sweep (single-flight) rather than each
    starting its own full worker sweep."""
    address = "127.0.0.1:1234"
    worker = _CountingWorker(
        {
            build_replica_model_uid("demo-model", 0): {
                "model_name": "demo-model",
                "model_type": "LLM",
            }
        }
    )
    supervisor = DummySupervisor(address, {address: worker})
    supervisor._model_uid_to_replica_info["demo-model"] = _replica_info(1)

    # Launch two concurrent list_models calls.
    results = await asyncio.gather(
        supervisor.list_models(),
        supervisor.list_models(),
    )

    # Both must return the same result.
    for r in results:
        assert set(r) == {"demo-model"}
    # The worker must have been called exactly once (single-flight).
    assert worker.list_models_call_count == 1
