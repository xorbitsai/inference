# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Keep one public actor address while replacing its serving process.

An inference request, including its iterator, belongs to exactly one generation.
The routing decision never waits for engine construction or request draining.
"""

import asyncio
import copy
import importlib
import inspect
import json
import os
import sys
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import xoscar as xo

# These messages operate on the stable address, rather than on the serving
# generation. In particular, iterator messages and serving-slot releases from
# streams accepted before cutover must still reach their original actor.
LOCAL_METHODS = {
    "abort_request",
    "get_worker_binding",
    "rebind_worker",
    "rebind_model_generations",
    "get_reload_status",
    "begin_generation_reload",
    "commit_model_generation",
    "model_generation_protocol",
    "create_generation_cache",
    "discard_generation_cache",
    "generation_serve_count",
    "retire_model_generation",
    "decrease_serve_count",
    "get_pid",
    "get_pool_addresses",
    "stop",
    "close_model_generations",
    "fail_generation_reload",
    "handle_generation_pool_exit",
}


def _generation_key(ref):
    return "initial" if ref is None else ref.uid


def _check_generation_capacity(model, caches, config, reused):
    """Leave the serving engine its configured budget before starting a peer.

    This deliberately refuses unknown driver ownership and tight budgets. It
    never allocates a reservation buffer that could itself starve live requests.
    """
    import psutil

    from ..model.llm.weight_cache import _gpu_memory_snapshot

    def tree(pid):
        return {
            pid,
            *(child.pid for child in psutil.Process(pid).children(recursive=True)),
        }

    cache = model._weight_cache
    cache_pids = tree(cache.process.pid)
    engine_pids = tree(os.getpid()) - cache_pids
    for owned in caches:
        if owned.process is not None:
            engine_pids -= tree(owned.process.pid)
    original = model._reload_config
    engine = model._weight_cache_engine
    fraction_field = (
        "gpu_memory_utilization" if engine == "vllm" else "mem_fraction_static"
    )
    default_fraction = 0.9
    margin = 1024**3
    checked = 0
    for index, (total, free, processes) in _gpu_memory_snapshot().items():
        owners = cache_pids | engine_pids
        if not cache_pids.intersection(processes):
            continue
        if not engine_pids.intersection(processes) or any(
            processes[pid] is None for pid in owners.intersection(processes)
        ):
            raise ValueError(
                "Cannot safely account for the serving engine's GPU memory"
            )
        weights = sum(processes[pid] for pid in cache_pids.intersection(processes))
        current = weights + sum(
            processes[pid] for pid in engine_pids.intersection(processes)
        )

        def budget(settings):
            fraction = float(settings.get(fraction_field) or default_fraction)
            # Explicit KV allocations can exceed the fractional budget. Leave
            # space for retained weights plus execution/context allocations.
            kv_bytes = (
                int(settings.get("kv_cache_memory_bytes") or 0)
                if engine == "vllm"
                else 0
            )
            return max(int(total * fraction), kv_bytes + weights + margin)

        reserve = max(0, budget(original) - current)
        candidate = budget(config)
        if reused:
            candidate -= weights
        else:
            # Preload can temporarily hold another copy while transferring
            # weights. Keep that allowance outside the old engine's budget.
            candidate = max(candidate, 2 * weights)
        required = candidate + reserve + margin
        if free < required:
            raise ValueError(
                f"Insufficient GPU memory for uninterrupted replacement on GPU {index}: "
                f"{free / 1024**3:.2f} GiB free, at least {required / 1024**3:.2f} GiB required "
                "while preserving the serving engine's memory budget. "
                "Use a smaller target memory budget or a GPU with more free memory."
            )
        checked += 1
    ranks = int(
        cache.config.get("tensor_parallel_size", cache.config.get("tp_size", 1))
    )
    if checked != ranks:
        raise ValueError(
            "Cannot safely account for every serving GPU. Expose driver process "
            "memory accounting before replacing the model; the current engine remains serving."
        )


class ModelGenerationMixin:
    if TYPE_CHECKING:
        _serve_count: int
        _n_worker: Optional[int]
        _request_limits: float

        def _get_weight_cached_model(self) -> Any: ...

        def _require_ready(self) -> None: ...

        async def validate_reload(self, config: Dict[str, Any]) -> Dict[str, Any]: ...

    async def close_model_generations(self):
        tasks = tuple(self.__dict__.get("_generation_retirement_tasks", ()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for ref in [
            self.__dict__.get("_serving_generation"),
            *self.__dict__.get("_retiring_generations", []),
        ]:
            if ref is not None:
                try:
                    await ref.stop()
                except Exception:
                    pass

    def model_generation_protocol(self) -> int:
        return 1

    async def __on_receive__(self, message):
        method, call_method, args, kwargs = message
        target = self.__dict__.get("_serving_generation")
        if (
            target is not None
            and not method.startswith("_")
            and method not in LOCAL_METHODS
        ):
            if call_method != 0:
                raise ValueError(
                    "Batched actor calls are unsupported during model replacement"
                )
            return await getattr(target, method)(*args, **kwargs)
        return await super().__on_receive__(message)

    def begin_generation_reload(self, operation_id: str) -> None:
        if self.__dict__.get("_retiring_generations"):
            raise RuntimeError(
                "The previous engine is still draining requests; retry the upgrade after they finish"
            )
        self._reload_status = {
            "operation_id": operation_id,
            "status": "reloading",
            "stage": "loading",
            "weights_reused": False,
        }

    def fail_generation_reload(self, operation_id: str, error: str):
        status = self.__dict__.get("_reload_status", {})
        if (
            status.get("operation_id") == operation_id
            and status.get("status") != "ready"
        ):
            status.update(status="error", stage="error", error=error, restored=True)

    def generation_serve_count(self) -> int:
        return self._serve_count

    async def export_model_generation(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        from importlib.metadata import version

        model = self._get_weight_cached_model()
        self._require_ready()
        if patch:
            await self.validate_reload(patch)
        if not model._enable_weight_cache or model._weight_cache is None:
            raise ValueError(
                "Seamless engine replacement requires enable_weight_cache=true"
            )
        if self._n_worker != 1:
            raise ValueError("Seamless engine replacement requires one worker")
        config = copy.deepcopy(model._reload_config)
        config.update(copy.deepcopy(patch))
        return {
            "module": type(model).__module__,
            "class": type(model).__name__,
            "model_uid": model.model_uid,
            "model_family": model.model_family.dict(),
            "model_path": model.model_path,
            "config": config,
            "engine": model._weight_cache_engine,
            "engine_version": version(model._weight_cache_engine),
            "torch_version": version("torch"),
            "cache": model._weight_cache.export(),
            "python_executable": sys.executable,
            "request_limits": (
                None if self._request_limits == float("inf") else self._request_limits
            ),
            "env": {
                key: os.environ[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                )
                if key in os.environ
            },
        }

    async def create_generation_cache(self, snapshot, python_executable, environment):
        from ..model.llm.weight_cache import (
            HOT_RELOAD_FIELDS,
            RELOAD_FIELDS,
            WeightCacheDaemon,
        )
        from .lifecycle import finish_before_cancelling

        # Start from the daemon's engine-only config. The wrapper snapshot also
        # contains Xinference options (for example enable_thinking) that an
        # upstream preload CLI must never receive.
        config = copy.deepcopy(snapshot["cache"]["config"])
        config.update(
            {
                key: value
                for key, value in snapshot["config"].items()
                if key in RELOAD_FIELDS[snapshot["engine"]]
                and key not in HOT_RELOAD_FIELDS
            }
        )
        cache = WeightCacheDaemon(snapshot["engine"], snapshot["model_path"], config)
        try:
            await finish_before_cancelling(
                asyncio.to_thread(
                    cache.start,
                    python_executable=python_executable,
                    environment=environment,
                )
            )
        except BaseException:
            await asyncio.to_thread(cache.stop)
            raise
        caches = self.__dict__.setdefault("_generation_caches", {})
        caches[str(cache.directory)] = cache
        return cache.export()

    async def check_generation_capacity(self, config, reused):
        await asyncio.to_thread(
            _check_generation_capacity,
            self._get_weight_cached_model(),
            tuple(self.__dict__.get("_generation_caches", {}).values()),
            config,
            reused,
        )

    async def discard_generation_cache(self, directory):
        cache = self.__dict__.get("_generation_caches", {}).pop(directory, None)
        if cache is not None:
            await asyncio.to_thread(cache.stop)

    async def commit_model_generation(
        self, target, operation_id, weights_reused, cache_directory
    ):
        if (
            self.__dict__.get("_reload_status", {}).get("operation_id") == operation_id
            and self.__dict__.get("_serving_generation") == target
        ):
            return True
        await target.generation_ready()
        # No await between choosing the old generation and publishing the new
        # one. Requests already admitted retain their original actor/iterator.
        previous = self.__dict__.get("_serving_generation")
        leases = self.__dict__.setdefault("_generation_cache_leases", {})
        if previous is None:
            leases["initial"] = str(self._model._weight_cache.directory)
        leases[_generation_key(target)] = cache_directory
        self.__dict__.setdefault("_generation_operation_ids", {})[
            _generation_key(target)
        ] = operation_id
        self._serving_generation = target
        self.__dict__.setdefault("_retiring_generations", []).append(previous)
        self._reload_status = {
            "operation_id": operation_id,
            "status": "ready",
            "stage": "ready",
            "weights_reused": weights_reused,
        }
        self._start_generation_retirement(previous)
        return True

    def _start_generation_retirement(self, previous):
        key = _generation_key(previous)
        task = asyncio.create_task(self.retire_model_generation(previous))
        tasks = self.__dict__.setdefault("_generation_retirement_tasks", set())
        self.__dict__.setdefault("_generation_retire_tasks", {})[key] = task
        tasks.add(task)
        task.add_done_callback(tasks.discard)

        def completed(done):
            retire_tasks = self.__dict__.get("_generation_retire_tasks", {})
            if retire_tasks.get(key) is done:
                retire_tasks.pop(key)

        task.add_done_callback(completed)

    async def handle_generation_pool_exit(self, address):
        """A failed candidate must not trigger recovery of a healthy engine."""
        failed = self.__dict__.setdefault("_failed_generation_addresses", set())
        current = self.__dict__.get("_serving_generation")
        retiring = self.__dict__.get("_retiring_generations", [])
        tasks = self.__dict__.get("_generation_retire_tasks", {})
        started = self.__dict__.get("_generation_retirement_started", set())
        if current is not None and current.address == address:
            failed.add(_generation_key(current))
            for previous in reversed(retiring):
                key = _generation_key(previous)
                if key in started or key in failed:
                    continue
                count = (
                    self._serve_count
                    if previous is None
                    else await previous.generation_serve_count()
                )
                if count and key not in started:
                    if key in tasks:
                        tasks[key].cancel()
                    retiring.remove(previous)
                    self._serving_generation = previous
                    retiring.append(current)
                    self._start_generation_retirement(current)
                    serving_operation = self.__dict__.get(
                        "_generation_operation_ids", {}
                    ).get(_generation_key(current))
                    if self._reload_status.get("operation_id") == serving_operation:
                        self._reload_status.update(
                            status="error",
                            stage="error",
                            restored=True,
                            error="Replacement engine exited; the previous engine is still serving",
                        )
                    return "restored"
            # No previous serving requests can be preserved. The normal Worker
            # recovery path recreates the model with its committed environment.
            return "recover"
        for previous in retiring:
            if previous is not None and previous.address == address:
                key = _generation_key(previous)
                failed.add(key)
                if key in tasks:
                    tasks[key].cancel()
                self._start_generation_retirement(previous)
                return "retiring"
        return "unpublished"

    async def retire_model_generation(self, previous):
        retiring = self.__dict__.get("_retiring_generations", [])
        key = _generation_key(previous)
        try:
            while True:
                if key in self.__dict__.get("_failed_generation_addresses", ()):
                    break
                count = (
                    self._serve_count
                    if previous is None
                    else await previous.generation_serve_count()
                )
                if not count:
                    break
                await asyncio.sleep(0.05)
            self.__dict__.setdefault("_generation_retirement_started", set()).add(key)
            if key not in self.__dict__.get("_failed_generation_addresses", ()):
                if previous is None:
                    await asyncio.to_thread(self._model._stop_engine)
                else:
                    await previous.retire_generation_engine()
            directory = self.__dict__.get("_generation_cache_leases", {}).pop(key, None)
            if (
                directory
                and directory
                not in self.__dict__.get("_generation_cache_leases", {}).values()
            ):
                cache = self.__dict__.get("_generation_caches", {}).pop(directory, None)
                if (
                    cache is None
                    and self._model._weight_cache is not None
                    and str(self._model._weight_cache.directory) == directory
                ):
                    cache, self._model._weight_cache = self._model._weight_cache, None
                if cache is not None:
                    await asyncio.to_thread(cache.stop)
            if previous is not None:
                while True:
                    try:
                        worker = await self._get_worker_ref()
                        await worker.retire_model_generation_pool(
                            self._replica_model_uid, previous.address
                        )
                        break
                    except Exception:
                        self._worker_ref = None
                        await asyncio.sleep(1)
            if previous in retiring:
                retiring.remove(previous)
            self.__dict__.get("_generation_retirement_started", set()).discard(key)
            self.__dict__.get("_failed_generation_addresses", set()).discard(key)
            self.__dict__.get("_generation_operation_ids", {}).pop(key, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Cleanup failure cannot revoke a serving generation. The owner
            # retains the pool for explicit termination and later retry.
            import logging

            logging.getLogger(__name__).exception("Failed to retire model generation")

    async def generation_ready(self):
        self._require_ready()
        model = self._get_weight_cached_model()
        model._weight_cache.client_config()
        engine = getattr(model, "_engine", None)
        check = getattr(engine, "check_health", None) or getattr(
            engine, "get_server_info", None
        )
        if check is not None:
            if inspect.iscoroutinefunction(check):
                await check()
            else:
                await asyncio.to_thread(check)
        return True

    async def retire_generation_engine(self):
        await asyncio.to_thread(self._model._stop_engine)

    async def rebind_model_generations(self, expected_address, address, generation):
        refs = [
            self.__dict__.get("_serving_generation"),
            *self.__dict__.get("_retiring_generations", []),
        ]
        for ref in refs:
            if ref is not None and _generation_key(ref) not in self.__dict__.get(
                "_failed_generation_addresses", ()
            ):
                await ref.rebind_worker(expected_address, address, generation)


def build_generation_model(snapshot):
    from ..model.llm.llm_family import LLMFamilyV2
    from ..model.llm.weight_cache import BorrowedWeightCache

    engine = snapshot["engine"]
    if snapshot["module"] != f"xinference.model.llm.{engine}.core" or engine not in {
        "vllm",
        "sglang",
    }:
        raise ValueError("Unsupported model generation class")
    cls = getattr(importlib.import_module(snapshot["module"]), snapshot["class"])
    family = LLMFamilyV2.parse_obj(snapshot["model_family"])
    model = cls(
        snapshot["model_uid"],
        family,
        snapshot["model_path"],
        {**snapshot["config"], "enable_weight_cache": True},
    )
    model._weight_cache = BorrowedWeightCache(snapshot["cache"])
    return model


async def probe_environment(executable: str, engine: str, env: Dict[str, str]):
    script = (
        "import importlib.metadata as m, json; "
        f"print(json.dumps({{p:m.version(p) for p in ['xinference','xoscar','torch',{engine!r}]}}))"
    )
    process = await asyncio.create_subprocess_exec(
        executable,
        "-c",
        script,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise ValueError(
            "Replacement environment is unavailable: "
            + stderr.decode(errors="replace")[-2000:]
        )
    return json.loads(stdout)


async def replace_model_generation(
    worker, model_uid, patch, python_executable=None, operation_id=None
):
    """Prepare privately, publish once, and never stop a serving engine first."""
    from .model import ModelGenerationActor
    from .worker_relaunch import validate_relaunch_options

    validate_relaunch_options(python_executable)
    anchor = worker._model_uid_to_model[model_uid]
    if operation_id is not None:
        current = await anchor.get_reload_status()
        if (
            current.get("operation_id") == operation_id
            and current.get("status") == "ready"
        ):
            return current.get("weights_reused", False)
    try:
        if await anchor.model_generation_protocol() != 1:
            raise ValueError("Model does not support seamless replacement")
    except AttributeError as error:
        raise ValueError(
            "The model was started without seamless replacement support; restart it once with a supporting Xinference version"
        ) from error
    snapshot = await anchor.export_model_generation(patch)
    operation_id = operation_id or uuid.uuid4().hex
    await anchor.begin_generation_reload(operation_id)
    env = dict(os.environ)
    env.update(worker._model_uid_to_launch_args[model_uid].get("envs") or {})
    env.update(snapshot["env"])
    executable = python_executable or snapshot["python_executable"]
    if python_executable is not None:
        from .utils import build_subpool_envs_for_python

        env = build_subpool_envs_for_python(env, executable)
    else:
        env["PATH"] = os.pathsep.join(
            [os.path.dirname(executable), env.get("PATH", "")]
        )
    cache_directory: Optional[str] = None
    address: Optional[str] = None
    committed = False
    try:
        versions = await probe_environment(executable, snapshot["engine"], env)
        reused = (
            versions[snapshot["engine"]] == snapshot["engine_version"]
            and versions["torch"] == snapshot["torch_version"]
        )
        await anchor.check_generation_capacity(snapshot["config"], reused)
        if not reused:
            # Upstream checks include engine/kernel versions. A different
            # environment gets its own cache; never relax the IPC fingerprint.
            snapshot["cache"] = await anchor.create_generation_cache(
                snapshot, executable, env
            )
            cache_directory = snapshot["cache"]["directory"]
        cache = snapshot["cache"]
        if snapshot["engine"] == "sglang":
            env["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"] = os.path.join(
                cache["directory"], "{device_uuid}.sock"
            )
            env["SGLANG_WEIGHT_CACHE_READY_TEMPLATE"] = os.path.join(
                cache["directory"], "{device_uuid}.ready"
            )
        # Device ownership remains with the public model. In particular, this
        # path never invokes orphan reclamation or releases its reservations.
        address = await worker._append_sub_pool_protected(
            env=env, start_python=executable, model_uid=model_uid
        )
        worker._model_uid_to_subpool_addresses.setdefault(model_uid, set()).add(address)
        candidate = await xo.create_actor(
            ModelGenerationActor,
            address=address,
            uid=f"{model_uid}-generation-{operation_id}",
            snapshot=snapshot,
            supervisor_address=worker._supervisor_address,
            worker_address=worker.address,
        )
        await candidate.load()
        await candidate.wait_for_load()
        if worker._model_uid_to_model.get(model_uid) is not anchor:
            raise RuntimeError("Model ownership changed before replacement")
        # Commit is idempotent at the stable actor. Cancellation of the caller
        # must not tear down an engine whose route has already been published.
        from .lifecycle import finish_before_cancelling

        await finish_before_cancelling(
            anchor.commit_model_generation(
                candidate, operation_id, reused, cache["directory"]
            )
        )
        committed = True
        await worker._ensure_subpool_monitor()
        return reused
    except BaseException as exc:
        try:
            await anchor.fail_generation_reload(operation_id, str(exc))
        except Exception:
            pass
        raise
    finally:
        if not committed:
            status = await anchor.get_reload_status()
            committed = (
                status.get("operation_id") == operation_id
                and status.get("status") == "ready"
            )
        if not committed:
            if address is not None:
                await worker._main_pool.remove_sub_pool(address, force=True)
                worker._model_uid_to_subpool_addresses.get(model_uid, set()).discard(
                    address
                )
            if cache_directory is not None:
                await anchor.discard_generation_cache(cache_directory)
