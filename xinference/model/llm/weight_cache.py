# Copyright 2026 Xorbits Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Engine-owned clients of a persistent, upstream GPU weight cache daemon."""

import copy
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional

from packaging.version import Version

logger = logging.getLogger(__name__)


def _cached_memory_fraction(pid: int) -> float:
    """Largest per-device reservation owned by the daemon, including its ranks.

    Query the driver without creating a CUDA context in the model actor. Engine
    processes are siblings and must not count towards the retained reservation.
    """
    import psutil
    import torch

    pids = {pid, *(child.pid for child in psutil.Process(pid).children(recursive=True))}
    fractions = []
    if torch.version.hip:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            for handle in amdsmi.amdsmi_get_processor_handles():
                used = 0
                for process in amdsmi.amdsmi_get_gpu_process_list(handle):
                    # Older AMD SMI returns process handles instead of records.
                    if not isinstance(process, dict):
                        process = amdsmi.amdsmi_get_gpu_process_info(handle, process)
                    if process["pid"] in pids:
                        used += process["memory_usage"]["vram_mem"]
                if used:
                    total = amdsmi.amdsmi_get_gpu_memory_total(
                        handle, amdsmi.AmdSmiMemoryType.VRAM
                    )
                    fractions.append(used / total)
        finally:
            amdsmi.amdsmi_shut_down()
    else:
        import pynvml

        pynvml.nvmlInit()
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                used = sum(
                    process.usedGpuMemory
                    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    if process.pid in pids
                )
                if used:
                    try:
                        memory = pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
                    except (pynvml.NVMLError, AttributeError):
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    fractions.append(used / memory.total)
        finally:
            pynvml.nvmlShutdown()
    if not fractions or not all(0 < value < 1 for value in fractions):
        raise RuntimeError("Cannot account for the weight cache daemon's GPU memory")
    return max(fractions)


# Only execution parameters may change while the daemon owns the weights.
# In particular, do not accept arbitrary engine arguments: some change packed
# weight layouts without changing a model name (attention/MoE backends, RoPE).
RELOAD_FIELDS = {
    "vllm": {
        "max_num_seqs": "positive_int",
        "max_num_batched_tokens": "positive_int",
        "max_model_len": "positive_int",
        "gpu_memory_utilization": "fraction",
        "enforce_eager": "bool",
    },
    "sglang": {
        "max_running_requests": "positive_int",
        "max_prefill_tokens": "positive_int",
        "context_length": "positive_int",
        "chunked_prefill_size": "chunk_size",
        "mem_fraction_static": "fraction",
        "disable_cuda_graph": "bool",
        "cuda_graph_max_bs_decode": "positive_int",
        "cuda_graph_max_bs_prefill": "positive_int",
        "schedule_conservativeness": "positive_number",
    },
}


class ModelReloadError(RuntimeError):
    def __init__(self, message: str, restored: bool = False):
        super().__init__(message)
        self.restored = restored


def parse_weight_cache_option(value: Any) -> bool:
    # CLI extra arguments arrive as strings.
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError("enable_weight_cache must be a boolean")


def validate_reload_patch(engine: str, patch: Dict[str, Any]) -> None:
    if not isinstance(patch, dict) or not patch:
        raise ValueError("model_config must be a non-empty object")
    fields = RELOAD_FIELDS[engine]
    for key, value in patch.items():
        kind = fields.get(key)
        if kind is None:
            raise ValueError(
                f"{key} cannot be changed while reusing weights. "
                f"Supported parameters: {', '.join(fields)}"
            )
        if kind == "bool":
            valid = isinstance(value, bool)
        elif kind in ("positive_int", "chunk_size"):
            valid = type(value) is int and (
                value > 0 or (kind == "chunk_size" and value == -1)
            )
        else:
            valid = (
                type(value) in (int, float)
                and math.isfinite(value)
                and value > 0
                and (kind != "fraction" or value <= 1)
            )
        if not valid:
            raise ValueError(f"Invalid value for {key}: expected {kind}")


class WeightCacheDaemon:
    """Keep GPU weights alive across engine restarts within one ModelActor.

    The actor and its GPU reservation outlive reload. Full model stop releases
    the engine's IPC mappings before terminating the daemon. A parent watcher
    also releases the daemon group if the actor process dies unexpectedly.
    """

    def __init__(self, engine: str, model_path: str, config: Dict[str, Any]):
        self.engine = engine
        self.config = copy.deepcopy(config)
        self.model_path = model_path
        self._directory = tempfile.TemporaryDirectory(prefix="xinfw-")
        # Linux AF_UNIX paths allow 107 bytes plus the terminator. Reserve room
        # for vLLM's socket name, including a possible 64-character UUID hash.
        if len(os.fsencode(self._directory.name)) > 20:
            self._directory.cleanup()
            self._directory = tempfile.TemporaryDirectory(prefix="xinfw-", dir="/tmp")
        self.directory = Path(self._directory.name)
        self.process: Optional[subprocess.Popen] = None
        self._log: Optional[IO[str]] = None
        self._saved_env: Dict[str, Optional[str]] = {}
        self._socket_paths: Dict[Path, int] = {}
        self._ready_paths: list[Path] = []

    def start(self, timeout: float = 1800) -> None:
        config = {k: v for k, v in self.config.items() if v is not None}
        config.pop("launch_timeout", None)
        env = os.environ.copy()
        if self.engine == "vllm":
            config["model"] = self.model_path
            config["weight-cache-socket-dir"] = str(self.directory)
            command = ["-m", "vllm.entrypoints.cli.main", "preload"]
            ranks = int(config.get("tensor_parallel_size", 1))
        else:
            config["model_path"] = self.model_path
            config["weight_cache_mode"] = "off"
            # SGLang accepts this legacy Python argument, but its YAML parser
            # rejects the deprecated CLI action. Use the current CLI fields.
            if config.pop("disable_cuda_graph", False):
                config["cuda_graph_backend_decode"] = "disabled"
                config["cuda_graph_backend_prefill"] = "disabled"
            command = [
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--timeout",
                str(timeout),
            ]
            ranks = int(config.get("tp_size", 1))
            for name, suffix in (
                ("SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE", ".sock"),
                ("SGLANG_WEIGHT_CACHE_READY_TEMPLATE", ".ready"),
            ):
                self._saved_env[name] = os.environ.get(name)
                env[name] = str(self.directory / ("{device_uuid}" + suffix))
                # This subprocess serves one ModelActor. Engine children inherit
                # the same private socket namespace as the standalone daemon.
                os.environ[name] = env[name]
            # SGLang's config merger emits keys directly as CLI flags.
            config = {key.replace("_", "-"): value for key, value in config.items()}
        # JSON is a YAML subset, preserving nested engine configs and booleans.
        config_path = self.directory / "config.yaml"
        try:
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self._log = (self.directory / "daemon.log").open("w+")
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "xinference.model.llm.weight_cache",
                    str(os.getpid()),
                    *command,
                    "--config",
                    str(config_path),
                ],
                env=env,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self._log.seek(0)
                    raise RuntimeError(
                        "Weight cache daemon failed: " + self._log.read()[-8000:]
                    )
                pattern = "*.sock" if self.engine == "vllm" else "*.ready"
                ready = [
                    p for p in self.directory.glob(pattern) if "_draft" not in p.name
                ]
                if len(ready) >= ranks:
                    self._socket_paths = {
                        p: p.stat().st_ino for p in self.directory.glob("*.sock")
                    }
                    if len(self._socket_paths) >= ranks:
                        self._ready_paths = list(self.directory.glob("*.ready"))
                        return
                time.sleep(0.1)
            raise RuntimeError("Timed out waiting for the weight cache daemon")
        except BaseException:
            self.stop()
            raise

    def client_config(self) -> Dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Weight cache daemon is unavailable")
        try:
            if any(p.stat().st_ino != inode for p, inode in self._socket_paths.items()):
                raise RuntimeError("Weight cache sockets changed")
            if self.engine == "sglang":
                for ready in self._ready_paths:
                    pid_line = ready.read_text().splitlines()[0]
                    os.kill(int(pid_line.removeprefix("pid=")), 0)
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError("Weight cache rank is unavailable") from exc
        if self.engine == "vllm":
            extra = dict(
                socket_dir=str(self.directory), mode="zero_copy", fallback=False
            )
            return {"load_format": "ipc_cache", "model_loader_extra_config": extra}
        return {"weight_cache_mode": "client"}

    def engine_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(config)
        if self.engine != "vllm":
            return config
        assert self.process is not None
        requested = config.get("gpu_memory_utilization", 0.9)
        if config.get("kv_cache_memory_bytes"):
            # vLLM ignores this fraction for an explicit KV allocation, but
            # still checks it against free memory before loading IPC weights.
            effective = min(requested, 0.01)
        else:
            effective = requested - _cached_memory_fraction(self.process.pid)
            if effective <= 0:
                raise ValueError(
                    "gpu_memory_utilization is too small to cover retained weights"
                )
        config["gpu_memory_utilization"] = effective
        logger.info(
            "vLLM memory budget: requested fraction=%s, engine fraction=%s; "
            "retained daemon memory is reserved separately",
            requested,
            effective,
        )
        return config

    def stop(self) -> None:
        if self.process is not None:
            # Kill the whole group even if the launcher exited before its ranks.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
            self.process = None
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved_env.clear()
        if self._log is not None:
            self._log.close()
            self._log = None
        self._directory.cleanup()


class WeightCachedModel:
    _enable_weight_cache: bool = False
    _weight_cache: Optional[WeightCacheDaemon] = None
    _weight_cache_engine: str
    _model_config: Any
    _n_worker: int
    model_path: str
    _loading_error: Any
    _loading_thread: Any

    if TYPE_CHECKING:

        def _sanitize_model_config(self, config: Any) -> Any: ...
        def _stop_engine(self) -> None: ...
        def load(self) -> None: ...
        def wait_for_load(self) -> None: ...

    def _init_weight_cache(self, config: Any) -> None:
        self._enable_weight_cache = parse_weight_cache_option(
            config.pop("enable_weight_cache", False)
        )
        self._reload_config = copy.deepcopy(config)
        self._weight_cache: Optional[WeightCacheDaemon] = None

    def _prepare_weight_cache(self) -> None:
        if not self._enable_weight_cache:
            return
        config = self._model_config
        if self._weight_cache is None:
            import torch

            engine = self._weight_cache_engine
            minimum = "0.31.0" if engine == "vllm" else "0.5.21"
            if Version(package_version(engine)) < Version(minimum):
                raise ValueError(f"enable_weight_cache requires {engine}>={minimum}")
            if sys.platform != "linux" or not torch.cuda.is_available():
                raise ValueError("Weight caching requires Linux with CUDA or ROCm")
            if (
                self._n_worker != 1
                or config.get("pipeline_parallel_size", config.get("pp_size", 1)) != 1
                or config.get("data_parallel_size", config.get("dp_size", 1)) != 1
                or config.get("nnodes", 1) != 1
                or getattr(self, "_xavier_config", None) is not None
                or getattr(self, "_nixl_config", None) is not None
                or config.get("kv_transfer_config")
                or config.get("disaggregation_mode", "null") not in ("null", None)
            ):
                raise ValueError(
                    "Weight caching currently supports single-worker tensor parallelism"
                )
            if config.get("enable_sleep_mode") or config.get("cpu_offload_gb", 0):
                raise ValueError(
                    "Weight caching cannot be combined with weight offloading"
                )
            if config.get("load_format") == "ipc_cache" or config.get(
                "weight_cache_mode"
            ):
                raise ValueError(
                    "Xinference manages the weight cache loader automatically"
                )
            if (
                getattr(self, "lora_modules", None)
                or config.get("enable_lora")
                or config.get("speculative_algorithm")
                or config.get("speculative_config")
            ):
                raise ValueError(
                    "Weight caching does not yet support this adapter/draft setup"
                )
            daemon_config = copy.deepcopy(config)
            # This is an engine process routing option, not a weight layout.
            daemon_config.pop("distributed_executor_backend", None)
            daemon = WeightCacheDaemon(engine, self.model_path, daemon_config)
            daemon.start()
            self._weight_cache = daemon
        config.update(self._weight_cache.client_config())

    def get_reload_config(self) -> Dict[str, Any]:
        return {
            "enabled": self._enable_weight_cache,
            "engine": self._weight_cache_engine,
            "model_config": {
                k: v
                for k, v in self._model_config.items()
                if k in RELOAD_FIELDS[self._weight_cache_engine] and v is not None
            },
            "parameters": RELOAD_FIELDS[self._weight_cache_engine],
        }

    def _weight_cache_engine_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self._weight_cache is None:
            return config
        return self._weight_cache.engine_config(config)

    def validate_reload(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        if self._weight_cache is None:
            raise ValueError(
                "Launch this model with enable_weight_cache=true before reloading"
            )
        self._weight_cache.client_config()
        validate_reload_patch(self._weight_cache_engine, patch)
        config = {**copy.deepcopy(self._reload_config), **patch}
        candidate = self._sanitize_model_config(copy.deepcopy(config))
        candidate.pop("reasoning_content", None)
        candidate.pop("enable_thinking", None)
        if self._weight_cache_engine == "vllm":
            from vllm.engine.arg_utils import AsyncEngineArgs

            AsyncEngineArgs(
                model=self.model_path, **self._weight_cache_engine_config(candidate)
            ).create_engine_config()
        else:
            from sglang.srt.server_args import ServerArgs

            candidate.pop("launch_timeout", None)
            ServerArgs(model_path=self.model_path, **candidate)
        return config

    def reload(self, patch: Dict[str, Any], progress: Callable[[str], None]) -> None:
        config = self.validate_reload(patch)
        previous = copy.deepcopy(self._reload_config)

        def load(config: Dict[str, Any]) -> None:
            self._model_config = copy.deepcopy(config)
            self._loading_error = None
            self._loading_thread = None
            self.load()
            self.wait_for_load()
            assert self._weight_cache is not None
            self._weight_cache.client_config()

        progress("loading")
        try:
            self._stop_engine()
            load(config)
        except Exception as exc:
            progress("restoring")
            try:
                self._stop_engine()
                load(previous)
            except Exception as restore_exc:
                raise ModelReloadError(
                    f"Reload failed: {exc}. Restoring previous configuration failed: {restore_exc}",
                    restored=False,
                ) from exc
            raise ModelReloadError(
                f"Reload failed; previous configuration restored: {exc}", restored=True
            ) from exc
        self._reload_config = config


def _run_watched(parent_pid: int, command: list) -> int:
    child = subprocess.Popen([sys.executable, *command])
    try:
        while child.poll() is None:
            if os.getppid() != parent_pid:
                os.killpg(os.getpgrp(), signal.SIGTERM)
            time.sleep(0.2)
        return child.returncode
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait()


if __name__ == "__main__":
    sys.exit(_run_watched(int(sys.argv[1]), sys.argv[2:]))
