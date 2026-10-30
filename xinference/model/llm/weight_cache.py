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
import dataclasses
import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from packaging.version import Version

logger = logging.getLogger(__name__)


def _gpu_process_memory() -> Dict[int, tuple[int, Dict[int, Optional[int]]]]:
    """Driver PID records, without creating a CUDA context in the actor."""
    return {
        index: (total, processes)
        for index, (total, _, processes) in _gpu_memory_snapshot().items()
    }


def _gpu_memory_snapshot():
    """Total/free bytes and process ownership, without allocating GPU memory."""
    import torch

    devices = {}
    if torch.version.hip:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            for index, handle in enumerate(amdsmi.amdsmi_get_processor_handles()):
                processes = {}
                for process in amdsmi.amdsmi_get_gpu_process_list(handle):
                    if not isinstance(process, dict):
                        process = amdsmi.amdsmi_get_gpu_process_info(handle, process)
                    processes[process["pid"]] = process["memory_usage"]["vram_mem"]
                total = amdsmi.amdsmi_get_gpu_memory_total(
                    handle, amdsmi.AmdSmiMemoryType.VRAM
                )
                used = amdsmi.amdsmi_get_gpu_memory_usage(
                    handle, amdsmi.AmdSmiMemoryType.VRAM
                )
                devices[index] = (total, total - used, processes)
        finally:
            amdsmi.amdsmi_shut_down()
    else:
        import pynvml

        pynvml.nvmlInit()
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                processes = {
                    process.pid: process.usedGpuMemory
                    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                }
                try:
                    memory = pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
                except (pynvml.NVMLError, AttributeError):
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                devices[index] = (memory.total, memory.free, processes)
        finally:
            pynvml.nvmlShutdown()
    return devices


def _cached_memory_fraction(
    pid: int, baseline: Optional[Dict[int, Set[int]]] = None
) -> float:
    """Largest per-device reservation owned by the daemon, including its ranks.

    Query the driver without creating a CUDA context in the model actor. Engine
    processes are siblings and must not count towards the retained reservation.
    """
    import psutil

    pids = {pid, *(child.pid for child in psutil.Process(pid).children(recursive=True))}
    fractions = []
    for index, (total, processes) in _gpu_process_memory().items():
        owners = pids
        if baseline is not None:
            # Driver PIDs may belong to the host namespace. Snapshot before
            # preload, then count new driver PIDs before any engine is started.
            # Concurrent new processes conservatively increase the reservation.
            owners = set(processes) - baseline.get(index, set())
        used = sum(
            memory
            for process, memory in processes.items()
            if process in owners and memory is not None
        )
        if used:
            fractions.append(used / total)
    if not fractions or not all(0 < value < 1 for value in fractions):
        raise RuntimeError(
            "Cannot account for the weight cache daemon's GPU memory. "
            "Use kv_cache_memory_bytes with an explicit KV allocation, or expose "
            "driver process memory accounting to this container."
        )
    return max(fractions)


HOT_RELOAD_FIELDS = {"reasoning_content": "bool"}
IMAGE_LIMIT_FIELDS = {
    "vllm": "limit_mm_per_prompt",
    "sglang": "limit_mm_data_per_request",
}

# Only execution parameters may change while the daemon owns the weights.
# In particular, do not accept arbitrary engine arguments: some change packed
# weight layouts without changing a model name (attention/MoE backends, RoPE).
RELOAD_FIELDS = {
    "vllm": {
        **HOT_RELOAD_FIELDS,
        "limit_mm_per_prompt": "image_limits",
        "max_num_seqs": "positive_int",
        "max_num_batched_tokens": "positive_int",
        "max_model_len": "positive_int",
        "gpu_memory_utilization": "fraction",
        "enforce_eager": "bool",
    },
    "sglang": {
        **HOT_RELOAD_FIELDS,
        "limit_mm_data_per_request": "image_limits",
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


def parse_weight_cache_option(value: Any) -> Optional[bool]:
    if value is None:
        return None
    # CLI extra arguments arrive as strings.
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError("enable_weight_cache must be a boolean or null")


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
        elif kind == "image_limits":
            valid = (
                isinstance(value, dict)
                and "image" in value
                and not set(value) - {"image", "video", "audio"}
                and all(type(v) is int and v >= 0 for v in value.values())
                and value["image"] > 0
            )
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


def _watcher_command(
    parent_pid: int, command: list[str], python_executable: Optional[str] = None
) -> list[str]:
    # Running with -m imports xinference.model.llm before the watcher can start.
    # That loads engine registries and heavyweight libraries such as torch even
    # though the watcher only manages processes. This file can run standalone.
    return [
        python_executable or sys.executable,
        str(Path(__file__).resolve()),
        str(parent_pid),
        *command,
    ]


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
        if sys.platform == "linux" and len(os.fsencode(self._directory.name)) > 20:
            self._directory.cleanup()
            self._directory = tempfile.TemporaryDirectory(prefix="xinfw-", dir="/tmp")
        self.directory = Path(self._directory.name)
        self.process: Optional[subprocess.Popen] = None
        self._log: Optional[IO[str]] = None
        self._saved_env: Dict[str, Optional[str]] = {}
        self._socket_paths: Dict[Path, int] = {}
        self._ready_paths: list[Path] = []
        self._memory_fraction: Optional[float] = None

    def start(
        self,
        timeout: float = 1800,
        *,
        python_executable: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> None:
        config = {k: v for k, v in self.config.items() if v is not None}
        config.pop("launch_timeout", None)
        # Parser dataclasses contain engine-initialized, init=False fields.
        # They do not affect retained weights and cannot be round-tripped
        # through the upstream preload CLI's strict config parser.
        for key in (
            "reasoning_config",
            "structured_outputs_config",
            "reasoning_parser",
            "reasoning_parser_plugin",
        ):
            config.pop(key, None)
        env = dict(os.environ if environment is None else environment)
        if python_executable:
            env["PATH"] = os.pathsep.join(
                [os.path.dirname(python_executable), env.get("PATH", "")]
            )
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
            config_path.write_text(
                json.dumps(config, default=dataclasses.asdict), encoding="utf-8"
            )
            self._log = (self.directory / "daemon.log").open("w+")
            baseline = None
            if self.engine == "vllm" and not self.config.get("kv_cache_memory_bytes"):
                baseline = {
                    index: set(processes)
                    for index, (_, processes) in _gpu_process_memory().items()
                }
            self.process = subprocess.Popen(
                _watcher_command(
                    os.getpid(),
                    [*command, "--config", str(config_path)],
                    python_executable,
                ),
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
                        if baseline is not None:
                            self._memory_fraction = _cached_memory_fraction(
                                self.process.pid, baseline
                            )
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
            else:
                # A dead rank can leave its socket inode behind while the
                # launcher remains alive. Connect to every listening rank.
                for path in self._socket_paths:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                        probe.settimeout(2)
                        probe.connect(str(path))
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
        try:
            native_budget = Version(package_version("vllm")) >= Version("0.32.0")
        except PackageNotFoundError:
            native_budget = False
        if native_budget:
            # vLLM 0.32 accounts for externally held IPC weights itself, even
            # with an explicit KV allocation. Subtracting twice (or passing
            # 0.01) fails its preflight before the engine can load.
            return config
        assert self.process is not None
        requested = float(config.get("gpu_memory_utilization", 0.9))
        if config.get("kv_cache_memory_bytes"):
            # vLLM ignores this fraction for an explicit KV allocation, but
            # still checks it against free memory before loading IPC weights.
            effective = min(requested, 0.01)
        else:
            fraction = self._memory_fraction
            if fraction is None:
                fraction = _cached_memory_fraction(self.process.pid)
            effective = requested - fraction
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

    def export(self) -> Dict[str, Any]:
        self.client_config()
        assert self.process is not None
        return {
            "engine": self.engine,
            "config": copy.deepcopy(self.config),
            "directory": str(self.directory),
            "pid": self.process.pid,
            "sockets": {str(path): inode for path, inode in self._socket_paths.items()},
            "ready": [str(path) for path in self._ready_paths],
            "memory_fraction": self._memory_fraction,
        }

    def stop(self) -> None:
        if self.process is not None:
            _terminate_process_group(self.process.pid, reap=self.process.poll)
            self.process.wait(timeout=5)
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


class _CacheProcess:
    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        import psutil

        return None if psutil.pid_exists(self.pid) else 1


class BorrowedWeightCache(WeightCacheDaemon):
    """A checked, non-owning lease. Only the stable actor may stop the daemon."""

    def __init__(self, descriptor: Dict[str, Any]):
        self.engine = descriptor["engine"]
        self.config = copy.deepcopy(descriptor["config"])
        self.directory = Path(descriptor["directory"])
        self.process = _CacheProcess(descriptor["pid"])  # type: ignore
        self._socket_paths = {
            Path(path): inode for path, inode in descriptor["sockets"].items()
        }
        self._ready_paths = [Path(path) for path in descriptor["ready"]]
        self._memory_fraction = descriptor["memory_fraction"]
        if self.engine == "sglang":
            os.environ["SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE"] = str(
                self.directory / "{device_uuid}.sock"
            )
            os.environ["SGLANG_WEIGHT_CACHE_READY_TEMPLATE"] = str(
                self.directory / "{device_uuid}.ready"
            )
        self.client_config()

    def stop(self) -> None:
        # Destroying a failed/retired engine must never revoke another engine's
        # mappings or the stable actor's cached weights.
        pass


class WeightCachedModel:
    _enable_weight_cache: bool = False
    _weight_cache: Optional[WeightCacheDaemon] = None
    _weight_cache_engine: str
    _model_config: Any
    _n_worker: int
    model_path: str
    model_spec: Any
    _loading_error: Any
    _loading_thread: Any
    reasoning_parser: Any

    if TYPE_CHECKING:

        def _sanitize_model_config(self, config: Any) -> Any: ...

        def _stop_engine(self) -> None: ...

        def load(self) -> None: ...

        def wait_for_load(self) -> None: ...

        def _get_xinference_executor_backend(self) -> str: ...

    def _init_weight_cache(self, config: Any) -> None:
        self._weight_cache_option = parse_weight_cache_option(
            config.pop("enable_weight_cache", None)
        )
        self._enable_weight_cache = self._weight_cache_option is True
        self._reload_config = copy.deepcopy(config)

    def _weight_cache_unavailable_reason(self) -> Optional[str]:
        engine = self._weight_cache_engine
        config = {**self._reload_config, **self._model_config}
        import torch

        if engine == "vllm":
            if getattr(self, "model_spec", None) is not None and (
                self.model_spec.model_format == "ggufv2"
            ):
                return "Weight caching does not support GGUF models"
            if (
                getattr(self, "_get_xinference_executor_backend", lambda: "auto")()
                == "xoscar"
            ):
                return "Weight caching requires the native vLLM executor"
        minimum = "0.31.0" if engine == "vllm" else "0.5.21"
        if Version(package_version(engine)) < Version(minimum):
            return f"enable_weight_cache requires {engine}>={minimum}"
        if sys.platform != "linux" or not torch.cuda.is_available():
            return "Weight caching requires Linux with CUDA or ROCm"
        if (
            self._n_worker != 1
            or config.get("pipeline_parallel_size", config.get("pp_size", 1)) != 1
            or config.get("data_parallel_size", config.get("dp_size", 1)) != 1
            or config.get("nnodes", 1) != 1
            or getattr(self, "_xavier_config", None) is not None
            or getattr(self, "_nixl_config", None) is not None
            or config.get("_xavier_cache_config") is not None
            or config.get("_nixl_config") is not None
            or config.get("kv_transfer_config")
            or config.get("disaggregation_mode", "null") not in ("null", None)
        ):
            return "Weight caching currently supports single-worker tensor parallelism"
        if config.get("enable_sleep_mode") or config.get("cpu_offload_gb", 0):
            return "Weight caching cannot be combined with weight offloading"
        if config.get("load_format") == "ipc_cache" or config.get("weight_cache_mode"):
            return "Xinference manages the weight cache loader automatically"
        if (
            getattr(self, "lora_modules", None)
            or config.get("draft_model_path")
            or config.get("lora_paths")
            or config.get("enable_lora")
            or config.get("speculative_algorithm")
            or config.get("speculative_config")
        ):
            return "Weight caching does not yet support this adapter/draft setup"
        return None

    def _resolve_weight_cache(self) -> None:
        # Resolve in the model subprocess, whose engine, device visibility and
        # environment can differ from the Worker's download environment.
        if self._weight_cache is not None:
            return
        option = getattr(self, "_weight_cache_option", self._enable_weight_cache)
        if option is False:
            self._enable_weight_cache = False
            return
        reason = self._weight_cache_unavailable_reason()
        if reason is not None:
            if option is True:
                raise ValueError(reason)
            self._enable_weight_cache = False
            logger.debug("GPU weight retention disabled: %s", reason)
            return
        self._enable_weight_cache = True

    def _prepare_weight_cache(self) -> None:
        if self._weight_cache is None:
            self._resolve_weight_cache()
        if not self._enable_weight_cache:
            return
        config = self._model_config
        engine = self._weight_cache_engine
        initial = self._weight_cache is None
        if initial:
            daemon_config = copy.deepcopy(config)
            # This is an engine process routing option, not a weight layout.
            daemon_config.pop("distributed_executor_backend", None)
            daemon = WeightCacheDaemon(engine, self.model_path, daemon_config)
            daemon.start()
            self._weight_cache = daemon
        assert self._weight_cache is not None
        config.update(self._weight_cache.client_config())
        # Publish a separate immutable snapshot. load() continues mutating the
        # engine config on another thread after this point.
        if initial:
            self._reload_config = {
                **self._reload_config,
                **{
                    k: copy.deepcopy(v)
                    for k, v in config.items()
                    if k in RELOAD_FIELDS[engine]
                },
            }

    @staticmethod
    def _is_hot_patch(patch: Dict[str, Any]) -> bool:
        return bool(patch) and set(patch) <= HOT_RELOAD_FIELDS.keys()

    def _reload_parameters(self) -> Dict[str, str]:
        parameters = dict(RELOAD_FIELDS[self._weight_cache_engine])
        abilities = getattr(getattr(self, "model_family", None), "model_ability", [])
        image_field = IMAGE_LIMIT_FIELDS[self._weight_cache_engine]
        limits = self._reload_config.get(image_field)
        image_limit = limits.get("image") if isinstance(limits, dict) else None
        minimum = 1 if self._weight_cache_engine == "vllm" else 0
        valid_limit = type(image_limit) is int and image_limit >= minimum
        if self._weight_cache_engine == "sglang" and (
            limits is None or (isinstance(limits, dict) and "image" not in limits)
        ):
            # SGLang defaults to no request-count limit; zero does not unload
            # its encoder either. Both can become a positive request limit.
            valid_limit = True
        if (
            not self._enable_weight_cache
            or not {"vision", "omni"}.intersection(abilities)
            or not valid_limit
        ):
            parameters.pop(image_field, None)
        if not getattr(
            getattr(self, "reasoning_parser", None), "reasoning_end_tag", None
        ):
            parameters.pop("reasoning_content", None)
        return parameters

    def get_reload_config(self) -> Dict[str, Any]:
        parameters = self._reload_parameters()
        config = {
            k: copy.deepcopy(v)
            for k, v in self._reload_config.items()
            if k in parameters and v is not None
        }
        if "reasoning_content" in parameters:
            config.setdefault(
                "reasoning_content", self.reasoning_parser.reasoning_content
            )
        return {
            "enabled": self._enable_weight_cache,
            "engine": self._weight_cache_engine,
            "model_config": config,
            "parameters": parameters,
            "hot_parameters": [key for key in HOT_RELOAD_FIELDS if key in parameters],
        }

    def _weight_cache_engine_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self._weight_cache is None:
            return config
        return self._weight_cache.engine_config(config)

    def validate_reload(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        validate_reload_patch(self._weight_cache_engine, patch)
        from .context_length import CONTEXT_LENGTH_KEYS

        family = getattr(self, "model_family", None)
        extension = getattr(family, "context_length_extension", None)
        context_key = CONTEXT_LENGTH_KEYS[self._weight_cache_engine]
        if (
            family is not None
            and extension is not None
            and self._weight_cache_engine in extension.engines
        ):
            current = self._reload_config.get(context_key) or family.context_length
            if context_key in patch and patch[context_key] > current:
                raise ValueError(
                    "Increasing context length for a model with a YaRN recipe "
                    "requires a new deployment so RoPE settings can be recalculated"
                )
        config = {**copy.deepcopy(self._reload_config), **copy.deepcopy(patch)}
        parameters = self._reload_parameters()
        if "reasoning_content" in patch and "reasoning_content" not in parameters:
            raise ValueError("This model does not support reasoning-content parsing")
        if self._is_hot_patch(patch):
            return config
        if self._weight_cache is None:
            raise ValueError(
                "Launch this model with enable_weight_cache=true before reloading"
            )
        self._weight_cache.client_config()
        image_field = IMAGE_LIMIT_FIELDS[self._weight_cache_engine]
        if image_field in patch:
            abilities = getattr(
                getattr(self, "model_family", None), "model_ability", []
            )
            if not {"vision", "omni"}.intersection(abilities):
                raise ValueError("Image-limit reload requires a vision model")
            previous = self._sanitize_model_config(copy.deepcopy(self._reload_config))
            limits = previous.get(image_field) or {}
            image_limit = limits.get("image")
            if self._weight_cache_engine == "vllm" and (
                type(image_limit) is not int or image_limit <= 0
            ):
                raise ValueError(
                    "Image-limit reload requires an existing positive image limit; "
                    "switching between disabled and enabled encoders is not supported"
                )
            requested = patch[image_field]
            if any(k != "image" and v != limits.get(k) for k, v in requested.items()):
                raise ValueError("Only the image count can change through reload")
            config[image_field] = {**limits, **requested}
            validate_reload_patch(
                self._weight_cache_engine,
                {image_field: config[image_field]},
            )
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
        if self._is_hot_patch(patch):
            # Replace the parser rather than mutating an object held by a response.
            parser = copy.copy(self.reasoning_parser)
            parser.reasoning_content = config["reasoning_content"]
            self.reasoning_parser = parser
            self._reload_config = config
            progress("ready")
            return
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


def _process_group_alive(pgid: int, exclude_pid: Optional[int] = None) -> bool:
    import psutil

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Darwin can report EPERM for an orphaned group containing only
        # zombies. Inspect live members before treating it as still running.
        pass
    # Zombies have released GPU memory but can keep killpg(pgid, 0) alive until
    # reaped by init. The watcher itself must not count as a surviving rank.
    for process in psutil.process_iter(["pid", "status"]):
        try:
            if (
                process.pid != exclude_pid
                and process.info["status"] != psutil.STATUS_ZOMBIE
                and os.getpgid(process.pid) == pgid
            ):
                return True
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return False


def _terminate_process_group(
    pgid: int,
    exclude_pid: Optional[int] = None,
    reap: Optional[Callable[[], Any]] = None,
    grace: float = 3,
) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            if not _process_group_alive(pgid, exclude_pid):
                return
            raise
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if reap is not None:
                reap()
            if not _process_group_alive(pgid, exclude_pid):
                return
            time.sleep(0.05)
    raise RuntimeError("Weight cache process group did not terminate")


def _run_watched(parent_pid: int, command: list) -> int:
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    child = subprocess.Popen([sys.executable, *command])
    try:
        while child.poll() is None and not stopping:
            if os.getppid() != parent_pid:
                break
            time.sleep(0.2)
    finally:
        _terminate_process_group(os.getpgrp(), exclude_pid=os.getpid(), reap=child.poll)
        child.wait(timeout=5)
    return child.returncode


if __name__ == "__main__":
    sys.exit(_run_watched(int(sys.argv[1]), sys.argv[2:]))
