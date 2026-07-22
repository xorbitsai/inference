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

import asyncio
import json
import logging
import os
import pathlib
import platform
import queue
import re
import shutil
import signal
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import xoscar as xo
from async_timeout import timeout
from xoscar import MainActorPoolType

from ..client.restful.restful_client import Client as RESTfulClient
from ..constants import (
    XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU,
    XINFERENCE_CACHE_DIR,
    XINFERENCE_DISABLE_HEALTH_CHECK,
    XINFERENCE_ENABLE_VIRTUAL_ENV,
    XINFERENCE_HEALTH_CHECK_INTERVAL,
    XINFERENCE_HOME,
    XINFERENCE_LOG_CONSOLE,
    XINFERENCE_LOG_DOWNLOAD_PROGRESS,
    XINFERENCE_MAX_CONCURRENT_LAUNCHES,
    XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT,
    XINFERENCE_MODEL_DOWNLOAD_WORKERS,
    XINFERENCE_STATUS_GATHER_TIMEOUT,
    XINFERENCE_STATUS_REPORT_MULTIPLIER,
    XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT,
    XINFERENCE_TCP_REQUEST_TIMEOUT,
    XINFERENCE_VIRTUAL_ENV_DIR,
    XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL,
    XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED,
    is_metrics_disabled,
)
from ..core.model import ModelActor
from ..core.status_guard import LaunchStatus
from ..device_utils import get_available_device_env_name, gpu_count
from ..model.core import VirtualEnvSettings, create_model_instance
from ..model.utils import (
    CancellableDownloader,
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from ..types import PeftModelConfig
from ..utils import get_pip_config_args, get_real_path
from .cache_tracker import CacheTrackerActor
from .event import Event, EventCollectorActor, EventType
from .exceptions import ModelNotReadyError
from .metrics import (
    launch_metrics_export_server,
    record_metrics,
    set_build_info,
    set_config_info,
)
from .resource import gather_node_info
from .status_guard import StatusGuardActor
from .utils import (
    apply_engine_virtualenv_settings,
    build_subpool_envs_for_virtual_env,
    filter_virtualenv_packages_by_markers,
    find_direct_reference_packages,
    log_async,
    log_sync,
    merge_virtual_env_packages,
    parse_replica_model_uid,
    purge_dir,
    rewrite_direct_url_packages_for_index,
)
from .virtual_env_manager import VirtualEnvManager as XinferenceVirtualEnvManager
from .virtual_env_manager import (
    expand_engine_dependency_placeholders,
    get_engine_critical_dependency_specs,
    get_engine_model_format_virtualenv_packages,
    is_cuda_compatible,
    resolve_virtualenv_python_path,
)

try:
    from xoscar.virtualenv import VirtualEnvManager
except ImportError:
    VirtualEnvManager = None

if TYPE_CHECKING:
    from .progress_tracker import Progressor

logger = getLogger(__name__)


@contextmanager
def _exclusive_venv_path_lock(env_path: str):
    """
    Serialize ``uv venv`` creation and subsequent .pth injection for the same
    logical virtualenv path. Avoids TOCTOU races when multiple model replicas
    call ``create_env`` concurrently (see Xinference virtualenv v4 layout).

    Ensures the lock file's parent directory exists before ``os.open`` so cold
    starts work after the entire venv tree was removed.

    Uses a sibling lock file ``{realpath(env_path)}.xinference-venv.lock``.
    On Windows this is a no-op (``fcntl`` unavailable / different semantics).
    """
    if os.name == "nt":
        yield
        return

    import fcntl

    real = os.path.realpath(os.path.normpath(env_path))
    lock_path = f"{real}.xinference-venv.lock"
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# Strip test-injected envs from cached launch_args before recover.
# All test-specific env vars MUST use the XINFERENCE_TEST_ prefix so they can be
# stripped here and not persist across recover / worker restart cycles.
_TEST_ENV_RE = re.compile(r"^XINFERENCE_TEST_")

# Max retries for persisting cleaned launch_args to disk.
_PERSIST_RETRY_MAX = 3

# VRAM reclaim timeout (seconds) for orphan cleanup waits.
_VRAM_RECLAIM_TIMEOUT = 30
# VRAM free ratio threshold — ratio >= this means "released".
_VRAM_READY_RATIO = 0.90
# nvmlInit timeout (seconds) — bounds a stuck GPU driver so worker startup
# and pre-launch VRAM checks cannot hang indefinitely.
_NVML_INIT_TIMEOUT = 10


def _strip_test_envs(launch_args: dict) -> Tuple[dict, Set[str]]:
    """Strip XINFERENCE_TEST_* envs from launch_args.

    Returns (cleaned_launch_args, stripped_keys). The cleaned dict is a
    shallow copy of the top level with an independent copy of envs, so
    mutating the result never affects the original. stripped_keys lets
    callers determine whether anything was actually removed.

    Exception-safe: any error returns (shallow copy of input, empty set)
    so the recover path is never blocked.
    """
    try:
        launch_args = dict(launch_args)
        original_envs = launch_args.get("envs")

        if not original_envs:
            return launch_args, set()

        if not isinstance(original_envs, dict):
            _uid = launch_args.get("model_uid", "<unknown>")
            logger.error(
                "launch_args['envs'] is %s (not dict) for model_uid=%s, "
                "launch_args_keys=%s, skip strip",
                type(original_envs).__name__,
                _uid,
                sorted(launch_args.keys()),
            )
            return launch_args, set()

        cleaned = {k: v for k, v in original_envs.items() if not _TEST_ENV_RE.match(k)}
        stripped = set(original_envs.keys()) - set(cleaned.keys())

        if cleaned:
            launch_args["envs"] = cleaned
        else:
            launch_args.pop("envs")

        return launch_args, stripped
    except Exception as e:
        logger.error("_strip_test_envs unexpected error: %s", e, exc_info=True)
        return dict(launch_args), set()


def _snapshot_gpu_occupying_pids(device_indices: list) -> set:
    """List PIDs currently occupying GPU memory via pynvml.

    Returns set of int. Gracefully degrades to empty set if pynvml is
    unavailable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return set()
    pids: set = set()
    try:
        for idx in device_indices:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
                for pro in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                    if pro.pid:
                        pids.add(pro.pid)
            except Exception:
                continue
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return pids


def _snapshot_gpu_free_ratio(device_indices: list) -> float:
    """Return the minimum free VRAM ratio across specified GPUs.

    Returns 0.0~1.0, or -1 if pynvml is unavailable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return -1
    min_ratio = float("inf")
    try:
        for idx in device_indices:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                ratio = info.free / info.total
                if ratio < min_ratio:
                    min_ratio = ratio
            except Exception:
                continue
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return min_ratio if min_ratio != float("inf") else -1


def _parse_gpu_indices(gpu_idx) -> list:
    """Parse gpu_idx parameter into a list of GPU indices."""
    if gpu_idx is None:
        return []
    if isinstance(gpu_idx, int):
        return [gpu_idx]
    if isinstance(gpu_idx, list):
        return [int(x) for x in gpu_idx]
    if isinstance(gpu_idx, str):
        return [int(x.strip()) for x in gpu_idx.split(",") if x.strip()]
    return []


async def _wait_pids_dead(pids: set, timeout: float = 5.0):
    """Wait until all given PIDs have exited or timeout expires."""
    import psutil

    deadline = time.monotonic() + timeout
    remaining = set(pids)
    while remaining and time.monotonic() < deadline:
        still_alive = set()
        for pid in remaining:
            try:
                p = psutil.Process(pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    still_alive.add(pid)
            except Exception:
                pass
        remaining = still_alive
        if remaining:
            await asyncio.sleep(0.2)


def _process_or_ancestor_has_uid(proc: Any, uid_lower: str) -> bool:
    """Return True if proc or any of its ancestors has uid_lower in cmdline."""
    current = proc
    while current is not None:
        try:
            cmd = " ".join(current.cmdline()).lower()
            if uid_lower in cmd:
                return True
            current = current.parent()
        except Exception:
            return False
    return False


async def _kill_orphan_gpu_pids(
    device_indices: list,
    pre_pids: set,
    model_uid: str = "",
    grace: float = 2.0,
):
    """Snapshot current GPU PIDs, kill orphans not in pre_pids.

    An orphan is a PID present on the GPU that was not in pre_pids (the set
    of PIDs known to belong to other models or the worker itself).

    Requires model_uid in the process cmdline (or in an ancestor's cmdline)
    to avoid killing unrelated processes on shared GPUs.
    """
    post_pids = _snapshot_gpu_occupying_pids(device_indices)
    orphan_pids = [pid for pid in post_pids if pid not in pre_pids]
    if not orphan_pids:
        return []

    import psutil

    killed = []
    uid_lower = model_uid.lower() if model_uid else ""
    for pid in orphan_pids:
        try:
            p = psutil.Process(pid)
            if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
                continue
            cmd = " ".join(p.cmdline()).lower()
            if uid_lower:
                if uid_lower not in cmd and not _process_or_ancestor_has_uid(
                    p, uid_lower
                ):
                    continue
            if not any(k in cmd for k in ("vllm", "enginecore", "python")):
                continue
            p.kill()
            killed.append(pid)
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            ProcessLookupError,
            PermissionError,
        ):
            continue
        except Exception:
            pass
    if killed:
        await asyncio.sleep(grace)
    return killed


def _nvml_init_with_timeout(timeout: int = _NVML_INIT_TIMEOUT) -> bool:
    """Initialize pynvml with a timeout guard.

    nvmlInit is a blocking C call that asyncio.wait_for cannot cancel. On
    Unix we use SIGALRM to bound it so a stuck GPU driver cannot hang worker
    startup. On Windows SIGALRM is unavailable; we call nvmlInit directly
    (Windows rarely runs vLLM, and the no-lock degradation is acceptable).
    signal.signal also requires the main thread; in background-thread
    contexts (some test setups, customized deployments) we fall back to a
    direct nvmlInit rather than raising ValueError.

    Returns True on success, False on failure or timeout (caller degrades
    gracefully). The original SIGALRM handler is always restored.
    """
    import threading

    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        try:
            import pynvml

            pynvml.nvmlInit()
            return True
        except Exception:
            return False

    import pynvml

    def _alarm_handler(signum, frame):
        raise TimeoutError("nvmlInit timed out")

    _old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)
        try:
            pynvml.nvmlInit()
            return True
        except Exception:
            return False
        finally:
            signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, _old_handler)


async def _kill_gpu_orphans_by_ppid(
    device_indices: list,
    model_uid: str = "",
) -> List[int]:
    """SIGKILL GPU-occupying processes whose parent is init (PPID == 1).

    Used at worker startup (H1) and pre-launch VRAM recheck (H2). Unlike
    _kill_orphan_gpu_pids (which uses post - pre diff and would mis-kill
    other workers' vLLM processes on shared GPUs), this only targets true
    orphans: processes whose parent has died and been reparented to PID 1.

    A process is killed only if ALL hold:
      1. occupies GPU memory on one of device_indices (per NVML)
      2. PPID == 1 (parent dead, reparented to init)
      3. cmdline contains vllm / enginecore / start_sub_pool
         (prevents PID-reuse misidentification of unrelated processes)

    cmdline is re-checked immediately before SIGKILL to defend against PID
    reuse between the snapshot and the kill. SIGTERM is sent first, then
    after a 1s grace, SIGKILL is applied to survivors. psutil.Process
    objects are retained across the grace sleep so is_running() detects
    PID reuse via create_time comparison rather than re-resolving the PID.
    """
    import psutil

    occupying_pids = _snapshot_gpu_occupying_pids(device_indices)
    orphan_processes: List["psutil.Process"] = []
    for pid in occupying_pids:
        if pid == os.getpid():
            continue
        try:
            p = psutil.Process(pid)
            if p.ppid() != 1:
                continue
            cmd = " ".join(p.cmdline()).lower()
            if not any(k in cmd for k in ("vllm", "enginecore", "start_sub_pool")):
                continue
            orphan_processes.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not orphan_processes:
        return []

    logger.warning(
        "Found %d vLLM orphan(s) on GPUs %s: %s",
        len(orphan_processes),
        device_indices,
        sorted(p.pid for p in orphan_processes),
    )

    for p in orphan_processes:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    await asyncio.sleep(1.0)

    killed: List[int] = []
    for p in orphan_processes:
        try:
            if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
                continue
            cmd = " ".join(p.cmdline()).lower()
            if not any(k in cmd for k in ("vllm", "enginecore", "start_sub_pool")):
                continue
            # Use psutil.kill() rather than os.kill(pid, signal.SIGKILL):
            # signal.SIGKILL is undefined on Windows, and psutil's kill() is
            # cross-platform (TerminateProcess on Windows).
            p.kill()
            killed.append(p.pid)
            logger.warning("SIGKILL orphan pid %s on GPUs %s", p.pid, device_indices)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            logger.debug("Kill pid %s failed", p.pid, exc_info=True)
    return killed


@dataclass
class ModelStatus:
    last_error: str = ""
    model_state: str = ""


@dataclass
class LaunchInfo:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # virtualenv manager
    virtual_env_manager: Optional["VirtualEnvManager"] = None
    # downloader, report progress or cancel entire download
    downloader: Optional[CancellableDownloader] = None
    # sub pools created for the model
    sub_pools: Optional[List[str]] = None


class WorkerActor(xo.StatelessActor):
    def __init__(
        self,
        supervisor_address: str,
        supervisor_endpoint: Optional[str],
        main_pool: MainActorPoolType,
        gpu_devices: List[int],
        metrics_exporter_host: Optional[str] = None,
        metrics_exporter_port: Optional[int] = None,
    ):
        super().__init__()
        # static attrs.
        self._total_gpu_devices = gpu_devices
        self._supervisor_address = supervisor_address
        self._supervisor_endpoint = supervisor_endpoint
        self._supervisor_ref: Optional[xo.ActorRefType] = None
        self._main_pool = main_pool
        self._main_pool.recover_sub_pool = self.recover_sub_pool
        self._status_guard_ref: xo.ActorRefType[
            "StatusGuardActor"
        ] = None  # type: ignore
        self._event_collector_ref: xo.ActorRefType[  # type: ignore
            EventCollectorActor
        ] = None
        self._cache_tracker_ref: xo.ActorRefType[
            CacheTrackerActor
        ] = None  # type: ignore
        self._progress_tracker_ref: Optional[xo.ActorRefType] = None
        # Tracks whether this worker has successfully called supervisor.add_worker().
        # _supervisor_ref being non-None no longer implies registered, because
        # heartbeat() uses add_worker=False and may populate the ref cache before
        # registration completes. This flag is the source of truth for registration
        # state, used by get_supervisor_ref to decide whether add_worker is needed.
        self._registered: bool = False

        # Virtual environment management
        self._virtual_env_manager: XinferenceVirtualEnvManager = None  # type: ignore

        # internal states.
        # temporary placeholder during model launch process:
        self._model_uid_launching_guard: Dict[str, LaunchInfo] = {}
        # Launch concurrency control
        self._launch_semaphore = asyncio.Semaphore(XINFERENCE_MAX_CONCURRENT_LAUNCHES)
        self._launch_active = 0
        self._launch_waiting = 0
        # attributes maintained after model launched:
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, Dict[str, Any]] = {}
        self._model_uid_to_model_status: Dict[str, ModelStatus] = {}
        self._gpu_to_model_uids: Dict[int, Set[str]] = defaultdict(set)
        # Dict structure: gpu_index: {(replica_model_uid, model_type)}
        self._user_specified_gpu_to_model_uids: Dict[int, Set[Tuple[str, str]]] = (
            defaultdict(set)
        )
        self._allow_multi_replica_per_gpu = XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU
        self._model_uid_to_addr: Dict[str, str] = {}
        self._model_uid_to_recover_count: Dict[str, Optional[int]] = {}
        self._model_uid_to_launch_args: Dict[str, Dict] = {}
        self._model_uid_to_pid: Dict[str, int] = {}
        # Per-replica sub-pool process PIDs (primary ModelActor pool + per-device
        # vLLM/SGLang rank pools). Used by report_status to attribute NVML GPU
        # memory back to the owning replica deterministically, without reading
        # any process environ.
        self._model_uid_to_subpool_pids: Dict[str, Set[int]] = {}

        if is_metrics_disabled():
            logger.info(
                "Worker metrics is disabled due to the environment XINFERENCE_DISABLE_METRICS=1"
            )
        elif metrics_exporter_host is not None or metrics_exporter_port is not None:
            # metrics export server.
            logger.info(
                f"Starting metrics export server at {metrics_exporter_host}:{metrics_exporter_port}"  # noqa: E231
            )
            q: queue.Queue = queue.Queue()
            self._metrics_thread = threading.Thread(
                name="Metrics Export Server",
                target=launch_metrics_export_server,
                args=(q, metrics_exporter_host, metrics_exporter_port),
                daemon=True,
            )
            self._metrics_thread.start()
            logger.info("Checking metrics export server...")
            while self._metrics_thread.is_alive():
                try:
                    host, port = q.get(block=False)[:2]
                    logger.info(
                        f"Metrics server is started at: http://{host}:{port}"  # noqa: E231
                    )
                    break
                except queue.Empty:
                    pass
            else:
                raise Exception("Metrics server thread exit.")

        # Initialize virtual environment manager
        self._virtual_env_manager = XinferenceVirtualEnvManager(self.address)

        self._lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        # Serializes MainActorPool.append_sub_pool calls. xoscar's
        # append_sub_pool has no internal lock; concurrent fork+exec under
        # XINFERENCE_MAX_CONCURRENT_LAUNCHES>1 can deadlock on fork+GIL and
        # block the event loop (which also disables the sub-pool creation
        # timeout below). The lock only covers append_sub_pool (milliseconds),
        # not download or model load, so launch throughput is unaffected.
        self._subpool_creation_lock = asyncio.Lock()
        self._persist_launch_args_dirty_uids: Set[str] = set()
        self._persist_retry_count: Dict[str, int] = {}

    async def recover_sub_pool(self, address):
        logger.warning("Process %s is down.", address)
        # Xoscar does not remove the address from sub_processes.
        try:
            await self._main_pool.remove_sub_pool(address)
        except Exception:
            pass
        for model_uid, addr in self._model_uid_to_addr.items():
            if addr == address:
                launch_args = self._model_uid_to_launch_args.get(model_uid)
                if launch_args is None:
                    logger.warning(
                        "Not recreate model because the it is down during launch."
                    )
                else:
                    recover_count = self._model_uid_to_recover_count.get(model_uid)

                    # Strip test-injected envs from cached launch_args before
                    # passing them to recover_model, so test env vars don't
                    # persist across recover cycles.
                    launch_args, stripped_keys = _strip_test_envs(launch_args)
                    if stripped_keys:
                        logger.warning(
                            "Stripped test envs on recover for %s: stripped=%s, kept=%s",
                            model_uid,
                            sorted(stripped_keys),
                            sorted(launch_args.get("envs", {}).keys()),
                        )
                        self._model_uid_to_launch_args[model_uid] = launch_args
                        try:
                            async with self._persist_lock:
                                self._persist_launch_args()
                                self._persist_launch_args_dirty_uids.discard(model_uid)
                                self._persist_retry_count.pop(model_uid, None)
                        except Exception as e:
                            retry_n = self._persist_retry_count.get(model_uid, 0) + 1
                            self._persist_retry_count[model_uid] = retry_n
                            if retry_n >= _PERSIST_RETRY_MAX:
                                self._persist_launch_args_dirty_uids.discard(model_uid)
                                logger.error(
                                    "Persist cleaned launch_args for %s failed "
                                    "%d times, giving up: %s",
                                    model_uid,
                                    retry_n,
                                    e,
                                )
                            else:
                                self._persist_launch_args_dirty_uids.add(model_uid)
                                logger.warning(
                                    "Failed to persist cleaned launch_args for "
                                    "%s (attempt %d/%d): %s",
                                    model_uid,
                                    retry_n,
                                    _PERSIST_RETRY_MAX,
                                    e,
                                )

                    try:
                        await self.terminate_model(model_uid, is_model_die=True)
                    except Exception:
                        pass

                    # Wait for VRAM reclaim after terminate, then clean up
                    # any orphan GPU processes (e.g. spawn-created EngineCore
                    # that survived subpool removal).
                    _recover_gpu_idx = _parse_gpu_indices(launch_args.get("gpu_idx"))
                    if not _recover_gpu_idx:
                        try:
                            import pynvml

                            pynvml.nvmlInit()
                            try:
                                _recover_gpu_idx = list(
                                    range(pynvml.nvmlDeviceGetCount())
                                )
                            finally:
                                try:
                                    pynvml.nvmlShutdown()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    if _recover_gpu_idx:
                        try:
                            await asyncio.sleep(1.0)
                            _free_ratio = _snapshot_gpu_free_ratio(_recover_gpu_idx)
                            if 0 <= _free_ratio < _VRAM_READY_RATIO:
                                _other_pids: set = {os.getpid()}
                                for (
                                    _uid,
                                    _pids,
                                ) in self._model_uid_to_subpool_pids.items():
                                    _other_pids.update(_pids)
                                _killed = await _kill_orphan_gpu_pids(
                                    _recover_gpu_idx,
                                    _other_pids,
                                    model_uid=model_uid,
                                )
                                if _killed:
                                    logger.warning(
                                        "Killed %d GPU orphan(s) after "
                                        "recover terminate for %s: %s",
                                        len(_killed),
                                        model_uid,
                                        _killed,
                                    )
                            # Poll VRAM until released or timeout
                            _vram_deadline = time.monotonic() + _VRAM_RECLAIM_TIMEOUT
                            while time.monotonic() < _vram_deadline:
                                _free_ratio = _snapshot_gpu_free_ratio(_recover_gpu_idx)
                                if _free_ratio < 0 or _free_ratio >= _VRAM_READY_RATIO:
                                    break
                                await asyncio.sleep(1.0)
                            else:
                                logger.warning(
                                    "VRAM reclaim timed out after %.0fs for "
                                    "%s, min_free_ratio=%.2f",
                                    _VRAM_RECLAIM_TIMEOUT,
                                    model_uid,
                                    _snapshot_gpu_free_ratio(_recover_gpu_idx),
                                )
                        except Exception:
                            logger.debug(
                                "VRAM/orphan cleanup error for %s",
                                model_uid,
                                exc_info=True,
                            )

                    if recover_count is not None:
                        if recover_count > 0:
                            logger.warning(
                                "Recreating model actor %s, remain %s times ...",
                                model_uid,
                                recover_count - 1,
                            )
                            event_model_uid, _ = parse_replica_model_uid(model_uid)
                            try:
                                if self._event_collector_ref is not None:
                                    await self._event_collector_ref.report_event(
                                        event_model_uid,
                                        Event(
                                            event_type=EventType.WARNING,
                                            event_ts=int(time.time()),
                                            event_content="Recreate model",
                                        ),
                                    )
                            except Exception as e:
                                # Report callback error can be log and ignore, should not interrupt the Process
                                logger.error("report_event error: %s" % (e))
                            finally:
                                del event_model_uid

                            self._model_uid_to_recover_count[model_uid] = (
                                recover_count - 1
                            )
                            # Symmetric to the unbounded branch below: if recreate
                            # itself fails, evict the dead replica so it cannot sit
                            # in a "stopping"/"error" state poisoning routing with
                            # persistent 500s. Same launch_args would fail again, and
                            # recover_model failure leaves no new subpool to retrigger
                            # recover_sub_pool, so eviction (not retry) is the only
                            # recovery. mark_replica_dead is idempotent (safe).
                            try:
                                await self.recover_model(launch_args)
                            except Exception:
                                logger.warning(
                                    "Recreate failed for %s, evicting dead replica "
                                    "from supervisor",
                                    model_uid,
                                    exc_info=True,
                                )
                                await self._evict_replica_from_supervisor(model_uid)
                        else:
                            logger.warning("Stop recreating model actor.")

                            # Bounded retries exhausted: evict the dead replica
                            # from the supervisor's round-robin so traffic stops
                            # routing to it. Failure/timeout is non-fatal -- the
                            # next death detection / redeploy will reconcile.
                            await self._evict_replica_from_supervisor(model_uid)
                    else:
                        logger.warning("Recreating model actor %s ...", model_uid)
                        # Unbounded branch (default, recover_count is None). If
                        # recreate itself fails, evict the dead replica so it
                        # cannot poison routing as a permanent "loading" zombie.
                        # mark_replica_dead is idempotent, so this is safe.
                        try:
                            await self.recover_model(launch_args)
                        except Exception:
                            logger.warning(
                                "Recreate failed for %s, evicting dead replica "
                                "from supervisor",
                                model_uid,
                                exc_info=True,
                            )
                            await self._evict_replica_from_supervisor(model_uid)
                break

    async def _evict_replica_from_supervisor(self, model_uid: str) -> None:
        """Notify the supervisor to evict a dead replica from round-robin.

        Best-effort: ``get_supervisor_ref`` (``add_worker=False``, no
        re-registration) + ``mark_replica_dead`` are wrapped in a single
        ``xo.wait_for(5s)``. ``get_supervisor_ref`` issues blocking
        ``xo.actor_ref`` calls when the cached ref is missing, so the bound
        must cover both to keep a stalled supervisor from holding up the
        worker's local shutdown. A failure/timeout is non-fatal --
        ``mark_replica_dead`` is idempotent and the next death detection /
        redeploy will reconcile.
        """

        async def _notify_replica_dead():
            supervisor_ref = await self.get_supervisor_ref(add_worker=False)
            await supervisor_ref.mark_replica_dead(model_uid)

        try:
            await xo.wait_for(
                _notify_replica_dead(),
                timeout=5,
            )
        except Exception:
            logger.warning(
                "Failed to notify supervisor of dead replica %s",
                model_uid,
                exc_info=True,
            )

    @classmethod
    def default_uid(cls) -> str:
        return "worker"

    def _get_spec_dicts_with_cache_status(
        self, model_family: Any, cache_manager_cls: Type
    ) -> Tuple[List[dict], List[str]]:
        """
        Build model_specs with cache_status and collect download_hubs.
        """

        specs: List[dict] = []
        download_hubs: List[str] = []
        for spec in model_family.model_specs:
            model_hub = spec.model_hub
            if model_hub not in download_hubs:
                download_hubs.append(model_hub)

            family_copy = model_family.copy()
            family_copy.model_specs = [spec]
            cache_manager = cache_manager_cls(family_copy)
            specs.append(
                {**spec.dict(), "cache_status": cache_manager.get_cache_status()}
            )
        return specs, download_hubs

    def _prefer_model_hub(self, model_family: Any, preferred_hub: str = "huggingface"):
        """
        Return a copy of model_family with a single spec, preferring the given hub.
        """
        specs = getattr(model_family, "model_specs", None)
        if not specs:
            return model_family

        target_spec = next(
            (
                spec
                for spec in specs
                if getattr(spec, "model_hub", None) == preferred_hub
            ),
            specs[0],
        )
        family_copy = model_family.copy()
        family_copy.model_specs = [target_spec]
        return family_copy

    async def _cleanup_gpu_orphans_on_startup(self) -> None:
        """Best-effort cleanup of vLLM GPU orphans left by a previous worker.

        vLLM EngineCore / GPU worker processes have no _check_ppid thread, so
        when the worker dies they are reparented to PID 1 and keep holding
        VRAM. The next worker's first sub-pool creation can then deadlock in
        CUDA init. This scans all GPUs at startup and SIGKILLs processes that
        (a) occupy GPU memory, (b) have PPID == 1, and (c) have a vLLM-like
        cmdline. nvmlInit is bounded by _nvml_init_with_timeout so a stuck
        driver cannot block startup. Never raises — failures degrade to a
        warning and rely on H2/C as backstops.
        """
        if not _nvml_init_with_timeout():
            logger.warning(
                "Startup GPU orphan cleanup skipped: pynvml init failed or "
                "timed out (%ss). If the previous worker left vLLM orphans, "
                "launch may hang. Manual check: nvidia-smi + "
                "ps -eo pid,ppid,cmd | grep vllm",
                _NVML_INIT_TIMEOUT,
            )
            return

        try:
            import pynvml

            try:
                device_count = pynvml.nvmlDeviceGetCount()
                all_devices = list(range(device_count))
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            logger.warning(
                "Startup GPU orphan cleanup: nvmlDeviceGetCount failed",
                exc_info=True,
            )
            return

        if not all_devices:
            return

        # Pre-snapshot for diagnostics: distinguish "GPU empty" from
        # "GPU busy but no orphans". _kill_gpu_orphans_by_ppid re-snapshots
        # internally for the actual kill decision.
        occupying_pids = _snapshot_gpu_occupying_pids(all_devices)
        if not occupying_pids:
            logger.info("Startup GPU orphan cleanup: no GPU-occupying processes found")
            return

        killed = await _kill_gpu_orphans_by_ppid(all_devices)
        if not killed:
            logger.info(
                "Startup GPU orphan cleanup: %d GPU-occupying process(es) "
                "found, none are vLLM orphans (all have live parents or "
                "non-matching cmdline)",
                len(occupying_pids),
            )
            return

        try:
            _free_ratio = _snapshot_gpu_free_ratio(all_devices)
            if 0 <= _free_ratio < _VRAM_READY_RATIO:
                _vram_deadline = time.monotonic() + _VRAM_RECLAIM_TIMEOUT
                while time.monotonic() < _vram_deadline:
                    await asyncio.sleep(1.0)
                    _free_ratio = _snapshot_gpu_free_ratio(all_devices)
                    if _free_ratio < 0 or _free_ratio >= _VRAM_READY_RATIO:
                        break
                logger.info(
                    "Startup VRAM reclaim: min_free_ratio=%.2f after orphan " "cleanup",
                    _free_ratio if _free_ratio >= 0 else -1,
                )
        except Exception:
            logger.debug("Startup VRAM poll failed", exc_info=True)

    async def __post_create__(self):
        from ..model.audio import (
            CustomAudioModelFamilyV2,
            generate_audio_description,
            register_audio,
            unregister_audio,
        )
        from ..model.embedding import (
            CustomEmbeddingModelFamilyV2,
            generate_embedding_description,
            register_embedding,
            unregister_embedding,
        )
        from ..model.flexible import (
            FlexibleModelSpec,
            generate_flexible_model_description,
            register_flexible_model,
            unregister_flexible_model,
        )
        from ..model.image import (
            CustomImageModelFamilyV2,
            generate_image_description,
            register_image,
            unregister_image,
        )
        from ..model.llm import (
            CustomLLMFamilyV2,
            generate_llm_version_info,
            register_llm,
            unregister_llm,
        )
        from ..model.rerank import (
            CustomRerankModelFamilyV2,
            generate_rerank_description,
            register_rerank,
            unregister_rerank,
        )
        from ..model.video import (
            CustomVideoModelFamilyV2,
            generate_video_description,
            register_video,
            unregister_video,
        )

        self._custom_register_type_to_cls: Dict[str, Tuple] = {  # type: ignore
            "LLM": (
                CustomLLMFamilyV2,
                register_llm,
                unregister_llm,
                generate_llm_version_info,
            ),
            "embedding": (
                CustomEmbeddingModelFamilyV2,
                register_embedding,
                unregister_embedding,
                generate_embedding_description,
            ),
            "rerank": (
                CustomRerankModelFamilyV2,
                register_rerank,
                unregister_rerank,
                generate_rerank_description,
            ),
            "image": (
                CustomImageModelFamilyV2,
                register_image,
                unregister_image,
                generate_image_description,
            ),
            "audio": (
                CustomAudioModelFamilyV2,
                register_audio,
                unregister_audio,
                generate_audio_description,
            ),
            "flexible": (
                FlexibleModelSpec,
                register_flexible_model,
                unregister_flexible_model,
                generate_flexible_model_description,
            ),
            "video": (
                CustomVideoModelFamilyV2,
                register_video,
                unregister_video,
                generate_video_description,
            ),
        }

        logger.info("Purge cache directory: %s", XINFERENCE_CACHE_DIR)
        purge_dir(XINFERENCE_CACHE_DIR)

        try:
            await self.get_supervisor_ref(add_worker=True)
        except Exception:
            # Do not crash the worker if supervisor is down, auto re-connect later
            logger.error(f"cannot connect to supervisor", exc_info=True)

        # §4.3: If connected as fresh worker, try to recover persisted models
        if self._supervisor_ref is not None and not self._model_uid_to_launch_args:
            try:
                await self._try_recover_models()
            except Exception:
                logger.error("Model recovery failed", exc_info=True)

        if not XINFERENCE_DISABLE_HEALTH_CHECK:
            from ..isolation import Isolation

            # Run _periodical_report_status() in a dedicated thread.
            self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
            self._isolation.start()
            asyncio.run_coroutine_threadsafe(
                self._periodical_report_status(), loop=self._isolation.loop
            )
        logger.info(f"Xinference worker {self.address} started")

        # H1: asynchronously clean up vLLM GPU orphans left by a previous
        # worker that died without reaping its sub-pool descendants. vLLM
        # EngineCore / GPU workers have no _check_ppid, so on worker death
        # they are reparented to PID 1 and keep holding VRAM; the next
        # launch's sub-pool then deadlocks in CUDA init. Runs after
        # registration so it never blocks the worker from serving; H2 is the
        # per-launch backstop if an orphan appears later.
        asyncio.create_task(self._cleanup_gpu_orphans_on_startup())

        # Report build/config info for this worker
        from xinference.constants import XINFERENCE_HOME as _xf_home

        set_build_info(role="worker", worker_address=self.address)
        set_config_info(
            xinference_home=_xf_home,
            role="worker",
            worker_address=self.address,
        )

        # Windows does not have signal handler
        if os.name != "nt":

            async def signal_handler():
                try:
                    supervisor_ref = await self.get_supervisor_ref(add_worker=False)
                    await supervisor_ref.remove_worker(self.address)
                except Exception as e:
                    # Ignore the error of rpc, anyway we are exiting
                    logger.exception("remove worker rpc error: %s", e)
                os._exit(0)

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGINT, lambda: asyncio.create_task(signal_handler())
            )

    async def __pre_destroy__(self):
        self._isolation.stop()

    async def trigger_exit(self) -> bool:
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception as e:
            logger.info(f"trigger exit error: {e}")
            return False
        return True

    async def get_supervisor_ref(self, add_worker: bool = True) -> xo.ActorRefType:
        """
        Try connect to supervisor and return ActorRef. Raise exception on error
        Params:
            add_worker: By default will call supervisor.add_worker after first connect
        """
        from .supervisor import SupervisorActor

        # Cache hit: return immediately only when no registration is required,
        # or the worker has already been registered. When add_worker=True and
        # _registered=False, we must fall through to perform add_worker even if
        # the ref is cached (this happens after heartbeat populated the ref via
        # add_worker=False before registration completed).
        if self._supervisor_ref is not None and (not add_worker or self._registered):
            return self._supervisor_ref
        try:
            supervisor_ref = await xo.actor_ref(  # type: ignore
                address=self._supervisor_address, uid=SupervisorActor.default_uid()
            )
        except Exception:
            await self._refresh_supervisor_address()
            supervisor_ref = await xo.actor_ref(  # type: ignore
                address=self._supervisor_address, uid=SupervisorActor.default_uid()
            )
        # Prevent concurrent operations leads to double initialization, check again.
        if self._supervisor_ref is not None and (not add_worker or self._registered):
            return self._supervisor_ref
        self._supervisor_ref = supervisor_ref
        try:
            if add_worker and not self._registered:
                replica_states = self._get_running_replica_states()
                await supervisor_ref.add_worker(
                    self.address, replica_states=replica_states
                )
                self._registered = True
                if replica_states:
                    logger.info(
                        "Connected to supervisor and replayed %s running model replicas",
                        len(replica_states),
                    )
                else:
                    logger.info("Connected to supervisor as a fresh worker")

            self._status_guard_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=StatusGuardActor.default_uid()
            )
            self._event_collector_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=EventCollectorActor.default_uid()
            )
            self._cache_tracker_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=CacheTrackerActor.default_uid()
            )
            self._progress_tracker_ref = None
        except Exception:
            self._clear_supervisor_refs()
            raise

        # record_model_version is an auxiliary cache-management feature.
        # Its failure must NOT block the worker's core heartbeat/status-report
        # channel. Guard against _cache_tracker_ref being None when the first
        # try block failed and cleared all refs (defensive: avoids AttributeError).
        if self._cache_tracker_ref is None:
            return self._supervisor_ref
        try:
            # cache_tracker is on supervisor
            from ..model.audio import get_audio_model_descriptions
            from ..model.embedding import get_embedding_model_descriptions
            from ..model.flexible import get_flexible_model_descriptions
            from ..model.image import get_image_model_descriptions
            from ..model.llm import get_llm_version_infos
            from ..model.rerank import get_rerank_model_descriptions
            from ..model.video import get_video_model_descriptions

            # record model version
            model_version_infos: Dict[str, List[Dict]] = {}  # type: ignore
            model_version_infos.update(get_llm_version_infos())
            model_version_infos.update(get_embedding_model_descriptions())
            model_version_infos.update(get_rerank_model_descriptions())
            model_version_infos.update(get_image_model_descriptions())
            model_version_infos.update(get_audio_model_descriptions())
            model_version_infos.update(get_video_model_descriptions())
            model_version_infos.update(get_flexible_model_descriptions())
            await self._cache_tracker_ref.record_model_version(
                model_version_infos, self.address
            )
        except Exception:
            logger.warning(
                "Failed to record model version info to cache tracker; "
                "cache management may be degraded",
                exc_info=True,
            )

        return self._supervisor_ref

    def _clear_supervisor_refs(self):
        self._supervisor_ref = None
        # Reset registration state so the next get_supervisor_ref(add_worker=True)
        # re-runs add_worker. Keep this in sync with _supervisor_ref lifecycle.
        self._registered = False
        self._status_guard_ref = None  # type: ignore
        self._event_collector_ref = None  # type: ignore
        self._cache_tracker_ref = None  # type: ignore
        self._progress_tracker_ref = None

    async def _refresh_supervisor_address(self):
        if self._supervisor_endpoint is None:
            return

        refreshed_address = await asyncio.to_thread(
            lambda: RESTfulClient(
                base_url=self._supervisor_endpoint
            )._get_supervisor_internal_address()
        )
        if refreshed_address != self._supervisor_address:
            logger.info(
                "Refreshed supervisor internal address from %s to %s",
                self._supervisor_address,
                refreshed_address,
            )
        self._supervisor_address = refreshed_address

    def _get_running_replica_states(self) -> List[Dict[str, Any]]:
        replica_states: List[Dict[str, Any]] = []
        for replica_model_uid in sorted(self._model_uid_to_model_spec):
            launch_args = self._model_uid_to_launch_args.get(replica_model_uid, {})
            model_spec = self._model_uid_to_model_spec.get(replica_model_uid, {})
            origin_uid, _ = parse_replica_model_uid(replica_model_uid)
            xavier_config = launch_args.get("xavier_config")
            if xavier_config is not None:
                # Xavier recovery still depends on supervisor-owned coordination state,
                # so only replay replicas that the supervisor can reconstruct safely.
                continue
            created_ts = int(launch_args.get("launch_ts") or time.time())
            replica_states.append(
                {
                    "replica_model_uid": replica_model_uid,
                    "n_worker": launch_args.get("n_worker", 1),
                    "shard": launch_args.get("shard", 0),
                    "model_uid": origin_uid,
                    "model_name": model_spec.get(
                        "model_name", launch_args.get("model_name", origin_uid)
                    ),
                    "model_version": model_spec.get(
                        "model_version", launch_args.get("model_version")
                    ),
                    "model_ability": model_spec.get("model_ability", []),
                    "status": LaunchStatus.READY.name,
                    "created_ts": created_ts,
                    "instance_created_ts": created_ts,
                }
            )
        return replica_states

    @staticmethod
    def get_devices_count():
        from ..device_utils import gpu_count

        return gpu_count()

    # §4.3: Worker-side launch_args persistence for auto-recovery.
    def _get_recovery_file_path(self) -> str:
        safe_addr = self.address.replace(":", "_").replace("/", "_")
        recovery_dir = os.path.join(XINFERENCE_HOME, "worker_recovery", safe_addr)
        return os.path.join(recovery_dir, "models.json")

    def _persist_launch_args(self):
        """Atomically persist current launch_args to disk. Failures are non-blocking."""
        try:
            filepath = self._get_recovery_file_path()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Serialize: filter out non-serializable values
            data = {}
            for uid, args in self._model_uid_to_launch_args.items():
                serializable_args = {}
                for k, v in args.items():
                    try:
                        json.dumps(v)
                        serializable_args[k] = v
                    except (TypeError, ValueError):
                        serializable_args[k] = str(v)
                data[uid] = serializable_args
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, filepath)
            logger.debug(
                "Persisted launch_args for %d models to %s", len(data), filepath
            )
        except Exception:
            logger.warning(
                "Failed to persist launch_args, auto-recovery may not work",
                exc_info=True,
            )

    def _remove_persisted_launch_args(self, model_uid: str):
        """Remove a single model from the persisted launch_args file."""
        try:
            filepath = self._get_recovery_file_path()
            if not os.path.exists(filepath):
                return
            with open(filepath, "r") as f:
                data = json.load(f)
            if model_uid in data:
                del data[model_uid]
                tmp_path = filepath + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, filepath)
        except Exception:
            logger.warning("Failed to update persisted launch_args", exc_info=True)

    def _load_persisted_launch_args(self) -> dict:
        """Load persisted launch_args. Returns empty dict on any error."""
        try:
            filepath = self._get_recovery_file_path()
            if not os.path.exists(filepath):
                return {}
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            logger.warning(
                "Failed to load persisted launch_args, treating as fresh worker",
                exc_info=True,
            )
            return {}

    async def _try_recover_models(self):
        """
        After connecting to supervisor as a fresh worker, attempt to recover
        previously running models using persisted launch_args.
        Cross-validates with supervisor to avoid rebuilding manually deleted models.
        """
        persisted = self._load_persisted_launch_args()
        if not persisted:
            return

        logger.info(
            "Found %d persisted model(s) for recovery, validating with supervisor...",
            len(persisted),
        )
        supervisor_ref = self._supervisor_ref
        if supervisor_ref is None:
            logger.warning("No supervisor ref available, skipping model recovery")
            return

        recovered = 0
        skipped = 0
        failed = 0
        for model_uid, launch_args in persisted.items():
            try:
                # Cross-validate: check if supervisor still knows about this model
                origin_uid, _ = parse_replica_model_uid(model_uid)
                try:
                    model_info = await supervisor_ref.describe_model(origin_uid)
                except Exception:
                    model_info = None
                if model_info is None:
                    logger.info(
                        "Model %s no longer registered in supervisor, skipping recovery",
                        model_uid,
                    )
                    skipped += 1
                    continue

                logger.info("Recovering model %s ...", model_uid)
                # Remove non-callable keys that may have been serialized
                launch_args.pop("launch_ts", None)
                # Strip test-injected envs from persisted launch_args
                launch_args, _stripped = _strip_test_envs(launch_args)
                if _stripped:
                    logger.warning(
                        "Stripped test envs on startup recovery for %s: %s",
                        model_uid,
                        sorted(_stripped),
                    )
                await self.launch_builtin_model(**launch_args)
                # Mark the recovered replica ready, mirroring the normal launch
                # path. Same gap as recover_model: launch_builtin_model leaves
                # model_state="loading"; without this, models recovered on worker
                # restart stay "loading" forever. Idempotent if the supervisor
                # also reconciles on reconnect.
                await self.wait_for_load(model_uid)
                recovered += 1
            except Exception:
                logger.error(
                    "Failed to recover model %s, continuing with others",
                    model_uid,
                    exc_info=True,
                )
                failed += 1

        logger.info(
            "Model recovery complete: recovered=%d, skipped=%d, failed=%d",
            recovered,
            skipped,
            failed,
        )
        # Update persisted file to reflect current state
        self._persist_launch_args()

    @log_sync(logger=logger)
    def get_model_count(self) -> int:
        return len(self._model_uid_to_model)

    async def is_model_vllm_backend(self, model_uid: str) -> bool:
        _model_uid, _ = parse_replica_model_uid(model_uid)
        supervisor_ref = await self.get_supervisor_ref()
        model_ref = await supervisor_ref.get_model(_model_uid)
        return await model_ref.is_vllm_backend()

    def allocate_devices(self, model_uid: str, n_gpu: int) -> List[int]:
        if n_gpu > len(self._total_gpu_devices):
            raise RuntimeError("Requested GPUs exceed the number of available devices")

        # If multi-replica-per-GPU is disabled, only pick currently idle GPUs.
        if not self._allow_multi_replica_per_gpu:
            occupied_devices: Set[int] = set()
            for dev, model_uids in self._gpu_to_model_uids.items():
                if model_uids:
                    occupied_devices.add(dev)
            for dev, model_infos in self._user_specified_gpu_to_model_uids.items():
                if model_infos:
                    occupied_devices.add(dev)
            available_devices = [
                dev for dev in self._total_gpu_devices if dev not in occupied_devices
            ]
            if n_gpu > len(available_devices):
                raise RuntimeError("No available slot found for the model")
            selected_devices = available_devices[:n_gpu]
            for dev in selected_devices:
                self._gpu_to_model_uids[int(dev)].add(model_uid)
            return sorted(selected_devices)

        # Default: allow multi-tenant GPUs, pick least-loaded devices.
        gpu_loads: List[Tuple[int, int, int]] = []
        for dev in self._total_gpu_devices:
            running_models = len(self._gpu_to_model_uids.get(dev, set()))
            load = running_models + len(
                self._user_specified_gpu_to_model_uids.get(dev, set())
            )
            # Prefer devices with fewer existing model processes when loads tie
            gpu_loads.append((load, running_models, dev))

        devices: List[int] = []
        for _ in range(n_gpu):
            gpu_loads.sort(key=lambda x: (x[0], x[1], x[2]))
            load, running_models, dev = gpu_loads[0]
            devices.append(dev)
            gpu_loads[0] = (load + 1, running_models + 1, dev)

        for dev in devices:
            self._gpu_to_model_uids[int(dev)].add(model_uid)

        return sorted(devices)

    async def allocate_devices_with_gpu_idx(
        self, model_uid: str, model_type: str, gpu_idx: List[int]
    ) -> List[int]:
        """
        When user specifies the gpu_idx, allocate models on user-specified GPUs whenever possible
        """
        # must be subset of total devices visible to this worker
        if not set(gpu_idx) <= set(self._total_gpu_devices):
            raise ValueError(
                f"Worker {self.address} cannot use the GPUs with these indexes: {gpu_idx}. "
                f"Worker {self.address} can only see these GPUs: {self._total_gpu_devices}."
            )
        # currently just report a warning log when there are already models on these GPUs
        for idx in gpu_idx:
            existing_model_uids = []
            if idx in self._gpu_to_model_uids:
                for rep_uid in self._gpu_to_model_uids[idx]:
                    existing_model_uids.append(rep_uid)
            if not self._allow_multi_replica_per_gpu and (
                existing_model_uids
                or len(self._user_specified_gpu_to_model_uids.get(idx, set())) > 0
            ):
                raise RuntimeError(
                    f"GPU index {idx} has been occupied with models: {existing_model_uids}, "
                    f"and multi-replica-per-GPU is disabled."
                )

            if existing_model_uids:
                logger.warning(
                    f"WARNING!!! GPU index {idx} has been occupied "
                    f"with these models on it: {existing_model_uids}"
                )

        for idx in gpu_idx:
            self._user_specified_gpu_to_model_uids[idx].add((model_uid, model_type))
        return sorted(gpu_idx)

    @log_async(logger=logger)
    async def get_gpu_allocation_status(self) -> Dict[str, Any]:
        """Return current device allocation snapshot for scheduling/diagnostics."""
        return {
            "total": list(self._total_gpu_devices),
            "models": {int(k): list(v) for k, v in self._gpu_to_model_uids.items()},
            "user_specified": {
                int(k): [list(t) for t in v]
                for k, v in self._user_specified_gpu_to_model_uids.items()
            },
            "allow_multi_replica_per_gpu": self._allow_multi_replica_per_gpu,
        }

    def release_devices(self, model_uid: str):
        for dev, model_uids in list(self._gpu_to_model_uids.items()):
            if model_uid in model_uids:
                model_uids.remove(model_uid)
                if not model_uids:
                    del self._gpu_to_model_uids[dev]

        # check user-specified slots
        for dev in self._user_specified_gpu_to_model_uids:
            model_infos = list(
                filter(
                    lambda x: x[0] == model_uid,
                    self._user_specified_gpu_to_model_uids[dev],
                )
            )
            for model_info in model_infos:
                self._user_specified_gpu_to_model_uids[dev].remove(model_info)

    async def _ensure_subpool_monitor(self):
        # The worker main pool is created with n_process=0, so xoscar's
        # start_monitor() (called once in start()) is a no-op because its guard
        # `and self.sub_processes` is empty at that point; monitor_sub_pools
        # therefore never runs and subprocess deaths (OOM / crash / kill) go
        # undetected and unrecovered. After every append_sub_pool the new
        # subpool is already in sub_processes, so calling start_monitor() again
        # here actually starts the monitor loop. start_monitor has its own
        # `_monitor_task is None` guard, so this is idempotent: at most one
        # monitor task per worker process.
        await self._main_pool.start_monitor()

    async def _append_sub_pool_protected(
        self,
        env: Optional[Dict[str, str]] = None,
        start_python: Optional[str] = None,
        model_uid: str = "",
    ) -> str:
        """Append a sub-pool under _subpool_creation_lock with a launch timeout.

        All append_sub_pool call sites must route through this helper so they
        are uniformly protected against (H3) concurrent fork+GIL deadlock and
        (C) single-fork CUDA-init hang. The lock serializes only the
        append_sub_pool call (milliseconds), not download/load.

        On timeout, leftover start_sub_pool children (PPID == this worker) are
        SIGKILLed best-effort and asyncio.TimeoutError is re-raised so the
        caller can run its own failure-path cleanup (e.g. release_devices).
        env keys are logged (never values) because env may carry secrets.
        """
        async with self._subpool_creation_lock:
            try:
                return await xo.wait_for(
                    self._main_pool.append_sub_pool(env=env, start_python=start_python),
                    timeout=XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Subpool creation timed out after %ss for model %s "
                    "(likely fork-unsafe state in worker; restart worker to "
                    "recover). start_python=%s, env_keys=%s",
                    XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT,
                    model_uid,
                    start_python,
                    sorted(env.keys()) if env else [],
                )
                # Kill leftover sub_pool child processes. xo.wait_for only
                # cancels the future; the forked child may still be alive
                # holding GPU memory. Match PPID == this worker's pid and
                # cmdline containing start_sub_pool. Best-effort.
                try:
                    import psutil

                    _my_pid = os.getpid()
                    for _p in psutil.process_iter(["pid", "ppid", "cmdline"]):
                        _info = _p.info
                        if _info["ppid"] != _my_pid:
                            continue
                        _cmd = " ".join(_info.get("cmdline") or []).lower()
                        if "start_sub_pool" in _cmd:
                            try:
                                _p.kill()
                                logger.warning(
                                    "Killed leftover sub_pool process after "
                                    "timeout: pid=%s",
                                    _info["pid"],
                                )
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                except Exception:
                    logger.debug(
                        "Failed to kill leftover sub_pool process",
                        exc_info=True,
                    )
                raise

    async def _allocate_subpool_devices(
        self,
        model_uid: str,
        model_type: Optional[str] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
        gpu_idx: Optional[List[int]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str], List[str]]:
        """Reserve GPU devices for a model launch and build the sub-pool env.

        Split out of _create_subpool so callers can reserve devices before a
        lengthy preparation phase (model download, virtualenv install).
        allocate_devices/allocate_devices_with_gpu_idx record the reservation
        synchronously in self._gpu_to_model_uids /
        self._user_specified_gpu_to_model_uids. Concurrent launches (up to
        XINFERENCE_MAX_CONCURRENT_LAUNCHES) must see that reservation
        immediately for idle-first placement and multi-replica load
        balancing to work; deferring allocation until after preparation lets
        them all race for the same "idle" GPU snapshot instead.
        """
        env = {} if env is None else env
        devices = []
        env_name = get_available_device_env_name() or "CUDA_VISIBLE_DEVICES"
        if gpu_idx is None:
            if isinstance(n_gpu, int) or (n_gpu == "auto" and gpu_count() > 0):
                # Currently, n_gpu=auto means using 1 GPU
                gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
                devices = self.allocate_devices(model_uid=model_uid, n_gpu=gpu_cnt)
                env[env_name] = ",".join([str(dev) for dev in devices])
                logger.debug(f"GPU selected: {devices} for model {model_uid}")
            if n_gpu is None:
                env[env_name] = "-1"
                logger.debug(f"GPU disabled for model {model_uid}")
        else:
            assert isinstance(gpu_idx, list)
            devices = await self.allocate_devices_with_gpu_idx(
                model_uid, model_type, gpu_idx  # type: ignore
            )
            env[env_name] = ",".join([str(dev) for dev in devices])

        # G: inject model_uid into the sub-pool env dict (not os.environ).
        # os.environ mutation is thread-unsafe under concurrent launches and
        # ineffective (the sub-pool is forked from this env dict). Passing it
        # here lets the sub-pool and its vLLM descendants inherit the tag.
        env["XINFERENCE_MODEL_UID"] = model_uid

        return env, [str(dev) for dev in devices]

    async def _spawn_subpool(
        self,
        model_uid: str,
        env: Dict[str, str],
        devices: List[str],
        start_python: Optional[str] = None,
    ) -> str:
        """Spawn the model sub-pool process for devices already reserved by
        _allocate_subpool_devices."""
        # H2: pre-launch VRAM recheck. _cleanup_gpu_orphans_on_startup runs at
        # worker start, but orphans may appear between startup and this launch
        # (e.g. another worker on the same host died). If free VRAM on the
        # target GPUs is below the ready ratio, scan for PPID==1 vLLM orphans
        # and SIGKILL them, then poll VRAM release. PPID==1 (not the diff
        # heuristic used by _kill_orphan_gpu_pids) avoids mis-killing another
        # live worker's vLLM processes on shared GPUs. Best-effort: any error
        # is logged and the launch proceeds (the timeout below is the backstop).
        _target_device_ints = [int(d) for d in devices] if devices else []
        if _target_device_ints:
            try:
                _free_ratio = _snapshot_gpu_free_ratio(_target_device_ints)
                if 0 <= _free_ratio < _VRAM_READY_RATIO:
                    logger.warning(
                        "Pre-launch VRAM low for model %s on GPUs %s: "
                        "free_ratio=%.2f (target %.2f), scanning for orphans",
                        model_uid,
                        _target_device_ints,
                        _free_ratio,
                        _VRAM_READY_RATIO,
                    )
                    _killed = await _kill_gpu_orphans_by_ppid(
                        _target_device_ints, model_uid=model_uid
                    )
                    if _killed:
                        logger.warning(
                            "Pre-launch killed %d GPU orphan(s) for model %s: %s",
                            len(_killed),
                            model_uid,
                            _killed,
                        )
                        _vram_deadline = time.monotonic() + _VRAM_RECLAIM_TIMEOUT
                        while time.monotonic() < _vram_deadline:
                            await asyncio.sleep(1.0)
                            _free_ratio = _snapshot_gpu_free_ratio(_target_device_ints)
                            if _free_ratio < 0 or _free_ratio >= _VRAM_READY_RATIO:
                                break
                        logger.info(
                            "Pre-launch VRAM after cleanup for model %s: "
                            "free_ratio=%.2f",
                            model_uid,
                            _free_ratio,
                        )
            except Exception:
                logger.debug(
                    "Pre-launch VRAM check failed for model %s",
                    model_uid,
                    exc_info=True,
                )

        # H3 + C: serialize sub-pool creation and bound it with a timeout.
        # H3: append_sub_pool has no internal lock; concurrent fork+exec
        # deadlocks on fork+GIL and blocks the event loop, which also
        # disables the xo.wait_for timeout callback. The asyncio.Lock
        # serializes only append_sub_pool (milliseconds), not download/load.
        # C: a single fork that deadlocks in CUDA init would otherwise hang
        # the actor forever. xo.wait_for raises asyncio.TimeoutError after
        # XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT; we clean up and re-raise so the
        # supervisor's _workers_launching counter decrements.
        logger.debug(
            "Creating subpool for model %s, start_python=%s, env_keys=%s",
            model_uid,
            start_python,
            sorted(env.keys()) if env else [],
        )
        try:
            subpool_address = await self._append_sub_pool_protected(
                env=env, start_python=start_python, model_uid=model_uid
            )
        except asyncio.TimeoutError:
            # Release devices allocated above; otherwise the GPU allocation
            # table leaks and the next launch on the same GPUs reports them
            # as occupied. (Leftover-child kill is handled inside the helper.)
            self.release_devices(model_uid=model_uid)
            raise
        await self._ensure_subpool_monitor()
        return subpool_address

    async def _create_subpool(
        self,
        model_uid: str,
        model_type: Optional[str] = None,
        n_gpu: Optional[Union[int, str]] = "auto",
        gpu_idx: Optional[List[int]] = None,
        env: Optional[Dict[str, str]] = None,
        start_python: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Allocate devices and spawn the sub-pool in one call.

        Convenience wrapper combining _allocate_subpool_devices and
        _spawn_subpool for callers that don't need to split allocation from
        spawning (e.g. launch paths with no intervening preparation phase).
        """
        subpool_env, devices = await self._allocate_subpool_devices(
            model_uid, model_type, n_gpu=n_gpu, gpu_idx=gpu_idx, env=env
        )
        subpool_address = await self._spawn_subpool(
            model_uid, subpool_env, devices, start_python=start_python
        )
        return subpool_address, devices

    def _check_model_is_valid(self, model_name: str, model_format: Optional[str]):
        # baichuan-base and baichuan-chat depend on `cpm_kernels` module,
        # but `cpm_kernels` cannot run on Darwin system.
        if platform.system() == "Darwin" and model_format == "pytorch":
            if "baichuan" in model_name:
                raise ValueError(f"{model_name} model can't run on Darwin system.")

    @log_sync(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        # TODO: centralized model registrations
        if model_type in self._custom_register_type_to_cls:
            (
                model_spec_cls,
                register_fn,
                unregister_fn,
                generate_fn,
            ) = self._custom_register_type_to_cls[model_type]
            model_spec = model_spec_cls.parse_raw(model)
            try:
                register_fn(model_spec, persist)
            except ValueError as e:
                raise e
            except Exception as e:
                unregister_fn(model_spec.model_name, raise_error=False)
                raise e

            # cache sync is an auxiliary feature; failure must not roll back
            # the already-successful register_fn. Guard against _cache_tracker_ref
            # being None when get_supervisor_ref's core init failed (defensive).
            try:
                if self._cache_tracker_ref is not None:
                    await self._cache_tracker_ref.record_model_version(
                        generate_fn(model_spec), self.address
                    )
            except Exception:
                logger.warning(
                    "Failed to record model version for %s after registration; "
                    "cache management may be degraded",
                    model_spec.model_name,
                    exc_info=True,
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        # TODO: centralized model registrations
        if model_type in self._custom_register_type_to_cls:
            _, _, unregister_fn, _ = self._custom_register_type_to_cls[model_type]
            unregister_fn(model_name, False)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def update_model_type(self, model_type: str):
        """
        Update model configurations for a specific model type by downloading
        the latest JSON from the remote API and storing it locally.

        Args:
            model_type: Type of model (LLM, embedding, image, etc.)
        """
        import json

        import requests

        supported_types = list(self._custom_register_type_to_cls.keys())

        normalized_for_validation = model_type
        if model_type.lower() == "llm" and "LLM" in supported_types:
            normalized_for_validation = "LLM"
        elif model_type.lower() == "llm" and "llm" in supported_types:
            normalized_for_validation = "llm"

        if normalized_for_validation not in supported_types:
            logger.error(f"Unsupported model type: {normalized_for_validation}")
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Supported types are: {', '.join(supported_types)}"
            )

        # Construct the URL to download JSON
        url = (
            "https://model.xinference.io/api/models/download"
            f"?model_type={model_type.lower()}"
        )

        try:
            # Download JSON from remote API. Run the blocking request in a
            # worker thread so it does not freeze the actor's event loop.
            response = await asyncio.to_thread(requests.get, url, timeout=30)
            response.raise_for_status()

            # Parse JSON response
            model_data = response.json()

            # Store the JSON data using CacheManager as built-in models
            await self._store_complete_model_configurations(model_type, model_data)

            # Dynamically reload built-in models to make them immediately available
            try:
                if model_type.lower() == "llm":
                    from ..model.llm import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "embedding":
                    from ..model.embedding import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "audio":
                    from ..model.audio import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "image":
                    from ..model.image import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "rerank":
                    from ..model.rerank import register_builtin_model

                    register_builtin_model()
                elif model_type.lower() == "video":
                    from ..model.video import register_builtin_model

                    register_builtin_model()
                else:
                    logger.warning(
                        f"No dynamic loading available for model type: {model_type}"
                    )
            except Exception as reload_error:
                logger.error(
                    f"Error reloading built-in models: {reload_error}",
                    exc_info=True,
                )
                # Don't fail the update if reload fails, just log the error

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading model configurations: {e}")
            raise ValueError(f"Failed to download model configurations: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON response from remote API: {str(e)}")
        except Exception as e:
            logger.error(
                f"Unexpected error during model update: {e}",
                exc_info=True,
            )
            raise ValueError(f"Failed to update model configurations: {str(e)}")

    async def _store_complete_model_configurations(self, model_type: str, model_data):
        """
        Store complete model configurations as a unified JSON file.
        This is used by update_model_type to preserve the original JSON structure.

        Args:
            model_type: Type of model (as provided by user, e.g., "llm")
            model_data: JSON data containing model configurations (complete array)
        """
        import json

        from ..constants import XINFERENCE_MODEL_DIR

        try:
            model_type_lower = model_type.lower()

            # Use the unified JSON file path (same as original update_model_type logic)
            builtin_dir = os.path.join(
                XINFERENCE_MODEL_DIR, "v2", "builtin", model_type_lower
            )
            json_file_path = os.path.join(
                builtin_dir, f"{model_type_lower}_models.json"
            )

            # Ensure directory exists
            os.makedirs(builtin_dir, exist_ok=True)

            # Store the complete JSON file (preserving original structure)
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(
                f"Error storing complete model configurations: {str(e)}",
                exc_info=True,
            )
            raise ValueError(f"Failed to store complete model configurations: {str(e)}")

    @log_async(logger=logger)
    async def list_model_registrations(
        self, model_type: str, detailed: bool = False
    ) -> List[Dict[str, Any]]:
        def sort_helper(item):
            assert isinstance(item["model_name"], str)
            return item.get("model_name").lower()

        ret = []

        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families
            from ..model.llm.cache_manager import LLMCacheManager

            # Add built-in LLM families
            for family in BUILTIN_LLM_FAMILIES:
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        family, LLMCacheManager
                    )
                    ret.append(
                        {
                            **family.dict(),
                            "model_specs": specs,
                            "is_builtin": True,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": True})

            # Add user-defined LLM families
            for family in get_user_defined_llm_families():
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        family, LLMCacheManager
                    )
                    ret.append(
                        {
                            **family.dict(),
                            "model_specs": specs,
                            "is_builtin": False,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append({"model_name": family.model_name, "is_builtin": False})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.cache_manager import EmbeddingCacheManager
            from ..model.embedding.custom import get_user_defined_embeddings

            # Add built-in embedding models
            for model_name, family_list in BUILTIN_EMBEDDING_MODELS.items():
                for family in family_list:
                    if detailed:
                        specs, download_hubs = self._get_spec_dicts_with_cache_status(
                            family, EmbeddingCacheManager
                        )
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined embedding models
            for model_spec in get_user_defined_embeddings():
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        model_spec, EmbeddingCacheManager
                    )
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": specs,
                            "is_builtin": False,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.cache_manager import ImageCacheManager
            from ..model.image.custom import get_user_defined_images

            # Add built-in image models (BUILTIN_IMAGE_MODELS contains model_name -> families list)
            for model_name, families in BUILTIN_IMAGE_MODELS.items():
                download_hubs = []
                for family in families:
                    if family.model_hub not in download_hubs:
                        download_hubs.append(family.model_hub)
                for family in families:
                    if detailed:
                        cache_manager = ImageCacheManager(family)
                        model_specs = [
                            {
                                "model_format": "pytorch",
                                "model_hub": family.model_hub,
                                "model_id": family.model_id,
                                "cache_status": cache_manager.get_cache_status(),
                            }
                        ]
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": model_specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined image models
            for model_spec in get_user_defined_images():
                if detailed:
                    cache_manager = ImageCacheManager(model_spec)
                    model_specs = [
                        {
                            "model_format": "pytorch",
                            "model_hub": model_spec.model_hub,
                            "model_id": model_spec.model_id,
                            "cache_status": cache_manager.get_cache_status(),
                        }
                    ]
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": model_specs,
                            "is_builtin": False,
                            "download_hubs": [model_spec.model_hub],
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios
            from ..model.cache_manager import CacheManager

            # Add built-in audio models (BUILTIN_AUDIO_MODELS contains model_name -> families list)
            for model_name, families in BUILTIN_AUDIO_MODELS.items():
                download_hubs = []
                for family in families:
                    if family.model_hub not in download_hubs:
                        download_hubs.append(family.model_hub)
                for family in families:
                    if detailed:
                        audio_cache_manager = CacheManager(family)
                        model_specs = [
                            {
                                "model_format": "pytorch",
                                "model_hub": family.model_hub,
                                "model_id": family.model_id,
                                "cache_status": audio_cache_manager.get_cache_status(),
                            }
                        ]
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": model_specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined audio models
            for model_spec in get_user_defined_audios():
                if detailed:
                    audio_cache_manager = CacheManager(model_spec)
                    model_specs = [
                        {
                            "model_format": "pytorch",
                            "model_hub": model_spec.model_hub,
                            "model_id": model_spec.model_id,
                            "cache_status": audio_cache_manager.get_cache_status(),
                        }
                    ]
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": model_specs,
                            "is_builtin": False,
                            "download_hubs": [model_spec.model_hub],
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "video":
            from ..model.cache_manager import CacheManager
            from ..model.video import BUILTIN_VIDEO_MODELS

            # Add built-in video models (BUILTIN_VIDEO_MODELS contains model_name -> families list)
            for model_name, families in BUILTIN_VIDEO_MODELS.items():
                download_hubs = []
                for family in families:
                    if family.model_hub not in download_hubs:
                        download_hubs.append(family.model_hub)
                for family in families:
                    if detailed:
                        video_cache_manager = CacheManager(family)
                        model_specs = [
                            {
                                "model_format": "pytorch",
                                "model_hub": family.model_hub,
                                "model_id": family.model_id,
                                "cache_status": video_cache_manager.get_cache_status(),
                                "gguf_model_id": family.gguf_model_id,
                                "gguf_quantizations": family.gguf_quantizations,
                                "gguf_model_file_name_template": (
                                    family.gguf_model_file_name_template
                                ),
                            }
                        ]
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": model_specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.cache_manager import RerankCacheManager
            from ..model.rerank.custom import get_user_defined_reranks

            # Add built-in rerank models (BUILTIN_RERANK_MODELS contains model_name -> family_list list)
            for model_name, family_list in BUILTIN_RERANK_MODELS.items():
                for family in family_list:
                    if detailed:
                        specs, download_hubs = self._get_spec_dicts_with_cache_status(
                            family, RerankCacheManager
                        )
                        ret.append(
                            {
                                **family.dict(),
                                "model_specs": specs,
                                "is_builtin": True,
                                "download_hubs": download_hubs,
                            }
                        )
                    else:
                        ret.append({"model_name": model_name, "is_builtin": True})

            # Add user-defined rerank models
            for model_spec in get_user_defined_reranks():
                if detailed:
                    specs, download_hubs = self._get_spec_dicts_with_cache_status(
                        model_spec, RerankCacheManager
                    )
                    ret.append(
                        {
                            **model_spec.dict(),
                            "model_specs": specs,
                            "is_builtin": False,
                            "download_hubs": download_hubs,
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        elif model_type == "flexible":
            from ..model.flexible.custom import get_flexible_models

            ret = []

            for model_spec in get_flexible_models():
                if detailed:
                    ret.append(
                        {
                            **model_spec.dict(),
                            "cache_status": True,
                            "is_builtin": False,
                        }
                    )
                else:
                    ret.append(
                        {"model_name": model_spec.model_name, "is_builtin": False}
                    )

            ret.sort(key=sort_helper)
            return ret
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def get_model_registration(self, model_type: str, model_name: str) -> Any:
        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            # Check built-in LLM families
            for f in BUILTIN_LLM_FAMILIES:
                if f.model_name == model_name:
                    return f

            # Check user-defined LLM families
            for f in get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f
        elif model_type == "embedding":
            from ..model.embedding import BUILTIN_EMBEDDING_MODELS
            from ..model.embedding.custom import get_user_defined_embeddings

            # Check built-in embedding models
            for builtin_model_name, family_list in BUILTIN_EMBEDDING_MODELS.items():
                if builtin_model_name != model_name:
                    continue
                for family in family_list:
                    return self._prefer_model_hub(family)

            # Check user-defined embedding models
            for f in get_user_defined_embeddings():
                if f.model_name == model_name:
                    return self._prefer_model_hub(f)
        elif model_type == "image":
            from ..model.image import BUILTIN_IMAGE_MODELS
            from ..model.image.custom import get_user_defined_images

            # Check built-in image models
            if model_name in BUILTIN_IMAGE_MODELS:
                families = BUILTIN_IMAGE_MODELS[model_name]
                for f in families:
                    if f.model_hub == "huggingface":
                        return f

            # Check user-defined image models
            for f in get_user_defined_images():
                if f.model_name == model_name:
                    return f
        elif model_type == "audio":
            from ..model.audio import BUILTIN_AUDIO_MODELS
            from ..model.audio.custom import get_user_defined_audios

            # Check built-in audio models
            if model_name in BUILTIN_AUDIO_MODELS:
                families = BUILTIN_AUDIO_MODELS[model_name]
                for f in families:
                    if f.model_hub == "huggingface":
                        return f

            # Check user-defined audio models
            for f in get_user_defined_audios():
                if f.model_name == model_name:
                    return f
        elif model_type == "video":
            from ..model.video import BUILTIN_VIDEO_MODELS

            # Check built-in video models
            if model_name in BUILTIN_VIDEO_MODELS:
                families = BUILTIN_VIDEO_MODELS[model_name]
                for f in families:
                    if f.model_hub == "huggingface":
                        return f
            return None
        elif model_type == "rerank":
            from ..model.rerank import BUILTIN_RERANK_MODELS
            from ..model.rerank.custom import get_user_defined_reranks

            # Check built-in rerank models
            if model_name in BUILTIN_RERANK_MODELS:
                family_list = BUILTIN_RERANK_MODELS[model_name]
                for f in family_list:
                    return self._prefer_model_hub(f)

            # Check user-defined rerank models
            for f in get_user_defined_reranks():
                if f.model_name == model_name:
                    return self._prefer_model_hub(f)
        elif model_type == "flexible":
            from ..model.flexible import get_flexible_models

            for f in get_flexible_models():
                if f.model_name == model_name:
                    return f
        return None

    @log_async(logger=logger)
    async def query_engines_by_model_name(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        enable_virtual_env: Optional[bool] = None,
    ):
        if enable_virtual_env is None:
            enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV
        if enable_virtual_env:
            return get_engine_params_by_name_with_virtual_env(
                model_type, model_name, enable_virtual_env=enable_virtual_env
            )
        return get_engine_params_by_name(
            model_type, model_name, enable_virtual_env=enable_virtual_env
        )

    async def _get_model_ability(self, model: Any, model_type: str) -> List[str]:
        from ..model.llm.core import LLM

        ability_map = {
            "embedding": ["embed"],
            "rerank": ["rerank"],
            "flexible": ["flexible"],
        }
        if model_type in ability_map:
            return ability_map[model_type]
        if model_type in {"image", "audio", "video"}:
            return model.model_ability
        assert model_type == "LLM"
        assert isinstance(model, LLM)
        return model.model_family.model_ability  # type: ignore

    async def update_cache_status(self, model_name: str, version_info: Any):
        if isinstance(version_info, list):  # image model
            model_path = version_info[0]["model_file_location"]
            await self._cache_tracker_ref.update_cache_status(
                self.address, model_name, None, model_path
            )
        else:
            await self._cache_tracker_ref.update_cache_status(
                self.address,
                model_name,
                version_info["model_version"],
                version_info["model_file_location"],
            )

    @classmethod
    def _create_virtual_env_manager(
        cls,
        enable_virtual_env: Optional[bool],
        virtual_env_name: Optional[str],
        env_path: str,
    ) -> Optional[VirtualEnvManager]:
        if enable_virtual_env is None:
            enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV

        if not enable_virtual_env:
            # skip preparing virtualenv
            return None

        from xoscar.virtualenv import get_virtual_env_manager

        with _exclusive_venv_path_lock(env_path):
            virtual_env_manager: VirtualEnvManager = get_virtual_env_manager(
                virtual_env_name or "uv", env_path
            )
            # create env
            python_path = None
            if not hasattr(sys, "_MEIPASS"):
                # not in pyinstaller
                # we specify python_path explicitly
                # sometimes uv would find other versions.
                python_path = pathlib.Path(sys.executable)
            virtual_env_manager.create_env(python_path=python_path)

            import site as _site
            import sysconfig as _sysconfig

            if not hasattr(sys, "_MEIPASS"):
                # Normal execution (pip, venv, conda, source, Docker).
                # Inject parent site-packages via .pth so xinference and xoscar are
                # discoverable in the child venv while preserving child-venv isolation.
                # .pth paths are appended AFTER child site-packages so child-installed
                # packages always take precedence over parent ones.
                parent_site_packages = _sysconfig.get_paths()["purelib"]

                # Warn if xinference appears to be user-installed — child venvs
                # cannot see user site-packages (~/.local/lib/...) by design.
                user_site = (
                    _site.getusersitepackages()
                    if hasattr(_site, "getusersitepackages")
                    else None
                )
                if (
                    user_site
                    and not os.path.exists(
                        os.path.join(parent_site_packages, "xinference")
                    )
                    and os.path.exists(os.path.join(user_site, "xinference"))
                ):
                    logger.warning(
                        "xinference is installed in user site-packages (%s) which is "
                        "not visible to child venvs. Re-install inside a virtual "
                        "environment.",
                        user_site,
                    )

                if os.path.exists(parent_site_packages):
                    child_site_packages = pathlib.Path(
                        virtual_env_manager.get_lib_path()
                    )
                    child_site_packages.mkdir(parents=True, exist_ok=True)
                    pth_file = child_site_packages / "_xinference_parent.pth"
                    desired_content = parent_site_packages + "\n"
                    # Avoid truncate race when multiple replicas write the
                    # same .pth concurrently: skip if content already correct,
                    # otherwise atomic-write via a temp file + os.replace.
                    needs_write = True
                    try:
                        if (
                            pth_file.exists()
                            and pth_file.read_text() == desired_content
                        ):
                            needs_write = False
                    except OSError:
                        pass
                    if needs_write:
                        tmp_file = pth_file.with_suffix(".pth.tmp")
                        try:
                            tmp_file.write_text(desired_content)
                            os.replace(str(tmp_file), str(pth_file))
                        except OSError:
                            # Fallback: direct write (still better than no .pth)
                            pth_file.write_text(desired_content)
                        logger.debug(
                            "Injected parent site-packages into child venv "
                            "via %s -> %s",
                            pth_file,
                            parent_site_packages,
                        )
                    else:
                        logger.debug(
                            "Skipped .pth write (content unchanged): %s",
                            pth_file,
                        )
                else:
                    logger.warning(
                        "Parent site-packages path does not exist: %s — child venv "
                        "may not be able to import xinference or xoscar.",
                        parent_site_packages,
                    )
            else:
                # PyInstaller bundle: sys._MEIPASS is a private temp directory
                # belonging to the bundle process. The child venv runs an external
                # Python interpreter that has no access to that directory.
                # Skip .pth injection entirely — xinference in bundle mode manages
                # its own package visibility through the bundle mechanism.
                logger.debug(
                    "Running inside PyInstaller bundle — skipping parent site-packages "
                    "injection into child venv."
                )

            return virtual_env_manager

    @staticmethod
    def _is_cuda_device_available() -> bool:
        """
        Whether a usable CUDA device is actually present.

        ``get_cuda_version()`` reports ``torch.version.cuda`` (the version the
        installed PyTorch was built against), which is set even on a CPU-only
        host with a CUDA-built torch. Selecting a GPU wheel off that alone
        installs a wheel whose ``libcuda.so.1`` cannot be loaded at import time.
        Gate the GPU path on real device availability instead.
        """
        try:
            from xoscar.virtualenv.platform import check_cuda_available

            return bool(check_cuda_available())
        except Exception:
            try:
                import torch

                return bool(torch.cuda.is_available())
            except Exception:
                return False

    @staticmethod
    def _uninstall_venv_package(
        virtual_env_manager: "VirtualEnvManager", package: str
    ) -> None:
        """
        Uninstall a package from the virtual environment, if present.

        Used to force a fresh install of a package whose currently-installed
        build must be replaced (e.g. swapping a CPU xllamacpp wheel for the GPU
        wheel, which shares the same version number). Failures are logged and
        swallowed so a missing package or uninstall hiccup does not abort the
        launch.
        """
        import subprocess

        from .virtual_env_manager import resolve_virtualenv_python_path

        venv_python = resolve_virtualenv_python_path(virtual_env_manager)
        if not venv_python:
            return
        try:
            subprocess.run(
                [venv_python, "-m", "pip", "uninstall", "-y", package],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as e:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to uninstall %s from virtual env: %s", package, e)

    @staticmethod
    def _resolve_virtualenv_model_format(
        model: Any, requested_model_format: Optional[str]
    ) -> Optional[str]:
        """Return the concrete format selected while creating ``model``."""
        model_family = getattr(model, "model_family", None)
        model_specs = getattr(model_family, "model_specs", None)
        if model_specs:
            resolved_model_format = getattr(model_specs[0], "model_format", None)
            if resolved_model_format:
                return resolved_model_format

        model_spec = getattr(model, "model_spec", None)
        resolved_model_format = getattr(model_spec, "model_format", None)
        return resolved_model_format or requested_model_format

    @classmethod
    def _prepare_virtual_env(
        cls,
        virtual_env_manager: "VirtualEnvManager",
        settings: Optional[VirtualEnvSettings],
        virtual_env_packages: Optional[List[str]],
        model_engine: Optional[str],
        model_name: Optional[str] = None,
        architectures: Optional[List[str]] = None,
        model_format: Optional[str] = None,
    ):
        engine_defaults = get_engine_model_format_virtualenv_packages(
            model_engine, model_format
        )
        if (
            (not settings or not settings.packages)
            and not virtual_env_packages
            and not engine_defaults
        ):
            # no settings or no packages
            return

        if settings is None:
            settings = VirtualEnvSettings(packages=virtual_env_packages or [])

        assert settings is not None  # for mypy type narrowing

        if settings and model_engine and model_engine.lower() not in ("vllm", "sglang"):
            # Pydantic v1 compatibility: use copy() when model_copy is unavailable.
            if hasattr(settings, "model_copy"):
                settings = settings.model_copy(deep=True)
            else:
                settings = settings.copy(deep=True)
            assert settings is not None  # for mypy type narrowing after copy
            settings.extra_index_url = None
            settings.index_strategy = None

        if settings.inherit_pip_config:
            # inherit pip config
            pip_config = get_pip_config_args()
            for k, v in pip_config.items():
                if hasattr(settings, k) and not getattr(settings, k):
                    setattr(settings, k, v)

        # An extra index present at this point was configured explicitly — by
        # the model spec or inherited pip config (e.g. an offline/private
        # mirror) — as opposed to the engine defaults applied below.
        user_configured_extra_index = settings.extra_index_url is not None

        apply_engine_virtualenv_settings(settings, model_engine)

        base_packages = engine_defaults
        if settings.packages:
            base_packages = base_packages + settings.packages.copy()
        base_packages = expand_engine_dependency_placeholders(
            base_packages, model_engine
        )
        packages = merge_virtual_env_packages(base_packages, virtual_env_packages)

        # Auto-configure PyTorch wheel URL based on system packages
        # Check if packages contain PyTorch system markers (#system_torch#, etc.)
        # If so, detect CUDA version from system and configure wheel URL
        # Note: markers are kept as-is and resolved later by xoscar's process_packages
        from .virtual_env_manager import PYTORCH_CUDA_WHEEL_URLS, PYTORCH_PACKAGES

        system_cuda_urls = None
        for pkg in packages:
            if pkg.startswith("#system_") and pkg.endswith("#"):
                # Extract package name from marker
                marker_pkg = pkg[len("#system_") : -1].lower()
                if marker_pkg in PYTORCH_PACKAGES:
                    try:
                        import importlib.metadata

                        version = importlib.metadata.version(marker_pkg)
                        # Extract CUDA version from version string (e.g., "2.5.0+cu121" -> "cu121")
                        if "+" in version:
                            _, suffix = version.split("+", 1)
                            if suffix.startswith("cu") or suffix.startswith("rocm"):
                                wheel_url = PYTORCH_CUDA_WHEEL_URLS.get(suffix)
                                if wheel_url:
                                    system_cuda_urls = [wheel_url]
                                    logger.info(
                                        f"Auto-configuring PyTorch wheel URL for CUDA {suffix}: {wheel_url}"
                                    )
                                    break
                    except importlib.metadata.PackageNotFoundError:
                        # Package not installed, skip - will be resolved during install
                        pass

        # Add PyTorch wheel URL if detected from system packages
        if system_cuda_urls:
            if settings.extra_index_url is None:
                settings.extra_index_url = system_cuda_urls
            elif user_configured_extra_index:
                # An explicitly configured extra index (model spec or inherited
                # pip config, e.g. an offline/private mirror) stays
                # authoritative: uv treats an unreachable extra index as fatal
                # instead of falling back, so forcing the public CUDA wheel
                # index here would break air-gapped deployments even when the
                # wheels exist on the private index.
                logger.info(
                    "Skipping auto-configured PyTorch wheel URL %s: explicitly "
                    "configured extra index takes precedence: %s",
                    system_cuda_urls,
                    settings.extra_index_url,
                )
            else:
                # Merge with existing extra_index_url, system URLs first for priority
                existing_urls = (
                    settings.extra_index_url
                    if isinstance(settings.extra_index_url, list)
                    else [settings.extra_index_url]
                )
                settings.extra_index_url = system_cuda_urls + [
                    u for u in existing_urls if u not in system_cuda_urls
                ]

        try:
            from xoscar.virtualenv.platform import get_cuda_version

            cuda_version = get_cuda_version()
        except Exception:
            cuda_version = None

        if not is_cuda_compatible(settings.extra_index_url, cuda_version):
            logger.debug(
                f"[DEBUG] CUDA version mismatch: cuda_version={cuda_version}, extra_index_url={settings.extra_index_url}, clearing extra_index_url and index_strategy"
            )
            settings.extra_index_url = None
            settings.index_strategy = None
        else:
            logger.debug(
                f"[DEBUG] CUDA version check passed: cuda_version={cuda_version}, keeping settings.extra_index_url={settings.extra_index_url}"
            )

        # For the llama.cpp engine, xllamacpp ships CPU wheels on PyPI and GPU
        # wheels on a self-hosted per-CUDA index. When a compatible CUDA runtime
        # is detected, install from the matching GPU index so the GPU build is
        # pulled instead of the default CPU build.
        #
        # The GPU index is used exclusively: it becomes the sole index_url and
        # any inherited index/extra-index/find-links (e.g. a Tencent/Tsinghua
        # PyPI mirror pulled in via inherit_pip_config) is dropped for this
        # install. This is required for correctness: PyPI (and its mirrors)
        # only carry the CPU build of xllamacpp, and the CPU and GPU wheels
        # share the same version number, so leaving a mirror in the resolution
        # set lets the resolver satisfy "xllamacpp" with the CPU wheel -- the
        # user then gets a CPU runtime while believing the GPU build was
        # installed. Restricting to the GPU index guarantees the GPU wheel (or a
        # visible failure). xllamacpp GPU wheels are self-contained abi3 wheels
        # with no required runtime dependencies, so an exclusive index does not
        # break resolution.
        force_reinstall_xllamacpp = False
        if model_engine and model_engine.lower() == "llama.cpp":
            from .virtual_env_manager import get_xllamacpp_cuda_index_url

            xllamacpp_index_url = get_xllamacpp_cuda_index_url(cuda_version)
            # Only switch to the GPU wheel when a CUDA device is actually usable.
            # cuda_version reflects the PyTorch build, not device availability,
            # so a CPU-only host with a CUDA-built torch would otherwise get a
            # GPU wheel that fails to import (missing libcuda.so.1). In that
            # case keep the default CPU index and behavior.
            cuda_device_available = bool(
                xllamacpp_index_url and cls._is_cuda_device_available()
            )
            if XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL and cuda_device_available:
                logger.warning(
                    "Explicit offline-install mode cannot use the xllamacpp "
                    "GPU wheel index %s; installing the CPU build from the "
                    "configured private index instead. Preinstall the matching "
                    "GPU wheel in a custom runtime image to retain llama.cpp "
                    "GPU acceleration offline.",
                    xllamacpp_index_url,
                )
            elif xllamacpp_index_url and cuda_device_available:
                logger.info(
                    "Detected CUDA %s, installing GPU build of xllamacpp "
                    "exclusively from %s",
                    cuda_version,
                    xllamacpp_index_url,
                )
                settings.index_url = xllamacpp_index_url
                settings.extra_index_url = None
                settings.find_links = None
                settings.index_strategy = None
                # A CPU build of xllamacpp may already satisfy the
                # "xllamacpp>=..." requirement (e.g. inherited from the parent
                # env, or installed on a previous CPU-only launch). Because the
                # CPU and GPU wheels share the same version, the skip-installed
                # filter would drop the requirement before uv ever sees the GPU
                # index, leaving the CPU build in place. Uninstall it first and
                # force this install so the GPU wheel actually replaces it.
                force_reinstall_xllamacpp = True

        packages = filter_virtualenv_packages_by_markers(
            packages, model_engine, cuda_version
        )

        critical_specs = get_engine_critical_dependency_specs(model_engine, packages)
        if critical_specs:
            logger.info(
                "Engine %s will be inherited from the parent environment whose "
                "copies of its critical dependencies do not satisfy the "
                "engine's declared requirements; installing %s into the venv",
                model_engine,
                critical_specs,
            )
            packages = packages + critical_specs

        if XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL:
            if not settings.index_url:
                raise ValueError(
                    "XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL=1 requires a "
                    "private index_url (configure the offline pip.conf)"
                )
            # This explicit flag distinguishes the bundled offline mirror from
            # an ordinary user-configured PyPI proxy. Rewriting merely because
            # index_url is present breaks online users whose mirror does not
            # carry the direct wheel.
            packages = rewrite_direct_url_packages_for_index(packages)
            direct_references = find_direct_reference_packages(packages)
            if direct_references:
                raise ValueError(
                    "Offline virtualenv installation does not support "
                    "non-wheel direct references; preinstall or replace these "
                    f"requirements: {direct_references}"
                )

        conf = dict(settings)
        conf.pop("packages", None)
        conf.pop("inherit_pip_config", None)
        if XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED:
            conf["skip_installed"] = XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED
        if force_reinstall_xllamacpp:
            # Bypass the satisfied-package filter so uv is actually invoked with
            # the GPU index even when a same-version CPU wheel is already
            # present.
            conf["skip_installed"] = False
        variables = {}
        if model_engine:
            engine_value = model_engine.lower()
            variables["engine"] = engine_value
            variables["model_engine"] = engine_value

        logger.info(
            "Installing packages %s in virtual env %s, with settings(%s)",
            packages,
            virtual_env_manager.env_path,
            ", ".join([f"{k}={v}" for k, v in conf.items() if v]),
        )
        with _exclusive_venv_path_lock(str(virtual_env_manager.env_path)):
            if force_reinstall_xllamacpp:
                cls._uninstall_venv_package(virtual_env_manager, "xllamacpp")
            virtual_env_manager.install_packages(packages, **conf, **variables)

            # Post-install: flashinfer AOT workaround for sm_120 Blackwell.
            # vllm 0.21.0 hard-pins flashinfer-cubin==0.6.8.post1 which has JIT
            # compilation failure on sm_120. Force-upgrade to AOT versions.
            # Run under the same lock — uv pip install mutates the venv and
            # must stay serialized with install_packages() and other AOT
            # upgrades when multiple replicas/workers share this venv.
            # See optimize/20260702/2026070209.md
            from .virtual_env_manager import apply_flashinfer_aot_post_install

            if XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL:
                logger.info(
                    "Skipping the FlashInfer AOT post-install from its public "
                    "wheel index in explicit offline-install mode"
                )
            else:
                apply_flashinfer_aot_post_install(
                    model_engine,
                    architectures,
                    virtual_env_manager,
                    conf,
                    cuda_version,
                )

        # Apply engine-specific post-install patches
        if model_engine and model_engine.lower() == "vllm":
            try:
                from xinference.model.llm.vllm.patches import apply_vllm_patches
            except ImportError:
                pass
            else:
                apply_vllm_patches(
                    env_path=str(virtual_env_manager.env_path),
                    model_name=model_name,
                    architectures=architectures,
                )

    async def _get_progressor(self, request_id: str):
        from .progress_tracker import Progressor, ProgressTrackerActor

        progress_tracker_ref = self._progress_tracker_ref
        if progress_tracker_ref is None:
            progress_tracker_ref = self._progress_tracker_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=ProgressTrackerActor.default_uid()
            )

        progressor = Progressor(
            request_id,
            progress_tracker_ref,
            asyncio.get_running_loop(),
        )
        await progressor.start()
        progressor.set_progress(0.0, "start to launch model")
        return progressor

    @classmethod
    def _upload_download_progress(
        cls, progressor: "Progressor", downloader: CancellableDownloader
    ):
        while not downloader.done:
            progress = downloader.get_progress()
            progressor.set_progress(progress)
            downloader.wait(1)

        progressor.set_progress(1.0, "Start to load model")

    @log_async(logger=logger, level=logging.INFO)
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[Union[int, str]],
        model_format: Optional[str],
        quantization: Optional[str],
        model_engine: Optional[str],
        model_type: str = "LLM",
        n_gpu: Optional[Union[int, str]] = "auto",
        n_worker: Optional[int] = 1,
        shard: Optional[int] = 0,
        driver_info: Optional[dict] = None,
        peft_model_config: Optional[PeftModelConfig] = None,
        request_limits: Optional[int] = None,
        gpu_idx: Optional[Union[int, List[int]]] = None,
        download_hub: Optional[
            Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
        ] = None,
        model_path: Optional[str] = None,
        enable_virtual_env: Optional[bool] = None,
        virtual_env_packages: Optional[List[str]] = None,
        envs: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        # !!! Note that The following code must be placed at the very beginning of this function,
        # or there will be problems with auto-recovery.
        # Because `locals()` will collect all the local parameters of this function and pass to this function again.
        launch_args = locals()
        launch_args.pop("self")
        launch_args.pop("kwargs")
        launch_args.update(kwargs)
        launch_args["launch_ts"] = int(time.time())

        try:
            origin_uid, _ = parse_replica_model_uid(model_uid)
        except Exception as e:
            logger.exception(e)
            raise
        try:
            _ = await self.get_supervisor_ref()
            if self._event_collector_ref is not None:
                await self._event_collector_ref.report_event(
                    origin_uid,
                    Event(
                        event_type=EventType.INFO,
                        event_ts=int(time.time()),
                        event_content="Launch model",
                    ),
                )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e), exc_info=True)

        if gpu_idx is not None:
            logger.info(
                f"You specify to launch the model: {model_name} on GPU index: {gpu_idx} "
                f"of the worker: {self.address}, "
                f"xinference will automatically ignore the `n_gpu` option."
            )
            if isinstance(gpu_idx, int):
                gpu_idx = [gpu_idx]
            assert isinstance(gpu_idx, list)

        if n_gpu is not None:
            if isinstance(n_gpu, int) and (n_gpu <= 0 or n_gpu > gpu_count()):
                raise ValueError(
                    f"The parameter `n_gpu` must be greater than 0 and "
                    f"not greater than the number of GPUs: {gpu_count()} on the machine."
                )
            if isinstance(n_gpu, str) and n_gpu != "auto":
                raise ValueError("Currently `n_gpu` only supports `auto`.")

        device = kwargs.get("device")
        if device and device.lower().startswith("cpu"):
            n_gpu = None

        if peft_model_config is not None:
            if model_type in ("embedding", "rerank"):
                raise ValueError(
                    f"PEFT adaptors cannot be applied to embedding or rerank models."
                )
            if model_type == "LLM" and model_format in ("ggufv2",):
                raise ValueError(
                    f"PEFT adaptors can only be applied to pytorch-like models"
                )
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    f"Invalid input. `model_path`: {model_path} File or directory does not exist."
                )

        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name, model_format)

        if self.get_model_launch_status(model_uid) is not None:
            raise ValueError(f"{model_uid} is running")

        _was_queued = False
        try:
            self._model_uid_launching_guard[model_uid] = launch_info = LaunchInfo()

            # Launch concurrency control: queue if semaphore is full
            if self._launch_semaphore.locked():
                self._launch_waiting += 1
                logger.info(
                    "Launch queued: model_name=%s, model_uid=%s (active: %d/%d, queued: %d)",
                    model_name,
                    model_uid,
                    self._launch_active,
                    XINFERENCE_MAX_CONCURRENT_LAUNCHES,
                    self._launch_waiting,
                )
                _was_queued = True

            async with self._launch_semaphore:
                if _was_queued:
                    self._launch_waiting -= 1
                    _was_queued = False
                self._launch_active += 1
                logger.info(
                    "Launch started: model_name=%s, model_uid=%s (active: %d/%d, queued: %d)",
                    model_name,
                    model_uid,
                    self._launch_active,
                    XINFERENCE_MAX_CONCURRENT_LAUNCHES,
                    self._launch_waiting,
                )
                try:
                    # Check if cancelled while waiting in queue
                    if launch_info.cancel_event.is_set():
                        raise RuntimeError(
                            f"Launch cancelled while waiting in queue: {model_uid}"
                        )

                    # virtualenv
                    virtual_env_name = kwargs.pop("virtual_env_name", None)
                    # Use v4 structure: .xinference/virtualenv/v4/model_name/model_engine/python_version
                    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                    engine_name = (model_engine or "default").lower()
                    virtual_env_path = os.path.join(
                        XINFERENCE_VIRTUAL_ENV_DIR,
                        "v4",
                        model_name,
                        engine_name,
                        python_version,
                    )
                    virtual_env_manager = await asyncio.to_thread(
                        self._create_virtual_env_manager,
                        enable_virtual_env,
                        virtual_env_name,
                        virtual_env_path,
                    )
                    subpool_python_path = resolve_virtualenv_python_path(
                        virtual_env_manager
                    )
                    subpool_envs = build_subpool_envs_for_virtual_env(
                        envs, enable_virtual_env, virtual_env_manager
                    )
                    # Reserve devices now (before download/virtualenv
                    # install): allocate_devices/allocate_devices_with_gpu_idx
                    # record the reservation synchronously, so concurrent
                    # launches (up to XINFERENCE_MAX_CONCURRENT_LAUNCHES) see
                    # each other's picks for idle-first placement and
                    # multi-replica load balancing instead of all racing to
                    # allocate off the same stale snapshot once their prep
                    # phase finishes.
                    subpool_alloc_env, devices = await self._allocate_subpool_devices(
                        model_uid,
                        model_type,
                        n_gpu=n_gpu,
                        gpu_idx=gpu_idx,
                        env=subpool_envs,
                    )
                    # The model subprocess itself must not be spawned before
                    # the virtualenv is populated: the subprocess boots on
                    # the venv's python, and base libraries imported during
                    # its bootstrap (numpy, and torch via xinference's own
                    # import chain) are cached in sys.modules from whatever
                    # is visible at that moment — packages installed
                    # afterwards cannot replace them. Spawning before install
                    # therefore made the first launch after a venv (re)build
                    # fail with host/venv version mixes (e.g. venv numba vs
                    # host numpy, venv torchvision vs host torch).
                    # Single-worker launches defer only the spawn (not the
                    # device reservation above) until after
                    # _prepare_virtual_env; sharded multi-worker launches
                    # need the subpool address inside model_kwargs before
                    # model instantiation, so they still spawn immediately.
                    subpool_address: Optional[str] = None
                    all_subpool_addresses: List[str] = []
                    try:
                        # Multi-worker/sharded launches spawn the subpool up
                        # front (the address is needed in model_kwargs before
                        # model instantiation). Keep it inside the try so a
                        # failure here is caught below and release_devices +
                        # subpool cleanup run, avoiding GPU/subpool leaks.
                        if n_worker > 1:  # type: ignore
                            subpool_address = await self._spawn_subpool(
                                model_uid,
                                subpool_alloc_env,
                                devices,
                                start_python=subpool_python_path,
                            )
                            all_subpool_addresses.append(subpool_address)
                        xavier_config: Optional[Dict] = kwargs.pop(
                            "xavier_config", None
                        )
                        model_kwargs = kwargs.copy()
                        model_kwargs["enable_virtual_env"] = enable_virtual_env
                        if n_worker > 1:  # type: ignore
                            # for model across workers,
                            # add a few kwargs
                            model_kwargs.update(
                                dict(
                                    address=subpool_address,
                                    n_worker=n_worker,
                                    shard=shard,
                                    driver_info=driver_info,
                                )
                            )

                        with CancellableDownloader(
                            cancelled_event=launch_info.cancel_event
                        ) as downloader:
                            launch_info.downloader = downloader
                            progressor = await self._get_progressor(
                                "launching-" + model_uid
                            )
                            # split into download and launch
                            progressor.split_stages(2, stage_weight=[0, 0.8, 1.0])
                            with progressor:
                                upload_progress_task = asyncio.create_task(
                                    asyncio.to_thread(
                                        self._upload_download_progress,
                                        progressor,
                                        downloader,
                                    )
                                )
                                # Limit hf_hub download concurrency to reduce GIL
                                # contention that starves the event loop.
                                _orig_hf_workers = os.environ.get(
                                    "HF_HUB_DOWNLOAD_WORKERS"
                                )
                                os.environ["HF_HUB_DOWNLOAD_WORKERS"] = str(
                                    XINFERENCE_MODEL_DOWNLOAD_WORKERS
                                )
                                try:
                                    # Wrap download phase with stream redirect when console logging is disabled
                                    if not XINFERENCE_LOG_CONSOLE:
                                        from ..deploy.utils import (
                                            redirect_streams_to_logger,
                                        )

                                        def _create_with_redirect():
                                            with redirect_streams_to_logger(
                                                XINFERENCE_LOG_DOWNLOAD_PROGRESS
                                            ):
                                                return create_model_instance(
                                                    model_uid,
                                                    model_type,
                                                    model_name,
                                                    model_engine,
                                                    model_format,
                                                    model_size_in_billions,
                                                    quantization,
                                                    peft_model_config,
                                                    download_hub,
                                                    model_path,
                                                    **model_kwargs,
                                                )

                                        model = await asyncio.to_thread(
                                            _create_with_redirect
                                        )
                                    else:
                                        model = await asyncio.to_thread(
                                            create_model_instance,
                                            model_uid,
                                            model_type,
                                            model_name,
                                            model_engine,
                                            model_format,
                                            model_size_in_billions,
                                            quantization,
                                            peft_model_config,
                                            download_hub,
                                            model_path,
                                            **model_kwargs,
                                        )
                                finally:
                                    if _orig_hf_workers is not None:
                                        os.environ["HF_HUB_DOWNLOAD_WORKERS"] = (
                                            _orig_hf_workers
                                        )
                                    else:
                                        os.environ.pop("HF_HUB_DOWNLOAD_WORKERS", None)
                            model.model_family.multimodal_projector = model_kwargs.get(
                                "multimodal_projector", None
                            )
                            await self.update_cache_status(
                                model_name, model.model_family.to_version_info()
                            )

                        def check_cancel():
                            # check downloader first, sometimes download finished
                            # cancelled already
                            if downloader.cancelled:
                                with progressor:
                                    # just report progress
                                    pass
                                downloader.raise_error(error_msg="Launch cancelled")

                        # check cancel before prepare virtual env
                        check_cancel()

                        # install packages in virtual env
                        if virtual_env_manager:
                            await asyncio.to_thread(
                                self._prepare_virtual_env,
                                virtual_env_manager,
                                model.model_family.virtualenv,
                                virtual_env_packages,
                                model_engine,
                                model_name=model_name,
                                architectures=getattr(
                                    model.model_family,
                                    "_resolve_architectures",
                                    lambda: None,
                                )(),
                                model_format=self._resolve_virtualenv_model_format(
                                    model, model_format
                                ),
                            )
                            launch_info.virtual_env_manager = virtual_env_manager

                        # check before creating subpool and model actor
                        check_cancel()

                        if subpool_address is None:
                            # Devices were already reserved above; only spawn
                            # the subprocess now that the virtualenv (if any)
                            # is installed.
                            subpool_address = await self._spawn_subpool(
                                model_uid,
                                subpool_alloc_env,
                                devices,
                                start_python=subpool_python_path,
                            )
                            all_subpool_addresses.append(subpool_address)
                        if xavier_config is not None:
                            xavier_config["rank_address"] = subpool_address
                        model.model_family.address = subpool_address
                        model.model_family.accelerators = devices

                        model_ref = await xo.create_actor(
                            ModelActor,
                            address=subpool_address,
                            uid=model_uid,
                            supervisor_address=self._supervisor_address,
                            worker_address=self.address,
                            replica_model_uid=model_uid,
                            model=model,
                            request_limits=request_limits,
                            xavier_config=xavier_config,
                            n_worker=n_worker,
                            shard=shard,
                            driver_info=driver_info,
                            model_engine=model_engine,
                        )
                        if await model_ref.need_create_pools() and (
                            len(devices) > 1 or n_worker > 1  # type: ignore
                        ):
                            coros = []
                            env_name = (
                                get_available_device_env_name()
                                or "CUDA_VISIBLE_DEVICES"
                            )
                            env_value = ",".join(devices)
                            for device in devices:
                                coros.append(
                                    self._append_sub_pool_protected(
                                        env={env_name: env_value},
                                        start_python=subpool_python_path,
                                        model_uid=model_uid,
                                    )
                                )
                            pool_addresses = await asyncio.gather(*coros)
                            await self._ensure_subpool_monitor()
                            all_subpool_addresses.extend(pool_addresses)
                            await model_ref.set_pool_addresses(pool_addresses)

                        # check before loading
                        check_cancel()

                        # set all subpool addresses
                        # when cancelled, all subpool addresses need to be destroyed
                        launch_info.sub_pools = all_subpool_addresses

                        with progressor:
                            try:
                                _load_start = time.time()
                                await model_ref.load()
                                _load_duration = time.time() - _load_start
                                from .metrics import model_last_load_duration_seconds

                                model_last_load_duration_seconds.set(
                                    {
                                        "model_name": model_name,
                                        "model_type": model_type,
                                        "worker_address": self.address,
                                    },
                                    _load_duration,
                                )
                            except xo.ServerClosed:
                                check_cancel()
                                raise
                    except Exception:
                        logger.error(f"Failed to load model {model_uid}", exc_info=True)
                        await self._update_model_state(model_uid, "error")
                        self.release_devices(model_uid=model_uid)
                        for addr in all_subpool_addresses:
                            try:
                                await self._main_pool.remove_sub_pool(addr)
                            except KeyError:
                                continue
                        raise
                    self._model_uid_to_model[model_uid] = model_ref
                    try:
                        self._model_uid_to_pid[model_uid] = await model_ref.get_pid()
                    except Exception:
                        pass
                    # Deterministic provenance: register all sub-pool process PIDs for
                    # this replica (primary ModelActor pool + per-device vLLM/SGLang rank
                    # pools). report_status maps NVML PIDs back to the owning replica via
                    # this table, without reading any process environ.
                    subpool_pids: Set[int] = set()
                    for _addr in all_subpool_addresses:
                        try:
                            _proc = self._main_pool.sub_processes.get(_addr)
                            if _proc is not None and _proc.pid is not None:
                                subpool_pids.add(_proc.pid)
                        except Exception:
                            continue
                    self._model_uid_to_subpool_pids[model_uid] = subpool_pids
                    model_spec = model.model_family.to_description()
                    # ``to_description`` is derived from the model family alone and
                    # therefore does not know which engine was selected at launch.
                    # Surface it here so ``/v1/models`` can report the running
                    # engine (shown in the Web UI running-model detail view).
                    if model_engine is not None:
                        model_spec["model_engine"] = model_engine
                    self._model_uid_to_model_spec[model_uid] = model_spec
                    self._model_uid_to_model_status[model_uid] = ModelStatus(
                        model_state="loading"
                    )
                    self._model_uid_to_addr[model_uid] = subpool_address
                    self._model_uid_to_recover_count.setdefault(
                        model_uid, XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT
                    )
                    self._model_uid_to_launch_args[model_uid] = launch_args
                    # §4.3: Persist for auto-recovery on restart
                    self._persist_launch_args()
                finally:
                    self._launch_active -= 1
                    logger.info(
                        "Launch finished: model_name=%s, model_uid=%s (active: %d/%d, queued: %d)",
                        model_name,
                        model_uid,
                        self._launch_active,
                        XINFERENCE_MAX_CONCURRENT_LAUNCHES,
                        self._launch_waiting,
                    )
        finally:
            if _was_queued:
                self._launch_waiting -= 1
            del self._model_uid_launching_guard[model_uid]

        # Record virtual environment information if applicable
        if virtual_env_manager is not None and virtual_env_path is not None:
            try:
                # Get package information from virtual environment settings
                packages: List[str] = []
                package_info: Dict[str, Any] = {}
                if (
                    model_spec
                    and hasattr(model_spec, "virtualenv")
                    and model_spec.virtualenv
                ):
                    packages = model_spec.virtualenv.packages or []

                # Virtual environment tracking is no longer needed
                logger.info(
                    f"Virtual environment created for model: {model.model_family.model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to handle virtual environment info: {e}")

        # update status to READY
        abilities = await self._get_model_ability(model, model_type)
        _ = await self.get_supervisor_ref(add_worker=False)

        if self._status_guard_ref is None:
            _ = await self.get_supervisor_ref()
        assert self._status_guard_ref is not None
        await self._status_guard_ref.update_instance_info(
            origin_uid,
            {"model_ability": abilities, "status": LaunchStatus.READY.name},
        )
        if n_worker > 1 and shard == 0:  # type: ignore
            return subpool_address, await model_ref.get_driver_info()
        else:
            return subpool_address

    @log_async(logger=logger, level=logging.INFO)
    async def wait_for_load(self, model_uid: str):
        model_ref = self._model_uid_to_model[model_uid]
        await model_ref.wait_for_load()
        await self._update_model_state(model_uid, "ready")

    @log_sync(logger=logger, level=logging.INFO)
    async def cancel_launch_model(self, model_uid: str):
        try:
            launch_info = self._model_uid_launching_guard[model_uid]

            # downloader shared same cancel event
            # sometimes cancel happens very early before downloader
            # even if users cancel at this time,
            # downloader will know and stop everything
            launch_info.cancel_event.set()

            if launch_info.downloader:
                logger.debug("Try to cancel download, %s")
                launch_info.downloader.cancel()
            if launch_info.virtual_env_manager:
                launch_info.virtual_env_manager.cancel_install()
            if launch_info.sub_pools:
                logger.debug("Try to stop sub pools: %s", launch_info.sub_pools)
                coros = []
                for addr in launch_info.sub_pools:
                    coros.append(self._main_pool.remove_sub_pool(addr, force=True))
                await asyncio.gather(*coros)
            if self._status_guard_ref is not None:
                await self._status_guard_ref.update_instance_info(
                    parse_replica_model_uid(model_uid)[0],
                    {"status": LaunchStatus.ERROR.name},
                )
        except KeyError:
            logger.error("Fail to cancel launching", exc_info=True)
            raise RuntimeError(
                "Model is not launching, may have launched or not launched yet"
            )

    @log_async(logger=logger, level=logging.INFO)
    async def terminate_model(self, model_uid: str, is_model_die=False):
        # Terminate model while its launching is not allow
        if model_uid in self._model_uid_launching_guard:
            raise ValueError(f"{model_uid} is launching")
        await self._update_model_state(model_uid, "stopping")
        # In special cases, if the suffix is `-rank0`, this is the Xavier's rank 0 model actor.
        if model_uid.endswith("-rank0"):
            origin_uid = model_uid.removesuffix("-rank0")
        else:
            origin_uid, _ = parse_replica_model_uid(model_uid)
        try:
            _ = await self.get_supervisor_ref()
            if self._event_collector_ref is not None:
                await self._event_collector_ref.report_event(
                    origin_uid,
                    Event(
                        event_type=EventType.INFO,
                        event_ts=int(time.time()),
                        event_content="Terminate model",
                    ),
                )
        except Exception as e:
            # Report callback error can be log and ignore, should not interrupt the Process
            logger.error("report_event error: %s" % (e))

        if self._status_guard_ref is not None:
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": LaunchStatus.TERMINATING.name}
            )
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            logger.debug("Model not found, uid: %s", model_uid)

        pool_addresses = None
        if model_ref is not None:
            try:
                # pool addresses if model.need_create_pools()
                pool_addresses = await model_ref.get_pool_addresses()
            except Exception as e:
                # process may disappear, we just ignore it.
                logger.debug("Fail to get pool addresses, error: %s", e)

        # Resolve GPU indices for terminate-path orphan cleanup.
        # Prefer launch_args["gpu_idx"] (user-specified), but fall back to
        # model_spec["accelerators"] (allocated devices when gpu_idx=None
        # and _create_subpool() auto-selected GPUs).
        _gpu_indices_for_terminate: list = []
        if not is_model_die:
            _launch_args = self._model_uid_to_launch_args.get(model_uid)
            if _launch_args:
                _gpu_indices_for_terminate = _parse_gpu_indices(
                    _launch_args.get("gpu_idx")
                )
            if not _gpu_indices_for_terminate:
                _model_spec = self._model_uid_to_model_spec.get(model_uid)
                if _model_spec and _model_spec.get("accelerators"):
                    _gpu_indices_for_terminate = [
                        int(dev) for dev in _model_spec["accelerators"]
                    ]

        try:
            logger.debug("Start to destroy model actor: %s", model_ref)
            if model_ref is not None:
                try:
                    await model_ref.stop()
                except Exception as e:
                    logger.debug(
                        "Stop model actor failed, model uid: %s, error: %s",
                        model_uid,
                        e,
                    )
            coro = xo.destroy_actor(model_ref)
            # see https://github.com/xorbitsai/xoscar/pull/140
            # asyncio.wait_for cannot work for Xoscar actor call,
            # because when time out, the coroutine will be cancelled via raise CancelledEror,
            # inside actor call, the error will be caught and
            # a CancelMessage will be sent to dest actor pool,
            # however the actor pool may be stuck already,
            # thus the timeout will never be raised
            await xo.wait_for(coro, timeout=5)
        except Exception as e:
            logger.debug(
                "Destroy model actor failed, model uid: %s, error: %s", model_uid, e
            )
        try:
            to_remove_addresses = []
            subpool_address = self._model_uid_to_addr[model_uid]
            to_remove_addresses.append(subpool_address)
            if pool_addresses:
                to_remove_addresses.extend(pool_addresses)
            logger.debug("Remove sub pools: %s", to_remove_addresses)
            coros = []
            for to_remove_addr in to_remove_addresses:
                coros.append(
                    self._main_pool.remove_sub_pool(to_remove_addr, force=True)
                )
            await asyncio.gather(*coros)
        except Exception as e:
            logger.debug(
                "Remove sub pool failed, model uid: %s, error: %s", model_uid, e
            )
        finally:
            # Clean up virtual environment tracking
            # Virtual environment tracking is no longer needed

            self._model_uid_to_model.pop(model_uid, None)
            self._model_uid_to_model_spec.pop(model_uid, None)
            self.release_devices(model_uid)
            self._model_uid_to_addr.pop(model_uid, None)
            self._model_uid_to_recover_count.pop(model_uid, None)
            self._model_uid_to_launch_args.pop(model_uid, None)
            self._model_uid_to_pid.pop(model_uid, None)
            self._model_uid_to_subpool_pids.pop(model_uid, None)
            # §4.3: Remove from persisted recovery file
            self._remove_persisted_launch_args(model_uid)

            if is_model_die:
                status = LaunchStatus.ERROR.name
            else:
                status = LaunchStatus.TERMINATED.name
                await self._update_model_state(model_uid, "stopped")
                self._model_uid_to_model_status.pop(model_uid, None)

            if self._status_guard_ref is None:
                _ = await self.get_supervisor_ref()
            assert self._status_guard_ref is not None
            await self._status_guard_ref.update_instance_info(
                origin_uid, {"status": status}
            )

        # Per-uid persist state cleanup (prevent zombie entries on long-running workers)
        self._persist_launch_args_dirty_uids.discard(model_uid)
        self._persist_retry_count.pop(model_uid, None)

        # Terminate-path orphan cleanup for TP=1 spawn EngineCore.
        # When is_model_die=False (normal terminate), spawn-created EngineCore
        # may survive stop() + remove_sub_pool as an orphan, holding VRAM.
        # Scan current GPU pids and SIGKILL those whose cmdline matches
        # vllm/enginecore (same identity check as the is_model_die=True path).
        try:
            if not is_model_die and _gpu_indices_for_terminate:
                await asyncio.sleep(1.0)
                _free_ratio = _snapshot_gpu_free_ratio(_gpu_indices_for_terminate)
                if 0 <= _free_ratio < _VRAM_READY_RATIO:
                    _exclude_pids: set = {os.getpid()}
                    for _uid, _pids in self._model_uid_to_subpool_pids.items():
                        _exclude_pids.update(_pids)
                    _killed = await _kill_orphan_gpu_pids(
                        _gpu_indices_for_terminate,
                        _exclude_pids,
                        model_uid=model_uid,
                    )
                    if _killed:
                        logger.warning(
                            "Killed %d GPU-occupying orphan(s) for %s: %s",
                            len(_killed),
                            model_uid,
                            _killed,
                        )
                    else:
                        logger.warning(
                            "No vllm/enginecore orphans found on GPU, "
                            "but VRAM still held for %s (free_ratio=%.2f)",
                            model_uid,
                            _free_ratio,
                        )
                    # Wait for VRAM reclaim after orphan cleanup
                    _vram_deadline = time.monotonic() + _VRAM_RECLAIM_TIMEOUT
                    while time.monotonic() < _vram_deadline:
                        await asyncio.sleep(1.0)
                        _free_ratio = _snapshot_gpu_free_ratio(
                            _gpu_indices_for_terminate
                        )
                        if _free_ratio < 0:
                            break
                        if _free_ratio >= _VRAM_READY_RATIO:
                            logger.info(
                                "VRAM reclaimed after orphan cleanup "
                                "for %s, free_ratio=%.2f",
                                model_uid,
                                _free_ratio,
                            )
                            break
                    else:
                        logger.warning(
                            "VRAM still not freed after orphan cleanup "
                            "+ %ds for %s (free_ratio=%.2f)",
                            _VRAM_RECLAIM_TIMEOUT,
                            model_uid,
                            _snapshot_gpu_free_ratio(_gpu_indices_for_terminate),
                        )
        except Exception:
            logger.warning("Orphan cleanup failed for %s", model_uid, exc_info=True)

    # Provide an interface for future version of supervisor to call
    def get_model_launch_status(self, model_uid: str) -> Optional[str]:
        """
        returns:
            CREATING: model is launching
            RREADY: model is running
            None: model is not running (launch error might have happened)
        """

        if model_uid in self._model_uid_launching_guard:
            return LaunchStatus.CREATING.name
        if model_uid in self._model_uid_to_model:
            return LaunchStatus.READY.name
        return None

    @log_async(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self._model_uid_to_model_spec.items()}

    @log_async(logger=logger)
    async def _update_model_state(self, model_uid: str, state: str):
        """Update ModelStatus.model_state and sync to StatusGuard."""
        ms = self._model_uid_to_model_status.get(model_uid)
        if ms is None:
            ms = ModelStatus()
            self._model_uid_to_model_status[model_uid] = ms
        ms.model_state = state
        status_map = {
            "registering": LaunchStatus.CREATING.name,
            "loading": LaunchStatus.LOADING.name,
            "ready": LaunchStatus.READY.name,
            "error": LaunchStatus.ERROR.name,
            "stopping": LaunchStatus.TERMINATING.name,
            "stopped": LaunchStatus.TERMINATED.name,
        }
        if self._status_guard_ref is not None:
            try:
                origin_uid, rank_suffix = parse_replica_model_uid(model_uid)
                replica_id = rank_suffix - 1 if rank_suffix > 0 else 0
                await self._status_guard_ref.update_replica_status(
                    origin_uid,
                    replica_id,
                    {
                        "status": status_map.get(state, LaunchStatus.CREATING.name),
                        "model_state": state,
                    },
                )
            except Exception:
                logger.debug(
                    "Failed to sync model_state to StatusGuard for %s",
                    model_uid,
                    exc_info=True,
                )

    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        model_status = self._model_uid_to_model_status.get(model_uid)
        if model_status:
            if model_status.model_state in ("registering", "loading"):
                raise ModelNotReadyError(
                    f"Model {model_uid} is {model_status.model_state}"
                )
            if model_status.model_state in ("error", "stopping", "stopped"):
                raise RuntimeError(
                    f"Model {model_uid} is in {model_status.model_state} state"
                )
            if model_status.last_error:
                raise Exception(model_status.last_error)
        model_ref = self._model_uid_to_model.get(model_uid, None)
        if model_ref is None:
            raise ValueError(f"Model not found, uid: {model_uid}")
        return model_ref

    @log_sync(logger=logger)
    def describe_model(self, model_uid: str) -> Dict[str, Any]:
        model_desc = self._model_uid_to_model_spec.get(model_uid, None)
        if model_desc is None:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")
        return model_desc

    async def report_status(self):
        status = dict()
        try:
            # Use configurable timeout for status gathering
            async with timeout(XINFERENCE_STATUS_GATHER_TIMEOUT):
                status = await asyncio.to_thread(gather_node_info)

                # Collect per-model GPU memory. Each replica's GPU holders are
                # its registered sub-pool PIDs (primary ModelActor pool +
                # per-device vLLM/SGLang rank pools) plus their recursive
                # children (e.g. vLLM V1 forked EngineCore). Attribution is
                # deterministic and reads no process environ.
                if self._total_gpu_devices:
                    try:
                        import psutil

                        from ..device_utils import get_per_process_gpu_memory

                        gpu_mem = await asyncio.to_thread(get_per_process_gpu_memory)
                        model_gpu_mem: Dict[str, Dict[int, int]] = {}

                        for m_uid in set(self._model_uid_to_pid) | set(
                            self._model_uid_to_subpool_pids
                        ):
                            pids: Set[int] = set(
                                self._model_uid_to_subpool_pids.get(m_uid, set())
                            )
                            own_pid = self._model_uid_to_pid.get(m_uid)
                            if own_pid is not None:
                                pids.add(own_pid)
                            # Recursive children cover GPU holders forked outside
                            # the registered sub-pools (e.g. vLLM V1 EngineCore).
                            for base in list(pids):
                                try:
                                    pids.update(
                                        c.pid
                                        for c in psutil.Process(base).children(
                                            recursive=True
                                        )
                                    )
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            per_gpu: Dict[int, int] = {}
                            for pid in pids:
                                if pid in gpu_mem:
                                    for gpu_idx, mem in gpu_mem[pid].items():
                                        per_gpu[gpu_idx] = per_gpu.get(gpu_idx, 0) + mem
                            if per_gpu:
                                model_gpu_mem[m_uid] = per_gpu

                        if model_gpu_mem:
                            status["model_gpu_memory"] = model_gpu_mem
                    except Exception:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Report status got error.")
        try:
            supervisor_ref = await self.get_supervisor_ref()
            await supervisor_ref.report_worker_status(self.address, status)
        except Exception:
            logger.warning(
                "Failed to report worker status, clearing cached supervisor references",
                exc_info=True,
            )
            self._clear_supervisor_refs()
            supervisor_ref = await self.get_supervisor_ref(add_worker=True)
            await supervisor_ref.report_worker_status(self.address, status)

    async def ping(self) -> bool:
        """Lightweight liveness probe for supervisor reverse-channel check."""
        return True

    async def heartbeat(self):
        """
        Lightweight heartbeat for liveness detection.
        Only sends address to supervisor without collecting resource info.
        Uses add_worker=False to avoid triggering reconnect initialization
        (add_worker + record_model_version). Registry recovery is driven solely
        by report_status -> report_worker_status path, per supervisor's
        receive_heartbeat design contract (supervisor.py).
        """
        await xo.wait_for(
            (await self.get_supervisor_ref(add_worker=False)).receive_heartbeat(
                self.address
            ),
            XINFERENCE_TCP_REQUEST_TIMEOUT,
        )

    async def _periodical_report_status(self):
        """
        Periodically send heartbeat and status reports to supervisor.
        Heartbeat is sent every interval, full status is sent every N intervals.
        """
        report_count = 0
        _heartbeat_fail_count = 0
        while True:
            try:
                # Always send heartbeat for liveness detection
                await self.heartbeat()

                # Send full status every N heartbeats
                if report_count % XINFERENCE_STATUS_REPORT_MULTIPLIER == 0:
                    await self.report_status()

                report_count += 1
                _heartbeat_fail_count = 0  # reset on success
            except asyncio.CancelledError:  # pragma: no cover
                break
            except RuntimeError as ex:  # pragma: no cover
                if "cannot schedule new futures" not in str(ex):
                    # when atexit is triggered, the default pool might be shutdown
                    # and to_thread will fail
                    break
            except (
                Exception
            ) as ex:  # pragma: no cover  # noqa: E722  # nosec  # pylint: disable=bare-except
                _heartbeat_fail_count += 1
                # §4.4: Log exception type and full traceback.
                # Print full traceback on 1st failure and every 10th consecutive failure
                # to avoid log bloat during prolonged outages.
                logger.error(
                    "Failed to upload node info: %s(%s)",
                    type(ex).__name__,
                    ex or "(empty message)",
                    exc_info=(
                        _heartbeat_fail_count == 1 or _heartbeat_fail_count % 10 == 0
                    ),
                )
            try:
                await asyncio.sleep(XINFERENCE_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break

    async def list_cached_models(
        self, model_name: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        # Defensive: _cache_tracker_ref may be None if get_supervisor_ref's core
        # init failed and cleared all refs. Return empty list instead of raising
        # AttributeError (which would surface as HTTP 500 to the user).
        if self._cache_tracker_ref is None:
            logger.warning(
                "cache_tracker_ref is None, returning empty cached model list"
            )
            return []
        lists = await self._cache_tracker_ref.list_cached_models(
            self.address, model_name
        )
        cached_models = []
        for list in lists:
            cached_model = {
                "model_name": list.get("model_name"),
                "model_size_in_billions": list.get("model_size_in_billions"),
                "model_format": list.get("model_format"),
                "quantization": list.get("quantization"),
                "model_version": list.get("model_version"),
            }
            path = list.get("model_file_location")
            cached_model["path"] = path
            real_path = get_real_path(path)
            if real_path:
                cached_model["real_path"] = real_path
            cached_model["actor_ip_address"] = self.address
            cached_models.append(cached_model)
        return cached_models

    async def list_deletable_models(self, model_version: str) -> List[str]:
        # Defensive: see list_cached_models for rationale.
        if self._cache_tracker_ref is None:
            logger.warning(
                "cache_tracker_ref is None, returning empty deletable model list"
            )
            return []
        paths = set()
        path = await self._cache_tracker_ref.list_deletable_models(
            model_version, self.address
        )
        if not path:
            return []

        # Always keep the symlink itself so broken links can be unlinked.
        if os.path.islink(path):
            paths.add(path)

        if os.path.isfile(path):
            path = os.path.dirname(path)

        if os.path.isdir(path):
            paths.add(path)
            files = os.listdir(path)
            paths.update([os.path.join(path, file) for file in files])
            # search real path
            if paths:
                paths.update(
                    [
                        real_path
                        for path in paths
                        if os.path.exists((real_path := os.path.realpath(path)))
                    ]
                )

            # get tensorizer path
            from ..model.llm.transformers.tensorizer_utils import get_tensorizer_dir

            tensorizer_path = get_tensorizer_dir(path)
            if os.path.isdir(tensorizer_path):
                files = os.listdir(tensorizer_path)
                paths.update([os.path.join(tensorizer_path, file) for file in files])

        return list(paths)

    async def confirm_and_remove_model(self, model_version: str) -> bool:
        # Defensive: see list_cached_models for rationale.
        if self._cache_tracker_ref is None:
            logger.warning("cache_tracker_ref is None, cannot confirm and remove model")
            return False
        paths = await self.list_deletable_models(model_version)
        for path in paths:
            try:
                if os.path.islink(path):
                    os.unlink(path)
                elif os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    logger.debug(f"{path} is not a valid path.")
            except Exception as e:
                logger.error(f"Fail to delete {path} with error:{e}.")  # noqa: E231
                return False

        await self._cache_tracker_ref.confirm_and_remove_model(
            model_version, self.address
        )
        return True

    # Virtual environment management methods
    async def list_virtual_envs(
        self, model_name: Optional[str] = None, model_engine: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        """List all virtual environments or filter by model name."""
        try:
            result = self._virtual_env_manager.list_virtual_envs(
                model_name, model_engine
            )
            # Add IP address to each virtual environment, same as cache implementation
            virtual_envs = []
            for env in result:
                virtual_env = {
                    "model_name": env.get("model_name"),
                    "model_engine": env.get("model_engine"),
                    "path": env.get("path"),
                    "real_path": env.get("real_path"),
                    "python_version": env.get("python_version"),
                    "actor_ip_address": self.address,
                }
                virtual_envs.append(virtual_env)

            return virtual_envs
        except Exception as e:
            logger.error(f"Error in list_virtual_envs: {e}")
            raise

    async def list_virtual_env_packages(self, model_name: str) -> Dict[str, Any]:
        """List packages installed in a specific virtual environment."""
        return self._virtual_env_manager.list_virtual_env_packages(model_name)

    async def remove_virtual_env(
        self,
        model_name: str,
        model_engine: Optional[str] = None,
        python_version: Optional[str] = None,
    ) -> bool:
        """Remove a virtual environment for a specific model."""
        return self._virtual_env_manager.remove_virtual_env(
            model_name, model_engine, python_version
        )

    async def get_workers_info(self) -> Dict[str, Any]:
        ret = {
            "work-ip": self.address,
            "models": await self.list_models(),
        }
        return ret

    def update_model_status(self, model_uid: str, **kwargs):
        model_status = self._model_uid_to_model_status.get(model_uid)
        if model_status is not None:
            for k, v in kwargs.items():
                setattr(model_status, k, v)

    def get_model_status(self, model_uid: str):
        return self._model_uid_to_model_status.get(model_uid)

    @staticmethod
    def record_metrics(name, op, kwargs):
        record_metrics(name, op, kwargs)

    async def start_transfer_for_vllm(
        self, rep_model_uid: str, rank_addresses: List[str]
    ):
        model_ref = self._model_uid_to_model[rep_model_uid]
        await model_ref.start_transfer_for_vllm(rank_addresses)

    @log_async(logger=logger, level=logging.INFO)
    async def launch_rank0_model(
        self, rep_model_uid: str, xavier_config: Dict[str, Any]
    ) -> Tuple[str, int]:
        from ..model.llm.vllm.xavier.collective_manager import Rank0ModelActor

        subpool_address = await self._append_sub_pool_protected(model_uid=rep_model_uid)
        await self._ensure_subpool_monitor()

        store_address = subpool_address.split(":")[0]
        # Note that `store_port` needs to be generated on the worker,
        # as the TCP store is on rank 0, not on the supervisor.
        store_port = xo.utils.get_next_port()
        self._model_uid_launching_guard[rep_model_uid] = LaunchInfo()
        try:
            try:
                xavier_config["rank_address"] = subpool_address
                xavier_config["store_address"] = store_address
                xavier_config["store_port"] = store_port
                model_ref = await xo.create_actor(
                    Rank0ModelActor,
                    address=subpool_address,
                    uid=rep_model_uid,
                    xavier_config=xavier_config,
                )
            except Exception:
                await self._main_pool.remove_sub_pool(subpool_address)
                raise
            self._model_uid_to_model[rep_model_uid] = model_ref
            self._model_uid_to_addr[rep_model_uid] = subpool_address
        finally:
            del self._model_uid_launching_guard[rep_model_uid]
        return subpool_address, store_port

    @no_type_check
    async def recover_model(self, launch_args: Dict[str, Any]):
        # `launch_ts` is an internal timestamp stamped onto the launch snapshot at
        # the entry of `launch_builtin_model` (see the `launch_args["launch_ts"]`
        # assignment); it is not a model construction parameter. recover_model
        # splats the whole snapshot back via `launch_builtin_model(**launch_args)`,
        # so `launch_ts` would land in that call's `**kwargs` and sink into the
        # model's `self._kwargs`. Models that forward the full `self._kwargs` into a
        # strict constructor (e.g. jina-reranker-v3 ->
        # `AutoModelForCausalLM.from_pretrained(**model_kwargs)`) then crash with
        # `TypeError: ... unexpected keyword argument 'launch_ts'`. The cross-session
        # recovery path (`recover_models_on_startup`) already pops it before
        # relaunch; do the same here. Copy first so the cached snapshot in
        # `self._model_uid_to_launch_args` keeps its `launch_ts` (used as created_ts).
        launch_args = dict(launch_args)
        launch_args.pop("launch_ts", None)
        rep_model_uid = launch_args.get("model_uid")
        origin_uid, _ = parse_replica_model_uid(rep_model_uid)
        xavier_config: Optional[Dict[str, Any]] = launch_args.get("xavier_config", None)
        is_xavier: bool = xavier_config is not None
        supervisor_ref = await self.get_supervisor_ref(add_worker=False)
        if is_xavier:
            rank = xavier_config.get("rank")
            await supervisor_ref.call_collective_manager(
                origin_uid, "unregister_rank", rank
            )
        subpool_address = await self.launch_builtin_model(**launch_args)
        if is_xavier:
            model_ref = self._model_uid_to_model[rep_model_uid]
            await model_ref.start_transfer_for_vllm([])
            rank = xavier_config.get("rank")
            await supervisor_ref.call_collective_manager(
                origin_uid, "register_rank", rank, subpool_address, update=True
            )
        # Mark the recreated replica ready, mirroring the normal launch path
        # (supervisor calls wait_for_load after launch). Without this the worker
        # keeps model_state="loading" forever (set in launch_builtin_model), so
        # get_model raises ModelNotReadyError and the recreated replica is a
        # permanent "loading" zombie -- the original 33% symptom. launch_builtin_model
        # already awaited model_ref.load(), so wait_for_load is near-instant here.
        await self.wait_for_load(rep_model_uid)
