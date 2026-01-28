# Copyright 2022-2026 XProbe Inc.
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
import logging
import os
import platform
import random
import string
import uuid
import weakref
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import orjson
from packaging.requirements import Requirement

from .._compat import BaseModel
from ..constants import (
    XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    XINFERENCE_LOG_ARG_MAX_LENGTH,
)

logger = logging.getLogger(__name__)


class AbortRequestMessage(Enum):
    NOT_FOUND = 1
    DONE = 2
    NO_OP = 3


def truncate_log_arg(arg) -> str:
    s = str(arg)
    if len(s) > XINFERENCE_LOG_ARG_MAX_LENGTH:
        s = s[0:XINFERENCE_LOG_ARG_MAX_LENGTH] + "..."
    return s


def log_async(
    logger,
    level=logging.DEBUG,
    ignore_kwargs: Optional[List[str]] = None,
    log_exception=True,
):
    import time
    from functools import wraps
    from inspect import signature

    def decorator(func):
        func_name = func.__name__
        sig = signature(func)

        @wraps(func)
        async def wrapped(*args, **kwargs):
            request_id_str = kwargs.get("request_id")
            if not request_id_str:
                # sometimes `request_id` not in kwargs
                # we try to bind the arguments
                try:
                    bound_args = sig.bind_partial(*args, **kwargs)
                    arguments = bound_args.arguments
                except TypeError:
                    arguments = {}
                request_id_str = arguments.get("request_id", "")
            if not request_id_str:
                request_id_str = uuid.uuid1()
                if func_name == "text_to_image":
                    kwargs["request_id"] = request_id_str
            request_id_str = f"[request {request_id_str}]"
            formatted_args = ",".join(map(truncate_log_arg, args))
            formatted_kwargs = ",".join(
                [
                    "%s=%s" % (k, truncate_log_arg(v))
                    for k, v in kwargs.items()
                    if ignore_kwargs is None or k not in ignore_kwargs
                ]
            )
            logger.log(
                level,
                f"{request_id_str} Enter {func_name}, args: {formatted_args}, kwargs: {formatted_kwargs}",
            )
            start = time.time()
            try:
                ret = await func(*args, **kwargs)
                logger.log(
                    level,
                    f"{request_id_str} Leave {func_name}, elapsed time: {int(time.time() - start)} s",
                )
                return ret
            except Exception as e:
                if log_exception:
                    logger.error(
                        f"{request_id_str} Leave {func_name}, error: {e}, elapsed time: {int(time.time() - start)} s",
                        exc_info=True,
                    )
                else:
                    logger.log(
                        level,
                        f"{request_id_str} Leave {func_name}, error: {e}, elapsed time: {int(time.time() - start)} s",
                    )
                raise

        return wrapped

    return decorator


def log_sync(logger, level=logging.DEBUG, log_exception=True):
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            formatted_args = ",".join(map(truncate_log_arg, args))
            formatted_kwargs = ",".join(
                map(lambda x: "%s=%s" % (x[0], truncate_log_arg(x[1])), kwargs.items())
            )
            logger.log(
                level,
                f"Enter {func.__name__}, args: {formatted_args}, kwargs: {formatted_kwargs}",
            )
            start = time.time()
            try:
                ret = func(*args, **kwargs)
                logger.log(
                    level,
                    f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} s",
                )
                return ret
            except Exception as e:
                if log_exception:
                    logger.error(
                        f"Leave {func.__name__}, error: {e}, elapsed time: {int(time.time() - start)} s",
                        exc_info=True,
                    )
                else:
                    logger.log(
                        level,
                        f"Leave {func.__name__}, error: {e}, elapsed time: {int(time.time() - start)} s",
                    )
                raise

        return wrapped

    return decorator


def iter_replica_model_uid(model_uid: str, replica: int) -> Generator[str, None, None]:
    """
    Generates all the replica model uids.
    """
    replica = int(replica)
    for rep_id in range(replica):
        yield f"{model_uid}-{rep_id}"


def build_replica_model_uid(model_uid: str, rep_id: int) -> str:
    """
    Build a replica model uid.
    """
    return f"{model_uid}-{rep_id}"


def parse_replica_model_uid(replica_model_uid: str) -> Tuple[str, int]:
    """
    Parse replica model uid to model uid and rep id.
    """
    parts = replica_model_uid.split("-")
    if len(parts) == 1:
        return replica_model_uid, -1
    rep_id = int(parts.pop())
    model_uid = "-".join(parts)
    return model_uid, rep_id


def is_valid_model_uid(model_uid: str) -> bool:
    model_uid = model_uid.strip()
    if not model_uid or len(model_uid) > 100:
        return False
    return True


def gen_random_string(length: int) -> str:
    return "".join(random.sample(string.ascii_letters + string.digits, length))


def json_dumps(o):
    def _default(obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        raise TypeError

    return orjson.dumps(o, default=_default)


def purge_dir(d):
    if not os.path.exists(d) or not os.path.isdir(d):
        return
    for name in os.listdir(d):
        subdir = os.path.join(d, name)
        try:
            if (os.path.islink(subdir) and not os.path.exists(subdir)) or (
                len(os.listdir(subdir)) == 0
            ):
                logger.info("Remove empty directory: %s", subdir)
                os.rmdir(subdir)
        except Exception:
            pass


def parse_model_version(model_version: str, model_type: str) -> Tuple:
    results: List[str] = model_version.split("--")
    if model_type == "LLM":
        if len(results) != 4:
            raise ValueError(
                f"LLM model_version parses failed! model_version: {model_version}"
            )
        model_name = results[0]
        size = results[1]
        if not size.endswith("B"):
            raise ValueError(f"Cannot parse model_size_in_billions: {size}")
        size = size.rstrip("B")
        size_in_billions: Union[int, str] = size if "_" in size else int(size)
        model_format = results[2]
        quantization = results[3]
        return model_name, size_in_billions, model_format, quantization
    elif model_type == "embedding":
        assert len(results) > 0, "Embedding model_version parses failed!"
        return (results[0],)
    elif model_type == "rerank":
        assert len(results) > 0, "Rerank model_version parses failed!"
        return (results[0],)
    elif model_type == "image":
        assert 2 >= len(results) >= 1, "Image model_version parses failed!"
        return tuple(results)
    else:
        raise ValueError(f"Not supported model_type: {model_type}")


def merge_virtual_env_packages(
    base_packages: List[str], extra_packages: Optional[List[str]]
) -> List[str]:
    """
    Merge default virtualenv packages with user provided ones. Packages with the
    same name will be replaced by the user supplied version instead of appended.
    """

    def get_key(package: str) -> str:
        if package.startswith("#"):
            # special placeholders like #system_torch#
            return package
        try:
            return Requirement(package).name.lower()
        except Exception:
            # fallback: strip version/url markers best effort
            for sep in ["@", "==", ">=", "<=", "~=", "!=", "[", " "]:
                if sep in package:
                    return package.split(sep, 1)[0].strip().lower()
            return package.lower()

    merged: List[str] = []
    index_map: Dict[str, int] = {}
    for pkg in base_packages:
        key = get_key(pkg)
        if key in index_map:
            merged[index_map[key]] = pkg
        else:
            index_map[key] = len(merged)
            merged.append(pkg)

    if extra_packages:
        for pkg in extra_packages:
            key = get_key(pkg)
            if key in index_map:
                merged[index_map[key]] = pkg
            else:
                index_map[key] = len(merged)
                merged.append(pkg)

    return merged


def build_subpool_envs_for_virtual_env(
    envs: Optional[Dict[str, str]],
    enable_virtual_env: Optional[bool],
    virtual_env_manager: Any,
) -> Dict[str, str]:
    subpool_envs = {} if envs is None else envs.copy()
    if bool(enable_virtual_env) and virtual_env_manager is not None:
        from .virtual_env_manager import resolve_virtualenv_python_path

        venv_python = resolve_virtualenv_python_path(virtual_env_manager)
        if not venv_python:
            return subpool_envs
        venv_bin = os.path.dirname(venv_python)
        venv_path = os.path.dirname(venv_bin)
        current_path = subpool_envs.get("PATH") or os.environ.get("PATH", "")
        subpool_envs["PATH"] = os.pathsep.join([venv_bin, current_path])
        subpool_envs["VIRTUAL_ENV"] = venv_path
        subpool_envs.setdefault(
            "FLASHINFER_NINJA_PATH", os.path.join(venv_bin, "ninja")
        )
    return subpool_envs


def apply_engine_virtualenv_settings(
    settings: Any,
    model_engine: Optional[str],
) -> None:
    """
    Apply engine-specific virtualenv settings (extra indexes, strategies).
    """
    if not model_engine:
        return
    from .virtual_env_manager import (
        get_engine_virtualenv_extra_index_urls,
        get_engine_virtualenv_index_strategy,
    )

    engine_extra_index_urls = get_engine_virtualenv_extra_index_urls(model_engine)
    engine_index_strategy = get_engine_virtualenv_index_strategy(model_engine)
    if settings.extra_index_url is None and engine_extra_index_urls:
        settings.extra_index_url = engine_extra_index_urls
    if settings.index_strategy is None and engine_index_strategy:
        settings.index_strategy = engine_index_strategy


def filter_virtualenv_packages_by_markers(
    packages: List[str],
    model_engine: Optional[str],
    cuda_version: Optional[str],
) -> List[str]:
    """
    Filter virtualenv packages using custom markers and system placeholders.

    This keeps #system_*# entries intact while evaluating engine/cuda/platform
    markers for other packages.
    """
    if not packages:
        return []
    system_markers = {
        "#system_torch#",
        "#system_torchaudio#",
        "#system_torchvision#",
        "#system_torchcodec#",
        "#system_numpy#",
    }

    def _marker_allows(marker: str) -> bool:
        marker = marker.strip().lower()
        if "#engine#" in marker or "#model_engine#" in marker:
            if not model_engine:
                return False
            engine_value = model_engine.lower()
            if (
                f'#engine# == "{engine_value}"' not in marker
                and f'#model_engine# == "{engine_value}"' not in marker
                and f"#model_engine# == '{engine_value}'" not in marker
            ):
                return False
        if 'cuda_version == "13.0"' in marker and cuda_version != "13.0":
            return False
        if 'cuda_version < "13.0"' in marker:
            if not cuda_version or cuda_version == "13.0":
                return False
        if (
            'platform_machine == "x86_64"' in marker
            and platform.machine().lower() != "x86_64"
        ):
            return False
        if (
            'platform_machine == "aarch64"' in marker
            and platform.machine().lower() != "aarch64"
        ):
            return False
        return True

    def _has_custom_marker(marker: str) -> bool:
        marker = marker.lower()
        return (
            "#engine#" in marker
            or "#model_engine#" in marker
            or "cuda_version" in marker
        )

    resolved_packages: List[str] = []
    for pkg in packages:
        if ";" in pkg:
            req, marker = pkg.split(";", 1)
            req = req.strip()
            marker = marker.strip()
            if req in system_markers:
                if _has_custom_marker(marker):
                    if _marker_allows(marker):
                        resolved_packages.append(req)
                    continue
                resolved_packages.append(pkg)
                continue
            if _has_custom_marker(marker):
                if _marker_allows(marker):
                    resolved_packages.append(req)
                continue
        resolved_packages.append(pkg)
    return resolved_packages


def assign_replica_gpu(
    _replica_model_uid: str, replica: int, gpu_idx: Optional[Union[int, List[int]]]
) -> Optional[List[int]]:
    model_uid, rep_id = parse_replica_model_uid(_replica_model_uid)
    rep_id, replica = int(rep_id), int(replica)
    if isinstance(gpu_idx, int):
        gpu_idx = [gpu_idx]
    if isinstance(gpu_idx, list) and gpu_idx:
        if len(gpu_idx) % replica != 0:
            raise ValueError(
                "gpu_idx length must be a multiple of replica when specifying GPUs."
            )
        num_gpus = len(gpu_idx)
        gpus_per_replica = num_gpus // replica
        start = rep_id * gpus_per_replica
        end = start + gpus_per_replica
        return gpu_idx[start:end]
    return gpu_idx


class CancelMixin:
    _CANCEL_TASK_NAME = "abort_block"

    def __init__(self):
        self._running_tasks: weakref.WeakValueDictionary[  # type: ignore
            str, asyncio.Task
        ] = weakref.WeakValueDictionary()

    def _add_running_task(self, request_id: Optional[str]):
        """Add current asyncio task to the running task.
        :param request_id: The corresponding request id.
        """
        if request_id is None:
            return
        running_task = self._running_tasks.get(request_id)
        if running_task is not None:
            if running_task.get_name() == self._CANCEL_TASK_NAME:
                raise Exception(f"The request has been aborted: {request_id}")
            raise Exception(f"Duplicate request id: {request_id}")
        current_task = asyncio.current_task()
        assert current_task is not None
        self._running_tasks[request_id] = current_task

    def _cancel_running_task(
        self,
        request_id: Optional[str],
        block_duration: int = XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    ):
        """Cancel the running asyncio task.
        :param request_id: The request id to cancel.
        :param block_duration: The duration seconds to ensure the request can't be executed.
        """
        if request_id is None:
            return
        running_task = self._running_tasks.pop(request_id, None)
        if running_task is not None:
            running_task.cancel()

        async def block_task():
            """This task is for blocking the request for a duration."""
            try:
                await asyncio.sleep(block_duration)
                logger.info("Abort block end for request: %s", request_id)
            except asyncio.CancelledError:
                logger.info("Abort block is cancelled for request: %s", request_id)

        if block_duration > 0:
            logger.info("Abort block start for request: %s", request_id)
            self._running_tasks[request_id] = asyncio.create_task(
                block_task(), name=self._CANCEL_TASK_NAME
            )
