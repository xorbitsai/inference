# Copyright 2022-2023 XProbe Inc.
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
import logging
import os
import random
import string
import uuid
from typing import Dict, Generator, List, Optional, Tuple, Union

import orjson
from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

from .._compat import BaseModel
from ..constants import XINFERENCE_LOG_ARG_MAX_LENGTH

logger = logging.getLogger(__name__)


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

    def decorator(func):
        func_name = func.__name__

        @wraps(func)
        async def wrapped(*args, **kwargs):
            request_id_str = kwargs.get("request_id", "")
            if not request_id_str:
                request_id_str = uuid.uuid1()
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
        yield f"{model_uid}-{replica}-{rep_id}"


def build_replica_model_uid(model_uid: str, replica: int, rep_id: int) -> str:
    """
    Build a replica model uid.
    """
    return f"{model_uid}-{replica}-{rep_id}"


def parse_replica_model_uid(replica_model_uid: str) -> Tuple[str, int, int]:
    """
    Parse replica model uid to model uid, replica and rep id.
    """
    parts = replica_model_uid.split("-")
    if len(parts) == 1:
        return replica_model_uid, -1, -1
    rep_id = int(parts.pop())
    replica = int(parts.pop())
    model_uid = "-".join(parts)
    return model_uid, replica, rep_id


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


def _get_nvidia_gpu_mem_info(gpu_id: int) -> Dict[str, float]:
    from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

    handler = nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = nvmlDeviceGetMemoryInfo(handler)
    return {"total": mem_info.total, "used": mem_info.used, "free": mem_info.free}


def get_nvidia_gpu_info() -> Dict:
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        res = {}
        for i in range(device_count):
            res[f"gpu-{i}"] = _get_nvidia_gpu_mem_info(i)
        return res
    except:
        # TODO: add log here
        # logger.debug(f"Cannot init nvml. Maybe due to lack of NVIDIA GPUs or incorrect installation of CUDA.")
        return {}
    finally:
        try:
            nvmlShutdown()
        except:
            pass


def assign_replica_gpu(
    _replica_model_uid: str, gpu_idx: Union[int, List[int]]
) -> List[int]:
    model_uid, replica, rep_id = parse_replica_model_uid(_replica_model_uid)
    rep_id, replica = int(rep_id), int(replica)
    if isinstance(gpu_idx, int):
        gpu_idx = [gpu_idx]
    if isinstance(gpu_idx, list) and gpu_idx:
        return gpu_idx[rep_id::replica]
    return gpu_idx
