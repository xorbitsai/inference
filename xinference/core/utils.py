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
from typing import Generator, List, Tuple, Union

import orjson
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def log_async(logger):
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
            start = time.time()
            ret = await func(*args, **kwargs)
            logger.debug(
                f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} s"
            )
            return ret

        return wrapped

    return decorator


def log_sync(logger):
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
            start = time.time()
            ret = func(*args, **kwargs)
            logger.debug(
                f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} s"
            )
            return ret

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
