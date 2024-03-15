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
import copy
import logging
import os
import random
import re
import string
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import orjson
from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown
from typing_extensions import Literal

from .._compat import BaseModel

logger = logging.getLogger(__name__)


def log_async(logger, args_formatter=None):
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            if args_formatter is not None:
                formatted_args, formatted_kwargs = copy.copy(args), copy.copy(kwargs)
                args_formatter(formatted_args, formatted_kwargs)
            else:
                formatted_args, formatted_kwargs = args, kwargs
            logger.debug(
                f"Enter {func.__name__}, args: {formatted_args}, kwargs: {formatted_kwargs}"
            )
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


def get_model_size_from_model_id(model_id: str) -> str:
    """
    Get model size from model_id.

    Args:
        model_id: model_id in format of `user/repo`

    Returns:
        model size in format of `100B`, if size is in M, divide into 1000 and return as B.
        For example, `100M` will be returned as `0.1B`.

        If there is no model size in the repo name, return `UNKNOWN`.
    """

    def resize_to_billion(size: str) -> str:
        if size.lower().endswith("m"):
            return str(round(int(size[:-1]) / 1000, 2)).rstrip("0") + "B"
        if size[0] == "0":
            size = size[0] + "." + str(size[1:])
        return size.replace("_", ".").upper()

    split = model_id.split("/")
    if len(split) != 2:
        raise ValueError(f"Cannot parse model_id: {model_id}")
    user, repo = split
    segs = repo.split("-")
    param_pattern = re.compile(r"\d+(?:[._]\d+)?[bm]", re.I)
    partial_matched = "UNKNOWN"
    for seg in segs:
        if m := param_pattern.search(seg):
            if m.start() == 0 and m.end() == len(seg):
                return resize_to_billion(seg)
            else:
                # only match the first partial matched, and do not match `bit` for quantization mode
                if (
                    partial_matched == "UNKNOWN"
                    and seg[m.end(0) : m.end(0) + 2].lower() != "it"
                ):
                    partial_matched = m.group(0)
    return resize_to_billion(partial_matched)


SUPPORTED_QUANTIZATIONS = [
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "F32",
    "F16",
    "Q4_0",
    "Q4_1",
    "Q8_0",
    "Q5_0",
    "Q5_1",
    "Q2_K",
]


def get_match_quantization_filenames(
    filenames: List[str],
) -> List[Tuple[str, str, int]]:
    results: List[Tuple[str, str, int]] = []
    for filename in filenames:
        for quantization in SUPPORTED_QUANTIZATIONS:
            if (index := filename.upper().find(quantization)) != -1:
                results.append((filename, quantization, index))
    return results


def get_prefix_suffix(names: Iterable[str]) -> Tuple[str, str]:
    if len(list(names)) == 0:
        return "", ""

    # if all names are the same, or only one name, return the first name as prefix and suffix is empty
    if len(set(names)) == 1:
        return list(names)[0], ""

    min_len = min(map(len, names))
    name = [n for n in names if len(n) == min_len][0]

    for i in range(min_len):
        if len(set(map(lambda x: x[: i + 1], names))) > 1:
            prefix = name[:i]
            break
    else:
        prefix = name

    for i in range(min_len):
        if len(set(map(lambda x: x[-i - 1 :], names))) > 1:
            suffix = name[len(name) - i :]
            break
    else:
        suffix = name

    return prefix, suffix


def get_llama_cpp_quantization_info(
    filenames: List[str], model_type: Literal["ggmlv3", "ggufv2"]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the model file name template and split template from a list of filenames.

    NOTE: not support multiple quantization files in multi-part zip files.
         for example: a-16b.ggmlv3.zip a-16b.ggmlv3.z01 a-16b.ggmlv3.z02 are not supported
    """
    model_file_name_template = None
    model_file_name_split_template: Optional[str] = None
    if model_type == "ggmlv3":
        filenames = [
            filename
            for filename in filenames
            if filename.lower().endswith(".bin") or "ggml" in filename.lower()
        ]
    elif model_type == "ggufv2":
        filenames = [filename for filename in filenames if ".gguf" in filename]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    matched = get_match_quantization_filenames(filenames)

    if len(matched) == 0:
        raise ValueError("Cannot find any quantization files in this")

    prefixes = set()
    suffixes = set()

    for filename, quantization, index in matched:
        prefixes.add(filename[:index])
        suffixes.add(filename[index + len(quantization) :])

    if len(prefixes) == 1 and len(suffixes) == 1:
        model_file_name_template = prefixes.pop() + "{quantization}" + suffixes.pop()

    elif len(prefixes) == 1 and len(suffixes) > 1:
        shortest_suffix = min(suffixes, key=len)
        part_prefix, part_suffix = get_prefix_suffix(suffixes)
        if shortest_suffix == part_prefix + part_suffix:
            model_file_name_template = (
                list(prefixes)[0] + "{quantization}" + shortest_suffix
            )
            part_prefix, part_suffix = get_prefix_suffix(
                [suffix for suffix in suffixes if suffix != shortest_suffix]
            )
            model_file_name_split_template = (
                prefixes.pop() + "{quantization}" + part_prefix + "{part}" + part_suffix
            )
        else:
            model_file_name_split_template = (
                prefixes.pop() + "{quantization}" + part_prefix + "{part}" + part_suffix
            )

    elif len(prefixes) > 1 and len(suffixes) == 1:
        shortest_prefix = min(prefixes, key=len)
        part_prefix, part_suffix = get_prefix_suffix(prefixes)
        if shortest_prefix == part_prefix + part_suffix:
            model_file_name_template = (
                shortest_prefix + "{quantization}" + list(suffixes)[0]
            )
            part_prefix, part_suffix = get_prefix_suffix(
                [prefix for prefix in prefixes if prefix != shortest_prefix]
            )
            model_file_name_split_template = (
                part_prefix
                + "{quantization}"
                + shortest_prefix
                + "{part}"
                + part_suffix
            )
        else:
            model_file_name_split_template = (
                prefixes.pop() + "{quantization}" + part_prefix + "{part}" + part_suffix
            )
    else:
        logger.info("Cannot find a valid template for model file names")

    return model_file_name_template, model_file_name_split_template
