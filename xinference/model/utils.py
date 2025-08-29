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

import asyncio
import functools
import json
import logging
import os
import random
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from json import JSONDecodeError
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import huggingface_hub
import numpy as np
import torch
from tqdm.auto import tqdm

from ..constants import (
    XINFERENCE_CACHE_DIR,
    XINFERENCE_DOWNLOAD_MAX_ATTEMPTS,
    XINFERENCE_ENV_MODEL_SRC,
)
from ..device_utils import get_available_device, is_device_available
from .core import CacheableModelSpec

if TYPE_CHECKING:
    from .embedding.core import LlamaCppEmbeddingSpecV1
    from .llm.llm_family import LlamaCppLLMSpecV2

logger = logging.getLogger(__name__)
IS_NEW_HUGGINGFACE_HUB: bool = huggingface_hub.__version__ >= "0.23.0"


def is_locale_chinese_simplified() -> bool:
    import locale

    try:
        lang, _ = locale.getdefaultlocale()
        return lang == "zh_CN"
    except:
        return False


def download_from_modelscope() -> bool:
    if os.environ.get(XINFERENCE_ENV_MODEL_SRC):
        return os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "modelscope"
    elif is_locale_chinese_simplified():
        return True
    else:
        return False


def download_from_openmind_hub() -> bool:
    if os.environ.get(XINFERENCE_ENV_MODEL_SRC):
        return os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "openmind_hub"
    else:
        return False


def download_from_csghub() -> bool:
    if os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "csghub":
        return True
    return False


def symlink_local_file(path: str, local_dir: str, relpath: str) -> str:
    from huggingface_hub.file_download import _create_symlink

    # cross-platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*relpath.split("/"))
    if os.name == "nt":
        if relative_filename.startswith("..\\") or "\\..\\" in relative_filename:
            raise ValueError(
                f"Invalid filename: cannot handle filename '{relative_filename}' on Windows. Please ask the repository"
                " owner to rename this file."
            )
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    local_dir_filepath = os.path.join(local_dir, relative_filename)
    if (
        Path(os.path.abspath(local_dir))
        not in Path(os.path.abspath(local_dir_filepath)).parents
    ):
        raise ValueError(
            f"Cannot copy file '{relative_filename}' to local dir '{local_dir}': file would not be in the local"
            " directory."
        )

    os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)
    real_blob_path = os.path.realpath(path)
    _create_symlink(real_blob_path, local_dir_filepath, new_blob=False)
    return local_dir_filepath


def create_symlink(download_dir: str, cache_dir: str):
    for subdir, dirs, files in os.walk(download_dir):
        for file in files:
            relpath = os.path.relpath(os.path.join(subdir, file), download_dir)
            symlink_local_file(os.path.join(subdir, file), cache_dir, relpath)


def retry_download(
    download_func: Callable,
    model_name: str,
    model_info: Optional[Dict],
    *args,
    **kwargs,
):
    last_ex = None
    for current_attempt in range(1, XINFERENCE_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            return download_func(*args, **kwargs)
        except Exception as e:
            remaining_attempts = XINFERENCE_DOWNLOAD_MAX_ATTEMPTS - current_attempt
            last_ex = e
            logger.debug(
                "Download failed: %s, download func: %s, download args: %s, kwargs: %s",
                e,
                download_func,
                args,
                kwargs,
            )
            logger.warning(
                f"Attempt {current_attempt} failed. Remaining attempts: {remaining_attempts}"
            )

    else:
        model_size = (
            model_info.pop("model_size", None) if model_info is not None else None
        )
        model_format = (
            model_info.pop("model_format", None) if model_info is not None else None
        )
        if model_size is not None or model_format is not None:  # LLM models
            raise RuntimeError(
                f"Failed to download model '{model_name}' "
                f"(size: {model_size}, format: {model_format}) "
                f"after multiple retries"
            ) from last_ex
        else:  # Embedding models
            raise RuntimeError(
                f"Failed to download model '{model_name}' after multiple retries"
            ) from last_ex


def valid_model_revision(
    meta_path: str,
    expected_model_revision: Optional[str],
    expected_model_hub: Optional[str] = None,
) -> bool:
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, "r") as f:
        try:
            meta_data = json.load(f)
        except JSONDecodeError:  # legacy meta file for embedding models
            logger.debug("Legacy meta file detected.")
            return True

        if "model_revision" in meta_data:  # embedding, image
            real_revision = meta_data["model_revision"]
        elif "revision" in meta_data:  # llm
            real_revision = meta_data["revision"]
        else:
            logger.warning(
                f"No `revision` information in the `__valid_download` file. "
            )
            return False
        if expected_model_hub is not None and expected_model_hub != meta_data.get(
            "model_hub", "huggingface"
        ):
            logger.info("Use model cache from a different hub.")
            return True
        else:
            return real_revision == expected_model_revision


def get_cache_dir(model_spec: Any) -> str:
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def is_model_cached(model_spec: Any, name_to_revisions_mapping: Dict):
    cache_dir = get_cache_dir(model_spec)
    meta_path = os.path.join(cache_dir, "__valid_download")
    revisions = name_to_revisions_mapping[model_spec.model_name]
    if model_spec.model_revision not in revisions:  # Usually for UT
        revisions.append(model_spec.model_revision)
    return any([valid_model_revision(meta_path, revision) for revision in revisions])


def is_valid_model_name(model_name: str) -> bool:
    import re

    if len(model_name) == 0:
        return False

    # check if contains +/?%#&=\s
    return re.match(r"^[^+\/?%#&=\s]*$", model_name) is not None


def parse_uri(uri: str) -> Tuple[str, str]:
    import glob
    from urllib.parse import urlparse

    if os.path.exists(uri) or glob.glob(uri):
        return "file", uri
    else:
        parsed = urlparse(uri)
        scheme = parsed.scheme
        path = parsed.netloc + parsed.path
        if parsed.scheme == "" or len(parsed.scheme) == 1:  # len == 1 for windows
            scheme = "file"
        return scheme, path


def is_valid_model_uri(model_uri: Optional[str]) -> bool:
    if not model_uri:
        return False

    src_scheme, src_root = parse_uri(model_uri)

    if src_scheme == "file":
        if not os.path.isabs(src_root):
            raise ValueError(f"Model URI cannot be a relative path: {model_uri}")
        return os.path.exists(src_root)
    else:
        # TODO: handle other schemes.
        return True


def cache_from_uri(model_spec: CacheableModelSpec) -> str:
    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    )
    if os.path.exists(cache_dir):
        logger.info("cache %s exists", cache_dir)
        return cache_dir

    assert model_spec.model_uri is not None
    src_scheme, src_root = parse_uri(model_spec.model_uri)
    if src_root.endswith("/"):
        # remove trailing path separator.
        src_root = src_root[:-1]

    if src_scheme == "file":
        if not os.path.isabs(src_root):
            raise ValueError(
                f"Model URI cannot be a relative path: {model_spec.model_uri}"
            )
        os.makedirs(XINFERENCE_CACHE_DIR, exist_ok=True)
        os.symlink(src_root, cache_dir, target_is_directory=True)
        return cache_dir
    else:
        raise ValueError(f"Unsupported URL scheme: {src_scheme}")


def select_device(device):
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
        )

    if device == "auto":
        return get_available_device()
    else:
        if not is_device_available(device):
            raise ValueError(f"{device} is unavailable in your environment")

    return device


def convert_float_to_int_or_str(model_size: float) -> Union[int, str]:
    """convert float to int or string

    if float can be presented as int, convert it to int, otherwise convert it to string
    """
    if int(model_size) == model_size:
        return int(model_size)
    else:
        return str(model_size)


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CancellableDownloader:
    def __init__(
        self,
        cancel_error_cls: Type[BaseException] = asyncio.CancelledError,
        cancelled_event: Optional[threading.Event] = None,
    ):
        self._cancelled = cancelled_event
        if self._cancelled is None:
            self._cancelled = threading.Event()
        self._done_event = threading.Event()
        self._cancel_error_cls = cancel_error_cls
        self._original_update = None
        # progress for tqdm that is main
        self._main_progresses: Set[tqdm] = set()
        # progress for file downloader
        # mainly when tqdm unit is set
        self._download_progresses: Set[tqdm] = set()
        # tqdm original update
        self._original_tqdm_update = None

    def reset(self):
        self._main_progresses.clear()
        self._download_progresses.clear()

    def get_progress(self) -> float:
        if self.cancelled or self.done:
            # directly return 1.0 when cancelled or finished
            return 1.0

        tasks = finished_tasks = 0
        for main_progress in self._main_progresses:
            tasks += main_progress.total or 0
            finished_tasks += main_progress.n

        if tasks == 0:
            # we assumed at least 1 task
            tasks = 1

        finished_ratio = finished_tasks / tasks

        all_download_progress = finished_download_progress = 0
        for download_progress in self._download_progresses:
            # we skip finished download
            if download_progress.n == download_progress.total:
                continue
            all_download_progress += download_progress.total or (
                download_progress.n * 10
            )
            finished_download_progress += download_progress.n

        if all_download_progress > 0:
            rest_ratio = (
                (tasks - finished_tasks)
                / tasks
                * (finished_download_progress / all_download_progress)
            )
            return finished_ratio + rest_ratio
        else:
            return finished_ratio

    def cancel(self):
        self._cancelled.set()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    @property
    def done(self):
        return self._done_event.is_set()

    def wait(self, timeout: float):
        self._done_event.wait(timeout)

    def raise_error(self, error_msg: str = "Download cancelled"):
        raise self._cancel_error_cls(error_msg)

    def patch_tqdm(self):
        # patch tqdm
        # raise error if cancelled
        self._original_update = original_update = tqdm.update
        downloader = self

        def patched_update(self, n):
            if downloader.cancelled:
                downloader.raise_error()
            if not self.disable:
                progresses = (
                    downloader._main_progresses
                    if getattr(self, "unit", "it") == "it"
                    else downloader._download_progresses
                )
                progresses.add(self)
            return original_update(self, n)

        tqdm.update = patched_update

    def unpatch_tqdm(self):
        from tqdm.auto import tqdm

        if self._original_update:
            tqdm.update = self._original_update

    def __enter__(self):
        self.patch_tqdm()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unpatch_tqdm()
        self._done_event.set()
        self.reset()


def get_engine_params_by_name(
    model_type: Optional[str], model_name: str
) -> Optional[Dict[str, List[dict]]]:
    if model_type == "LLM":
        from .llm.llm_family import LLM_ENGINES

        if model_name not in LLM_ENGINES:
            return None

        # filter llm_class
        engine_params = deepcopy(LLM_ENGINES[model_name])
        for engine, params in engine_params.items():
            for param in params:
                del param["llm_class"]

        return engine_params
    elif model_type == "embedding":
        from .embedding.embed_family import EMBEDDING_ENGINES

        if model_name not in EMBEDDING_ENGINES:
            return None

        # filter embedding_class
        engine_params = deepcopy(EMBEDDING_ENGINES[model_name])
        for engine, params in engine_params.items():
            for param in params:
                del param["embedding_class"]

        return engine_params
    elif model_type == "rerank":
        from .rerank.rerank_family import RERANK_ENGINES

        if model_name not in RERANK_ENGINES:
            return None

        # filter rerank_class
        engine_params = deepcopy(RERANK_ENGINES[model_name])
        for engine, params in engine_params.items():
            for param in params:
                del param["rerank_class"]

        return engine_params
    else:
        raise ValueError(
            f"Cannot support model_engine for {model_type}, "
            f"only available for LLM, embedding"
        )


def generate_model_file_names_with_quantization_parts(
    model_spec: Union["LlamaCppLLMSpecV2", "LlamaCppEmbeddingSpecV1"],
    multimodal_projector: Optional[str] = None,
) -> Tuple[List[str], str, bool]:
    file_names = []
    final_file_name = model_spec.model_file_name_template.format(
        quantization=model_spec.quantization
    )
    need_merge = False

    if (
        model_spec.quantization_parts is None
        or model_spec.quantization not in model_spec.quantization_parts
    ):
        file_names.append(final_file_name)
    elif (
        model_spec.quantization is not None
        and model_spec.quantization in model_spec.quantization_parts
    ):
        parts = model_spec.quantization_parts[model_spec.quantization]
        need_merge = True

        logger.info(
            f"Model {model_spec.model_id} {model_spec.model_format} {model_spec.quantization} has {len(parts)} parts."
        )

        if model_spec.model_file_name_split_template is None:
            raise ValueError(
                f"No model_file_name_split_template for model spec {model_spec.model_id}"
            )

        for part in parts:
            file_name = model_spec.model_file_name_split_template.format(
                quantization=model_spec.quantization, part=part
            )
            file_names.append(file_name)
    if multimodal_projector:
        file_names.append(multimodal_projector)

    return file_names, final_file_name, need_merge


def merge_cached_files(
    cache_dir: str, input_file_names: List[str], output_file_name: str
):
    # now llama.cpp can find the gguf parts automatically
    # we only need to provide the first part
    # thus we create the symlink to the first part
    symlink_local_file(
        os.path.join(cache_dir, input_file_names[0]), cache_dir, output_file_name
    )

    logger.info(f"Merge complete.")


def flatten_model_src(input_json: dict):
    flattened = []
    base_info = {key: value for key, value in input_json.items() if key != "model_src"}
    for model_hub, hub_info in input_json["model_src"].items():
        record = base_info.copy()
        hub_info.pop("model_hub", None)
        record.update(hub_info)
        record["model_hub"] = model_hub
        flattened.append(record)
    return flattened


def flatten_quantizations(input_json: dict):
    flattened = []

    base_info = {key: value for key, value in input_json.items() if key != "model_src"}

    for model_hub, hub_info in input_json["model_src"].items():
        quantizations = hub_info["quantizations"]

        for quant in quantizations:
            record = base_info.copy()
            record["model_hub"] = model_hub
            record["quantization"] = quant

            for key, value in hub_info.items():
                if key != "quantizations":
                    record[key] = value

            flattened.append(record)
    return flattened


class ModelInstanceInfoMixin(ABC):
    @abstractmethod
    def to_description(self):
        """"""

    @abstractmethod
    def to_version_info(self):
        """"""


def is_flash_attn_available() -> bool:
    """
    Check if flash_attention can be enabled in the current environment.

    Checks the following conditions:
    1. Whether the flash_attn package is installed
    2. Whether CUDA GPU is available
    3. Whether PyTorch supports CUDA
    4. Whether GPU compute capability meets requirements (>= 8.0)

    Returns:
        bool: True if flash_attention can be enabled, False otherwise
    """
    import importlib.util

    # Check if flash_attn is installed
    if importlib.util.find_spec("flash_attn") is None:
        logger.debug("flash_attn package not found")
        return False

    try:
        import torch

        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.debug("CUDA not available")
            return False

        # Check GPU count
        if torch.cuda.device_count() == 0:
            logger.debug("No CUDA devices found")
            return False

        # Check current GPU compute capability
        # Flash Attention typically requires compute capability >= 8.0 (A100, H100, etc.)
        current_device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(current_device)
        major, minor = capability
        compute_capability = major + minor * 0.1

        if compute_capability < 8.0:
            logger.debug(
                f"GPU compute capability {compute_capability} < 8.0, "
                "flash_attn may not work optimally"
            )
            return False

        # Try to import flash_attn core module to verify correct installation
        try:
            import flash_attn

            logger.debug(
                f"flash_attn version: {getattr(flash_attn, '__version__', 'unknown')}"
            )
            return True
        except ImportError as e:
            logger.debug(f"Failed to import flash_attn: {e}")
            return False
    except Exception as e:
        logger.debug(f"Error checking flash_attn availability: {e}")
        return False


def cache_clean(fn):
    @functools.wraps(fn)
    async def _async_wrapper(self, *args, **kwargs):
        import gc

        from ..device_utils import empty_cache

        result = await fn(self, *args, **kwargs)

        gc.collect()
        empty_cache()
        return result

    @functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        import gc

        from ..device_utils import empty_cache

        result = fn(self, *args, **kwargs)

        gc.collect()
        empty_cache()
        return result

    if asyncio.iscoroutinefunction(fn):
        return _async_wrapper
    else:
        return _wrapper
