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
import functools
import importlib
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
    no_type_check,
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


def check_dependency_available(
    module_name: str, friendly_name: Optional[str] = None
) -> Union[bool, Tuple[bool, str]]:
    """Check whether a dependency can be imported, returning detailed errors."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        return False, f"Failed to import {friendly_name or module_name}: {exc}"
    except OSError as exc:
        return (
            False,
            f"Failed to load {friendly_name or module_name} native extension: {exc}",
        )
    except Exception as exc:
        return False, f"Error while importing {friendly_name or module_name}: {exc}"
    return True


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
    _global_lock = threading.Lock()
    _active_instances = 0
    _original_update = None  # Class-level original update method (tqdm.auto.tqdm)
    _original_update_plain = None  # Class-level original update method (tqdm.tqdm)
    _patch_lock = threading.Lock()  # Additional lock for patching operations

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
        # progress for tqdm that is main
        self._main_progresses: Set[tqdm] = set()
        # progress for file downloader
        # mainly when tqdm unit is set
        self._download_progresses: Set[tqdm] = set()
        # Instance-specific tqdm tracking
        self._patched_instances: Set[int] = set()

    def reset(self):
        self._main_progresses.clear()
        self._download_progresses.clear()

    def get_progress(self) -> float:
        if self.done:
            # directly return 1.0 when finished
            return 1.0
        # Don't return 1.0 when cancelled, calculate actual progress

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
        self._done_event.set()

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
        # Use class-level patching to avoid conflicts
        with self._patch_lock:
            import tqdm as tqdm_module

            if self._original_update is None:
                self._original_update = tqdm.update
            if self._original_update_plain is None:
                self._original_update_plain = tqdm_module.tqdm.update

            if self._original_update is None or self._original_update_plain is None:
                return

            original_update_plain = self._original_update_plain

            # Thread-safe patched update
            def patched_update(tqdm_instance, n):
                import gc

                # Get all CancellableDownloader instances and check for cancellation
                downloaders = [
                    obj
                    for obj in gc.get_objects()
                    if isinstance(obj, CancellableDownloader)
                ]

                for downloader in downloaders:
                    # if download cancelled, throw error
                    if getattr(downloader, "cancelled", False):
                        downloader.raise_error()

                    progresses = None
                    if not getattr(tqdm_instance, "disable", False):
                        unit = getattr(tqdm_instance, "unit", "it")
                        if unit == "it":
                            progresses = getattr(downloader, "_main_progresses", None)
                        else:
                            progresses = getattr(
                                downloader, "_download_progresses", None
                            )

                    if progresses is not None:
                        progresses.add(tqdm_instance)
                    else:
                        logger.debug(f"No progresses found for downloader {downloader}")

                # Call original update with safety check
                return original_update_plain(tqdm_instance, n)

            tqdm.update = patched_update
            tqdm_module.tqdm.update = patched_update

    def unpatch_tqdm(self):
        with self._patch_lock:
            if self._original_update is not None and self._active_instances == 0:
                import tqdm as tqdm_module

                tqdm.update = self._original_update
                self._original_update = None
                if self._original_update_plain is not None:
                    tqdm_module.tqdm.update = self._original_update_plain
                    self._original_update_plain = None

    def __enter__(self):
        # Use global lock to prevent concurrent patching
        with self._global_lock:
            if self._active_instances == 0:
                self.patch_tqdm()
            self._active_instances += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Use global lock to prevent concurrent unpatching
        with self._global_lock:
            self._active_instances -= 1
            if self._active_instances == 0:
                self.unpatch_tqdm()
        try:
            self._done_event.set()
            self.reset()
        except Exception as e:
            logger.debug(f"Error during CancellableDownloader cleanup: {e}")


@no_type_check
def get_engine_params_by_name(
    model_type: Optional[str], model_name: str
) -> Optional[Dict[str, Any]]:
    engine_params: Dict[str, Any] = {}

    def _normalize_match_result(
        result: Any, default_error: str, default_type: str
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        if result is True:
            return True, None, None, None
        if result is False or result is None:
            return False, default_error, default_type, None

        if isinstance(result, tuple) and len(result) >= 2:
            flag, reason = result[0], result[1]
            if isinstance(flag, bool):
                if flag:
                    return True, None, None, None
                reason_str = str(reason) if reason is not None else default_error
                return False, reason_str, default_type, None

        if hasattr(result, "is_match"):
            is_match = bool(getattr(result, "is_match"))
            reason = getattr(result, "reason", None)
            err_type = getattr(result, "error_type", default_type)
            technical_details = getattr(result, "technical_details", None)
            return is_match, reason, err_type, technical_details

        if isinstance(result, str):
            return False, result, default_type, None

        return False, str(result), default_type, None

    def _append_available_engine(
        engine: str, params: List[Dict[str, Any]], class_field: str
    ):
        cleaned_params: List[Dict[str, Any]] = []
        for param in params:
            new_param = {k: v for k, v in param.items() if k != class_field}
            cleaned_params.append(new_param)
        engine_params[engine] = cleaned_params

    def _append_unavailable_engine(
        engine: str,
        reason: Optional[str],
        error_type: Optional[str],
        technical_details: Optional[str],
    ):
        # Keep legacy string format for unavailable engines
        engine_params[engine] = (
            reason
            or technical_details
            or f"Engine {engine} is not compatible with current model or environment"
        )

    def _collect_supported_engines(
        family: Optional[Any],
        supported_engines: Dict[str, List[Type[Any]]],
        engine_type_label: str,
    ):
        if family is None:
            return
        specs = getattr(family, "model_specs", [])
        for engine_name, engine_classes in supported_engines.items():
            if engine_name in engine_params:
                continue

            error_reason: Optional[str] = None
            error_type: Optional[str] = None
            error_details: Optional[str] = None
            relevant = False

            for engine_class in engine_classes:
                try:
                    lib_ok, lib_reason, lib_type, lib_details = _normalize_match_result(
                        engine_class.check_lib(),
                        f"Engine {engine_name} library is not installed",
                        "dependency_missing",
                    )
                    if not lib_ok:
                        relevant = True
                        error_reason = lib_reason
                        error_type = lib_type
                        error_details = lib_details
                        break

                    for spec in specs:
                        quantization = getattr(spec, "quantization", None) or "none"
                        match_func = getattr(engine_class, "match_json", None)
                        match_res = (
                            match_func(family, spec, quantization)
                            if callable(match_func)
                            else False
                        )
                        (
                            is_match,
                            reason,
                            m_err_type,
                            m_details,
                        ) = _normalize_match_result(
                            match_res,
                            f"Engine {engine_name} is not compatible with current {engine_type_label} model or environment",
                            "model_compatibility",
                        )
                        if is_match:
                            relevant = False
                            error_reason = None
                            break

                        relevant = True
                        if reason:
                            error_reason = reason
                        if m_err_type:
                            error_type = m_err_type
                        if m_details:
                            error_details = m_details
                        # Return the first failure to avoid later specs overwriting the root cause.
                        break
                    if relevant and error_reason:
                        break
                except Exception as e:
                    relevant = True
                    error_reason = f"Engine {engine_name} is not available: {str(e)}"
                    error_type = "configuration_error"
                    break

            if relevant:
                _append_unavailable_engine(
                    engine_name, error_reason, error_type, error_details
                )

    def _collect_supported_image_engines(
        families: List[Any],
        supported_engines: Dict[str, List[Type[Any]]],
        engine_type_label: str,
    ):
        if not families:
            return
        for engine_name, engine_classes in supported_engines.items():
            if engine_name in engine_params:
                continue

            error_reason: Optional[str] = None
            error_type: Optional[str] = None
            error_details: Optional[str] = None
            relevant = False

            for engine_class in engine_classes:
                try:
                    match_func = getattr(engine_class, "match", None)
                    matched = False
                    last_reason: Optional[str] = None
                    last_type: Optional[str] = None
                    last_details: Optional[str] = None
                    for family in families:
                        match_res = (
                            match_func(family) if callable(match_func) else False
                        )
                        (
                            is_match,
                            reason,
                            m_err_type,
                            m_details,
                        ) = _normalize_match_result(
                            match_res,
                            f"Engine {engine_name} is not compatible with current {engine_type_label} model or environment",
                            "model_compatibility",
                        )
                        if is_match:
                            matched = True
                            break
                        last_reason = reason or last_reason
                        last_type = m_err_type or last_type
                        last_details = m_details or last_details

                    if matched:
                        check_lib = getattr(engine_class, "check_lib", None)
                        lib_ok, lib_reason, lib_type, lib_details = (
                            _normalize_match_result(
                                check_lib() if callable(check_lib) else True,
                                f"Engine {engine_name} library is not installed",
                                "dependency_missing",
                            )
                        )
                        if not lib_ok:
                            relevant = True
                            error_reason = lib_reason
                            error_type = lib_type
                            error_details = lib_details
                        else:
                            relevant = False
                            error_reason = None
                            error_type = None
                            error_details = None
                        break

                    relevant = True
                    error_reason = last_reason
                    error_type = last_type
                    error_details = last_details
                    break
                except Exception as e:
                    relevant = True
                    error_reason = f"Engine {engine_name} is not available: {str(e)}"
                    error_type = "configuration_error"
                    break

            if relevant:
                _append_unavailable_engine(
                    engine_name, error_reason, error_type, error_details
                )

    def _validate_available_image_engines(
        families: List[Any],
        supported_engines: Dict[str, List[Type[Any]]],
        engine_type_label: str,
    ):
        if not families:
            return
        for engine_name, engine_data in list(engine_params.items()):
            if not isinstance(engine_data, list):
                continue
            if engine_name not in supported_engines:
                continue

            matched = False
            error_reason: Optional[str] = None
            error_type: Optional[str] = None
            error_details: Optional[str] = None

            for engine_class in supported_engines[engine_name]:
                try:
                    match_func = getattr(engine_class, "match", None)
                    for family in families:
                        match_res = (
                            match_func(family) if callable(match_func) else False
                        )
                        (
                            is_match,
                            reason,
                            m_err_type,
                            m_details,
                        ) = _normalize_match_result(
                            match_res,
                            f"Engine {engine_name} is not compatible with current {engine_type_label} model or environment",
                            "model_compatibility",
                        )
                        if is_match:
                            matched = True
                            break
                        error_reason = reason or error_reason
                        error_type = m_err_type or error_type
                        error_details = m_details or error_details
                    if matched:
                        check_lib = getattr(engine_class, "check_lib", None)
                        lib_ok, lib_reason, lib_type, lib_details = (
                            _normalize_match_result(
                                check_lib() if callable(check_lib) else True,
                                f"Engine {engine_name} library is not installed",
                                "dependency_missing",
                            )
                        )
                        if not lib_ok:
                            _append_unavailable_engine(
                                engine_name, lib_reason, lib_type, lib_details
                            )
                        break
                except Exception as e:
                    _append_unavailable_engine(
                        engine_name,
                        f"Engine {engine_name} is not available: {str(e)}",
                        "configuration_error",
                        None,
                    )
                    break

            if not matched and engine_name in engine_params:
                _append_unavailable_engine(
                    engine_name,
                    error_reason,
                    error_type,
                    error_details,
                )

    if model_type == "LLM":
        from .llm.llm_family import BUILTIN_LLM_FAMILIES, LLM_ENGINES, SUPPORTED_ENGINES

        if model_name not in LLM_ENGINES:
            return None

        available_engines = deepcopy(LLM_ENGINES[model_name])
        for engine, params in available_engines.items():
            _append_available_engine(engine, params, "llm_class")

        llm_family = next(
            (f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name), None
        )
        _collect_supported_engines(llm_family, SUPPORTED_ENGINES, "LLM")

        return engine_params

    elif model_type == "embedding":
        from .embedding.embed_family import (
            BUILTIN_EMBEDDING_MODELS,
            EMBEDDING_ENGINES,
        )
        from .embedding.embed_family import (
            SUPPORTED_ENGINES as EMBEDDING_SUPPORTED_ENGINES,
        )

        if model_name not in EMBEDDING_ENGINES:
            return None

        available_engines = deepcopy(EMBEDDING_ENGINES[model_name])
        for engine, params in available_engines.items():
            _append_available_engine(engine, params, "embedding_class")

        embedding_family_list = BUILTIN_EMBEDDING_MODELS.get(model_name, [])
        embedding_family = embedding_family_list[0] if embedding_family_list else None
        _collect_supported_engines(
            embedding_family, EMBEDDING_SUPPORTED_ENGINES, "embedding"
        )

        return engine_params

    elif model_type == "rerank":
        from .rerank.rerank_family import BUILTIN_RERANK_MODELS, RERANK_ENGINES
        from .rerank.rerank_family import SUPPORTED_ENGINES as RERANK_SUPPORTED_ENGINES

        if model_name not in RERANK_ENGINES:
            return None

        available_engines = deepcopy(RERANK_ENGINES[model_name])
        for engine, params in available_engines.items():
            _append_available_engine(engine, params, "rerank_class")

        from .rerank.core import RerankModelFamilyV2

        rerank_family_list: List[RerankModelFamilyV2] = BUILTIN_RERANK_MODELS.get(
            model_name, []
        )
        rerank_family = rerank_family_list[0] if rerank_family_list else None
        _collect_supported_engines(rerank_family, RERANK_SUPPORTED_ENGINES, "rerank")

        return engine_params

    elif model_type == "image":
        from .image import BUILTIN_IMAGE_MODELS
        from .image.custom import get_user_defined_images
        from .image.engine_family import (
            IMAGE_ENGINES,
        )
        from .image.engine_family import SUPPORTED_ENGINES as IMAGE_SUPPORTED_ENGINES
        from .image.ocr.ocr_family import (
            OCR_ENGINES,
        )
        from .image.ocr.ocr_family import SUPPORTED_ENGINES as OCR_SUPPORTED_ENGINES

        def _get_image_families(model_name: str, is_ocr: bool) -> List[Any]:
            families: List[Any] = []
            if model_name in BUILTIN_IMAGE_MODELS:
                families.extend(BUILTIN_IMAGE_MODELS[model_name])
            families.extend(
                f for f in get_user_defined_images() if f.model_name == model_name
            )
            if is_ocr:
                return [
                    f
                    for f in families
                    if getattr(f, "model_ability", None)
                    and "ocr" in getattr(f, "model_ability")
                ]
            return [
                f
                for f in families
                if not (
                    getattr(f, "model_ability", None)
                    and "ocr" in getattr(f, "model_ability")
                )
            ]

        if model_name in OCR_ENGINES:
            available_engines = deepcopy(OCR_ENGINES[model_name])
            for engine, params in available_engines.items():
                _append_available_engine(engine, params, "ocr_class")
            ocr_families = _get_image_families(model_name, is_ocr=True)
            _validate_available_image_engines(
                ocr_families, OCR_SUPPORTED_ENGINES, "OCR"
            )
            _collect_supported_image_engines(ocr_families, OCR_SUPPORTED_ENGINES, "OCR")
            return engine_params

        if model_name not in IMAGE_ENGINES:
            return None

        available_engines = deepcopy(IMAGE_ENGINES[model_name])
        for engine, params in available_engines.items():
            _append_available_engine(engine, params, "image_class")
        image_families = _get_image_families(model_name, is_ocr=False)
        _validate_available_image_engines(
            image_families, IMAGE_SUPPORTED_ENGINES, "image"
        )
        _collect_supported_image_engines(
            image_families, IMAGE_SUPPORTED_ENGINES, "image"
        )

        return engine_params

    raise ValueError(
        "Cannot support model_engine for "
        f"{model_type}, only available for LLM, embedding, rerank, image"
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
    base_info = {
        key: value
        for key, value in input_json.items()
        if key not in ("model_src", "model_specs")
    }

    if "model_specs" in input_json:
        for spec in input_json["model_specs"]:
            spec_base = base_info.copy()
            spec_base.update({k: v for k, v in spec.items() if k != "model_src"})
            for model_hub, hub_info in spec["model_src"].items():
                record = spec_base.copy()
                hub_info = hub_info.copy()
                hub_info.pop("model_hub", None)
                record.update(hub_info)
                record["model_hub"] = model_hub
                flattened.append(record)
        return flattened

    for model_hub, hub_info in input_json["model_src"].items():
        record = base_info.copy()
        hub_info = hub_info.copy()
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
                    if isinstance(value, str) and "{quantization}" in value:
                        try:
                            value = value.format(quantization=quant)
                        except Exception:
                            pass
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


def load_downloaded_models_to_dict(
    target_dict: Dict[str, Any], model_type: str, json_filename: str, load_func
):
    """Load downloaded JSON configurations into the specified dictionary.

    Args:
        target_dict: Dictionary to load models into
        model_type: Type of model (e.g., "llm", "embedding", "audio", "image", "rerank")
        json_filename: Name of the JSON file to load
        load_func: Function to load model family from JSON
    """
    from ..constants import XINFERENCE_MODEL_DIR

    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", model_type)
    json_file_path = os.path.join(builtin_dir, json_filename)

    try:
        load_func(json_file_path, target_dict)
    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to load downloaded {model_type} models from {json_file_path}: {e}"
        )


def merge_models_by_timestamp(
    built_in_models: Dict[str, List[Any]], user_models: Dict[str, List[Any]]
) -> Dict[str, List[Any]]:
    """Merge built-in and user models, keeping the latest version based on updated_at.

    Args:
        built_in_models: Dictionary of built-in models
        user_models: Dictionary of user-defined models

    Returns:
        Merged dictionary with latest models based on updated_at timestamp
    """
    merged_models = {}

    # First, add all built-in models
    for model_name, model_list in built_in_models.items():
        merged_models[model_name] = model_list.copy()

    # Then merge with user models, keeping the latest based on updated_at
    for model_name, user_model_list in user_models.items():
        if model_name not in merged_models:
            # New model from user, just add it
            merged_models[model_name] = user_model_list.copy()
        else:
            # Existing model, need to compare and merge based on updated_at
            built_in_list = merged_models[model_name]

            # Create a mapping of updated_at to model for comparison
            all_models = []

            # Add built-in models
            for model in built_in_list:
                all_models.append((model.updated_at, model))

            # Add user models
            for model in user_model_list:
                all_models.append((model.updated_at, model))

            # Sort by updated_at (newest first) and keep the latest
            all_models.sort(key=lambda x: x[0], reverse=True)

            # Keep the latest version(s) - in case there are multiple with the same updated_at
            latest_updated_at = all_models[0][0]
            latest_models = [
                model
                for updated_at, model in all_models
                if updated_at == latest_updated_at
            ]

            merged_models[model_name] = latest_models

    return merged_models


def install_models_with_merge(
    built_in_dict: Dict[str, Any],
    builtin_json_file: str,
    user_model_type: str,
    user_json_filename: str,
    has_downloaded_models_func,
    load_model_family_func,
) -> None:
    """Install models with intelligent merging based on timestamps.

    Args:
        built_in_dict: Dictionary to store built-in models
        builtin_json_file: Path to built-in JSON file (relative to the model module)
        user_model_type: Type of model for user models (e.g., "llm", "embedding")
        user_json_filename: Name of user JSON file
        has_downloaded_models_func: Function to check if user models exist
        load_model_family_func: Function to load model family from JSON
    """
    import os.path

    # Always load built-in models first to ensure we have the latest models
    # For built-in models, use the path relative to the current model module
    current_dir = os.path.dirname(
        os.path.abspath(load_model_family_func.__code__.co_filename)
    )
    builtin_json_path = os.path.join(current_dir, builtin_json_file)
    load_model_family_func(builtin_json_path, built_in_dict)

    # Then load user-defined models and merge with built-in models
    if has_downloaded_models_func():
        user_models: Dict[str, Any] = {}
        load_downloaded_models_to_dict(
            user_models, user_model_type, user_json_filename, load_model_family_func
        )

        # Create a copy of built-in models for merging
        built_in_models_copy = dict(built_in_dict)

        # Merge models, keeping the latest version based on updated_at
        merged_models = merge_models_by_timestamp(built_in_models_copy, user_models)

        # Update the dictionary with merged results
        built_in_dict.clear()
        built_in_dict.update(merged_models)
