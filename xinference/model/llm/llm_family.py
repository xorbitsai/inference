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
import platform
import shutil
from threading import Lock
from typing import List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from . import LLM

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
DEFAULT_CONTEXT_LENGTH = 2048


class GgmlLLMSpecV1(BaseModel):
    model_format: Literal["ggmlv3", "ggufv2"]
    model_size_in_billions: int
    quantizations: List[str]
    model_id: str
    model_file_name_template: str
    model_uri: Optional[str]
    model_revision: Optional[str]


class PytorchLLMSpecV1(BaseModel):
    model_format: Literal["pytorch"]
    model_size_in_billions: int
    quantizations: List[str]
    model_id: str
    model_uri: Optional[str]
    model_revision: Optional[str]


class PromptStyleV1(BaseModel):
    style_name: str
    system_prompt: str = ""
    roles: List[str]
    intra_message_sep: str = ""
    inter_message_sep: str = ""
    stop: Optional[List[str]]
    stop_token_ids: Optional[List[int]]


class LLMFamilyV1(BaseModel):
    version: Literal[1]
    context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH
    model_name: str
    model_lang: List[Literal["en", "zh"]]
    model_ability: List[Literal["embed", "generate", "chat"]]
    model_description: Optional[str]
    model_specs: List["LLMSpecV1"]
    prompt_style: Optional["PromptStyleV1"]


LLMSpecV1 = Annotated[
    Union[GgmlLLMSpecV1, PytorchLLMSpecV1],
    Field(discriminator="model_format"),
]

LLMFamilyV1.update_forward_refs()


LLM_CLASSES: List[Type[LLM]] = []

BUILTIN_LLM_FAMILIES: List["LLMFamilyV1"] = []

UD_LLM_FAMILIES: List["LLMFamilyV1"] = []

UD_LLM_FAMILIES_LOCK = Lock()


def is_locale_chinese_simplified() -> bool:
    import locale

    try:
        lang, _ = locale.getdefaultlocale()
        return lang == "zh_CN"
    except:
        return False


def download_from_self_hosted_storage() -> bool:
    from ...constants import XINFERENCE_ENV_MODEL_SRC

    return (
        is_locale_chinese_simplified()
        or os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "xorbits"
    )


def get_legacy_cache_path(
    model_name: str,
    model_format: str,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
) -> str:
    full_name = f"{model_name}-{model_format}-{model_size_in_billions}b-{quantization}"
    return os.path.join(XINFERENCE_CACHE_DIR, full_name, "model.bin")


def cache(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    legacy_cache_path = get_legacy_cache_path(
        llm_family.model_name,
        llm_spec.model_format,
        llm_spec.model_size_in_billions,
        quantization,
    )
    if os.path.exists(legacy_cache_path):
        logger.info("Legacy cache path exists: %s", legacy_cache_path)
        return os.path.dirname(legacy_cache_path)
    elif download_from_self_hosted_storage() and is_self_hosted(llm_family, llm_spec):
        logger.info(f"Caching from self-hosted storage")
        return cache_from_self_hosted_storage(llm_family, llm_spec, quantization)
    else:
        if llm_spec.model_uri is not None:
            logger.info(f"Caching from URI: {llm_spec.model_uri}")
            return cache_from_uri(llm_family, llm_spec, quantization)
        else:
            logger.info(f"Caching from Hugging Face: {llm_spec.model_id}")
            return cache_from_huggingface(llm_family, llm_spec, quantization)


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


SUPPORTED_SCHEMES = ["s3"]


class AWSRegion:
    def __init__(self, region: str):
        self.region = region
        self.original_aws_default_region = None

    def __enter__(self):
        if "AWS_DEFAULT_REGION" in os.environ:
            self.original_aws_default_region = os.environ["AWS_DEFAULT_REGION"]
        os.environ["AWS_DEFAULT_REGION"] = self.region

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_aws_default_region:
            os.environ["AWS_DEFAULT_REGION"] = self.original_aws_default_region
        else:
            del os.environ["AWS_DEFAULT_REGION"]


def is_self_hosted(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
):
    from fsspec import AbstractFileSystem, filesystem

    with AWSRegion("cn-northwest-1"):
        src_fs: AbstractFileSystem = filesystem("s3", anon=True)
        model_dir = (
            f"/xinference-models/llm/"
            f"{llm_family.model_name}-{llm_spec.model_format}-{llm_spec.model_size_in_billions}b"
        )
        return src_fs.exists(model_dir)


def cache_from_self_hosted_storage(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    with AWSRegion("cn-northwest-1"):
        llm_spec = llm_spec.copy()
        llm_spec.model_uri = (
            f"s3://xinference-models/llm/"
            f"{llm_family.model_name}-{llm_spec.model_format}-{llm_spec.model_size_in_billions}b"
        )

        return cache_from_uri(
            llm_family, llm_spec, quantization, self_hosted_storage=True
        )


def cache_from_uri(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
    self_hosted_storage: bool = False,
) -> str:
    from fsspec import AbstractFileSystem, filesystem

    def copy(
        _src_fs: "AbstractFileSystem",
        _src_path: str,
        dst_fs: "AbstractFileSystem",
        dst_path: str,
        max_attempt: int = 3,
    ):
        from tqdm import tqdm

        for attempt in range(max_attempt):
            logger.info(f"Copy from {_src_path} to {dst_path}, attempt: {attempt}")
            try:
                with _src_fs.open(_src_path, "rb") as src_file:
                    file_size = _src_fs.info(src_path)["size"]

                    dst_fs.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    with dst_fs.open(dst_path, "wb") as dst_file:
                        chunk_size = 1024 * 1024  # 1 MB

                        with tqdm(
                            total=file_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=_src_path,
                        ) as pbar:
                            while True:
                                chunk = src_file.read(chunk_size)
                                if not chunk:
                                    break
                                dst_file.write(chunk)
                                pbar.update(len(chunk))
                logger.info(
                    f"Copy from {_src_path} to {dst_path} finished, attempt: {attempt}"
                )
                break
            except:
                logger.error(
                    f"Failed to copy from {_src_path} to {dst_path} on attempt {attempt + 1}",
                    exc_info=True,
                )
                if attempt + 1 == max_attempt:
                    raise

    cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b"
    )
    cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name))

    assert llm_spec.model_uri is not None
    src_scheme, src_root = parse_uri(llm_spec.model_uri)
    if src_root.endswith("/"):
        # remove trailing path separator.
        src_root = src_root[:-1]

    if src_scheme == "file":
        if not os.path.isabs(src_root):
            raise ValueError(
                f"Model URI cannot be a relative path: {llm_spec.model_uri}"
            )
        os.makedirs(XINFERENCE_CACHE_DIR, exist_ok=True)
        if os.path.exists(cache_dir):
            logger.info(f"Cache {cache_dir} exists")
            return cache_dir
        else:
            os.symlink(src_root, cache_dir, target_is_directory=True)
        return cache_dir
    elif src_scheme in SUPPORTED_SCHEMES:
        # use anonymous connection for self-hosted storage.
        src_fs: AbstractFileSystem = filesystem(src_scheme, anon=self_hosted_storage)
        local_fs: AbstractFileSystem = filesystem("file")

        files_to_download = []
        if llm_spec.model_format == "pytorch":
            if os.path.exists(cache_dir):
                logger.info(f"Cache {cache_dir} exists")
                return cache_dir
            else:
                os.makedirs(cache_dir, exist_ok=True)

            for path, _, files in src_fs.walk(llm_spec.model_uri):
                for file in files:
                    src_path = f"{path}/{file}"
                    local_path = src_path.replace(src_root, cache_dir)
                    files_to_download.append((src_path, local_path))
        elif llm_spec.model_format == "ggmlv3":
            file = llm_spec.model_file_name_template.format(quantization=quantization)
            if os.path.exists(os.path.join(cache_dir, file)):
                logger.info(f"Cache {os.path.join(cache_dir, file)} exists")
                return cache_dir
            else:
                os.makedirs(cache_dir, exist_ok=True)

            src_path = f"{src_root}/{file}"
            local_path = f"{cache_dir}/{file}"
            files_to_download.append((src_path, local_path))
        else:
            raise ValueError(f"Unsupported model format: {llm_spec.model_format}")

        from concurrent.futures import ThreadPoolExecutor

        failed = False
        with ThreadPoolExecutor(max_workers=min(len(files_to_download), 4)) as executor:
            futures = [
                (
                    src_path,
                    executor.submit(copy, src_fs, src_path, local_fs, local_path),
                )
                for src_path, local_path in files_to_download
            ]
            for src_path, future in futures:
                if failed:
                    future.cancel()
                else:
                    try:
                        future.result()
                    except:
                        logger.error(f"Download {src_path} failed", exc_info=True)
                        failed = True

        if failed:
            logger.warning(f"Removing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to download model '{llm_family.model_name}' "
                f"(size: {llm_spec.model_size_in_billions}, format: {llm_spec.model_format})"
            )
        return cache_dir
    else:
        raise ValueError(f"Unsupported URL scheme: {src_scheme}")


def cache_from_huggingface(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Hugging Face. Return the cache directory.
    """
    import huggingface_hub

    cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b"
    )
    cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if llm_spec.model_format == "pytorch":
        assert isinstance(llm_spec, PytorchLLMSpecV1)

        for current_attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                huggingface_hub.snapshot_download(
                    llm_spec.model_id,
                    revision=llm_spec.model_revision,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=True,
                )
                break
            except huggingface_hub.utils.LocalEntryNotFoundError:
                remaining_attempts = MAX_ATTEMPTS - current_attempt
                logger.warning(
                    f"Attempt {current_attempt} failed. Remaining attempts: {remaining_attempts}"
                )

        else:
            raise RuntimeError(
                f"Failed to download model '{llm_family.model_name}' "
                f"(size: {llm_spec.model_size_in_billions}, format: {llm_spec.model_format}) "
                f"after multiple retries"
            )

    elif llm_spec.model_format in ["ggmlv3", "ggufv2"]:
        assert isinstance(llm_spec, GgmlLLMSpecV1)
        file_name = llm_spec.model_file_name_template.format(quantization=quantization)

        for current_attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                huggingface_hub.hf_hub_download(
                    llm_spec.model_id,
                    revision=llm_spec.model_revision,
                    filename=file_name,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=True,
                )
                break
            except huggingface_hub.utils.LocalEntryNotFoundError:
                remaining_attempts = MAX_ATTEMPTS - current_attempt
                logger.warning(
                    f"Attempt {current_attempt} failed. Remaining attempts: {remaining_attempts}"
                )

        else:
            raise RuntimeError(
                f"Failed to download model '{llm_family.model_name}' "
                f"(size: {llm_spec.model_size_in_billions}, format: {llm_spec.model_format}) "
                f"after multiple retries"
            )
    else:
        raise ValueError(f"Unsupported model format: {llm_spec.model_format}")

    return cache_dir


def _is_linux():
    return platform.system() == "Linux"


def _has_cuda_device():
    # `cuda_count` method already contains the logic for the
    # number of GPUs specified by `CUDA_VISIBLE_DEVICES`.
    from xorbits._mars.resource import cuda_count

    return cuda_count() > 0


def get_user_defined_llm_families():
    with UD_LLM_FAMILIES_LOCK:
        return UD_LLM_FAMILIES.copy()


def match_llm(
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
    is_local_deployment: bool = False,
) -> Optional[Tuple[LLMFamilyV1, LLMSpecV1, str]]:
    """
    Find an LLM family, spec, and quantization that satisfy given criteria.
    """
    user_defined_llm_families = get_user_defined_llm_families()

    def _match_quantization(q: Union[str, None], quantizations: List[str]):
        # Currently, the quantization name could include both uppercase and lowercase letters,
        # so it is necessary to ensure that the case sensitivity does not
        # affect the matching results.
        if q is None:
            return q
        for quant in quantizations:
            if q.lower() == quant.lower():
                return quant

    for family in BUILTIN_LLM_FAMILIES + user_defined_llm_families:
        if model_name != family.model_name:
            continue
        for spec in family.model_specs:
            matched_quantization = _match_quantization(quantization, spec.quantizations)
            if (
                model_format
                and model_format != spec.model_format
                or model_size_in_billions
                and model_size_in_billions != spec.model_size_in_billions
                or quantization
                and matched_quantization is None
            ):
                continue
            if quantization:
                return family, spec, matched_quantization
            else:
                # by default, choose the most coarse-grained quantization.
                # TODO: too hacky.
                quantizations = spec.quantizations
                quantizations.sort()
                for q in quantizations:
                    if (
                        is_local_deployment
                        and not (_is_linux() and _has_cuda_device())
                        and q == "4-bit"
                    ):
                        logger.warning(
                            "Skipping %s for non-linux or non-cuda local deployment .",
                            q,
                        )
                        continue
                    return family, spec, q
    return None


def register_llm(llm_family: LLMFamilyV1, persist: bool):
    from .utils import is_valid_model_name

    if not is_valid_model_name(llm_family.model_name):
        raise ValueError(
            f"Invalid model name {llm_family.model_name}. The model name must start with a letter"
            f" or a digit, and can only contain letters, digits, underscores, or dashes."
        )

    with UD_LLM_FAMILIES_LOCK:
        for family in BUILTIN_LLM_FAMILIES + UD_LLM_FAMILIES:
            if llm_family.model_name == family.model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {family.model_name}"
                )

        UD_LLM_FAMILIES.append(llm_family)

    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "llm", f"{llm_family.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(llm_family.json())


def unregister_llm(model_name: str):
    with UD_LLM_FAMILIES_LOCK:
        llm_family = None
        for i, f in enumerate(UD_LLM_FAMILIES):
            if f.model_name == model_name:
                llm_family = f
                break
        if llm_family:
            UD_LLM_FAMILIES.remove(llm_family)

            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "llm", f"{llm_family.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            llm_spec = llm_family.model_specs[0]
            cache_dir_name = (
                f"{llm_family.model_name}-{llm_spec.model_format}"
                f"-{llm_spec.model_size_in_billions}b"
            )
            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"Remove the cache of user-defined model {llm_family.model_name}. "
                    f"Cache directory: {cache_dir}"
                )
                if os.path.islink(cache_dir):
                    os.remove(cache_dir)
                else:
                    logger.warning(
                        f"Cache directory is not a soft link, please remove it manually."
                    )
        else:
            raise ValueError(f"Model {model_name} not found")


def match_llm_cls(family: LLMFamilyV1, llm_spec: "LLMSpecV1") -> Optional[Type[LLM]]:
    """
    Find an LLM implementation for given LLM family and spec.
    """
    for cls in LLM_CLASSES:
        if cls.match(family, llm_spec):
            return cls
    return None
