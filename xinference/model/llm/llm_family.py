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
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from typing_extensions import Annotated, Literal

from ..._compat import (
    ROOT_KEY,
    BaseModel,
    ErrorWrapper,
    Field,
    Protocol,
    StrBytes,
    ValidationError,
    load_str_bytes,
    validator,
)
from ...constants import (
    XINFERENCE_CACHE_DIR,
    XINFERENCE_ENV_CSG_TOKEN,
    XINFERENCE_MODEL_DIR,
)
from ..utils import (
    IS_NEW_HUGGINGFACE_HUB,
    create_symlink,
    download_from_csghub,
    download_from_modelscope,
    is_valid_model_uri,
    parse_uri,
    retry_download,
    symlink_local_file,
    valid_model_revision,
)
from . import LLM

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LENGTH = 2048
BUILTIN_LLM_PROMPT_STYLE: Dict[str, "PromptStyleV1"] = {}
BUILTIN_LLM_MODEL_CHAT_FAMILIES: Set[str] = set()
BUILTIN_LLM_MODEL_GENERATE_FAMILIES: Set[str] = set()
BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES: Set[str] = set()


class GgmlLLMSpecV1(BaseModel):
    model_format: Literal["ggmlv3", "ggufv2"]
    # Must in order that `str` first, then `int`
    model_size_in_billions: Union[str, int]
    quantizations: List[str]
    model_id: Optional[str]
    model_file_name_template: str
    model_file_name_split_template: Optional[str]
    quantization_parts: Optional[Dict[str, List[str]]]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]

    @validator("model_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        if isinstance(v, str):
            if (
                "_" in v
            ):  # for example, "1_8" just returns "1_8", otherwise int("1_8") returns 18
                return v
            else:
                return int(v)
        return v


class PytorchLLMSpecV1(BaseModel):
    model_format: Literal["pytorch", "gptq", "awq"]
    # Must in order that `str` first, then `int`
    model_size_in_billions: Union[str, int]
    quantizations: List[str]
    model_id: Optional[str]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]

    @validator("model_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        if isinstance(v, str):
            if (
                "_" in v
            ):  # for example, "1_8" just returns "1_8", otherwise int("1_8") returns 18
                return v
            else:
                return int(v)
        return v


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
    model_lang: List[str]
    model_ability: List[Literal["embed", "generate", "chat", "tools", "vision"]]
    model_description: Optional[str]
    # reason for not required str here: legacy registration
    model_family: Optional[str]
    model_specs: List["LLMSpecV1"]
    prompt_style: Optional["PromptStyleV1"]


class CustomLLMFamilyV1(LLMFamilyV1):
    prompt_style: Optional[Union["PromptStyleV1", str]]  # type: ignore

    @classmethod
    def parse_raw(
        cls: Any,
        b: StrBytes,
        *,
        content_type: Optional[str] = None,
        encoding: str = "utf8",
        proto: Protocol = None,
        allow_pickle: bool = False,
    ) -> LLMFamilyV1:
        # See source code of BaseModel.parse_raw
        try:
            obj = load_str_bytes(
                b,
                proto=proto,
                content_type=content_type,
                encoding=encoding,
                allow_pickle=allow_pickle,
                json_loads=cls.__config__.json_loads,
            )
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            raise ValidationError([ErrorWrapper(e, loc=ROOT_KEY)], cls)
        llm_spec: CustomLLMFamilyV1 = cls.parse_obj(obj)

        # check model_family
        if llm_spec.model_family is None:
            raise ValueError(
                f"You must specify `model_family` when registering custom LLM models."
            )
        assert isinstance(llm_spec.model_family, str)
        if (
            llm_spec.model_family != "other"
            and "chat" in llm_spec.model_ability
            and llm_spec.model_family not in BUILTIN_LLM_MODEL_CHAT_FAMILIES
        ):
            raise ValueError(
                f"`model_family` for chat model must be `other` or one of the following values: \n"
                f"{', '.join(list(BUILTIN_LLM_MODEL_CHAT_FAMILIES))}"
            )
        if (
            llm_spec.model_family != "other"
            and "tools" in llm_spec.model_ability
            and llm_spec.model_family not in BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES
        ):
            raise ValueError(
                f"`model_family` for tool call model must be `other` or one of the following values: \n"
                f"{', '.join(list(BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES))}"
            )
        if (
            llm_spec.model_family != "other"
            and "chat" not in llm_spec.model_ability
            and llm_spec.model_family not in BUILTIN_LLM_MODEL_GENERATE_FAMILIES
        ):
            raise ValueError(
                f"`model_family` for generate model must be `other` or one of the following values: \n"
                f"{', '.join(list(BUILTIN_LLM_MODEL_GENERATE_FAMILIES))}"
            )
        # set prompt style when it is the builtin model family
        if (
            llm_spec.prompt_style is None
            and llm_spec.model_family != "other"
            and "chat" in llm_spec.model_ability
        ):
            llm_spec.prompt_style = llm_spec.model_family

        # handle prompt style when user choose existing style
        if llm_spec.prompt_style is not None and isinstance(llm_spec.prompt_style, str):
            prompt_style_name = llm_spec.prompt_style
            if prompt_style_name not in BUILTIN_LLM_PROMPT_STYLE:
                raise ValueError(
                    f"Xinference does not support the prompt style name: {prompt_style_name}"
                )
            llm_spec.prompt_style = BUILTIN_LLM_PROMPT_STYLE[prompt_style_name]

        # check model ability, registering LLM only provides generate and chat
        # but for vision models, we add back the abilities so that
        # gradio chat interface can be generated properly
        if (
            llm_spec.model_family != "other"
            and llm_spec.model_family
            in {
                family.model_name
                for family in BUILTIN_LLM_FAMILIES
                if "vision" in family.model_ability
            }
            and "vision" not in llm_spec.model_ability
        ):
            llm_spec.model_ability.append("vision")

        return llm_spec


LLMSpecV1 = Annotated[
    Union[GgmlLLMSpecV1, PytorchLLMSpecV1],
    Field(discriminator="model_format"),
]

LLMFamilyV1.update_forward_refs()
CustomLLMFamilyV1.update_forward_refs()


LLAMA_CLASSES: List[Type[LLM]] = []

BUILTIN_LLM_FAMILIES: List["LLMFamilyV1"] = []
BUILTIN_MODELSCOPE_LLM_FAMILIES: List["LLMFamilyV1"] = []
BUILTIN_CSGHUB_LLM_FAMILIES: List["LLMFamilyV1"] = []

SGLANG_CLASSES: List[Type[LLM]] = []
TRANSFORMERS_CLASSES: List[Type[LLM]] = []

UD_LLM_FAMILIES: List["LLMFamilyV1"] = []

UD_LLM_FAMILIES_LOCK = Lock()

VLLM_CLASSES: List[Type[LLM]] = []

LLM_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[LLM]]] = {}

LLM_LAUNCH_VERSIONS: Dict[str, List[str]] = {}


def download_from_self_hosted_storage() -> bool:
    from ...constants import XINFERENCE_ENV_MODEL_SRC

    return os.environ.get(XINFERENCE_ENV_MODEL_SRC) == "xorbits"


def get_legacy_cache_path(
    model_name: str,
    model_format: str,
    model_size_in_billions: Optional[Union[str, int]] = None,
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
            if llm_spec.model_hub == "huggingface":
                logger.info(f"Caching from Hugging Face: {llm_spec.model_id}")
                return cache_from_huggingface(llm_family, llm_spec, quantization)
            elif llm_spec.model_hub == "modelscope":
                logger.info(f"Caching from Modelscope: {llm_spec.model_id}")
                return cache_from_modelscope(llm_family, llm_spec, quantization)
            elif llm_spec.model_hub == "csghub":
                logger.info(f"Caching from CSGHub: {llm_spec.model_id}")
                return cache_from_csghub(llm_family, llm_spec, quantization)
            else:
                raise ValueError(f"Unknown model hub: {llm_spec.model_hub}")


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

    from ..utils import copy_from_src_to_dst

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
                    executor.submit(
                        copy_from_src_to_dst, src_fs, src_path, local_fs, local_path
                    ),
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


def cache_model_config(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
):
    """Download model config.json into cache_dir,
    returns local filepath
    """
    cache_dir = _get_cache_dir_for_model_mem(llm_family, llm_spec)
    config_file = os.path.join(cache_dir, "config.json")
    if not os.path.islink(config_file) and not os.path.exists(config_file):
        os.makedirs(cache_dir, exist_ok=True)
        if llm_spec.model_hub == "huggingface":
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=llm_spec.model_id, filename="config.json", local_dir=cache_dir
            )
        else:
            from modelscope.hub.file_download import model_file_download

            download_path = model_file_download(
                model_id=llm_spec.model_id, file_path="config.json"
            )
            os.symlink(download_path, config_file)
    return config_file


def _get_cache_dir_for_model_mem(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    create_if_not_exist=True,
):
    """
    For cal-model-mem only. (might called from supervisor / cli)
    Temporary use separate dir from worker's cache_dir, due to issue of different style of symlink.
    """
    quant_suffix = ""
    for q in llm_spec.quantizations:
        if llm_spec.model_id and q in llm_spec.model_id:
            quant_suffix = q
            break
    cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b"
    )
    if quant_suffix:
        cache_dir_name += f"-{quant_suffix}"
    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, "model_mem", cache_dir_name)
    )
    if create_if_not_exist and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_cache_dir(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    create_if_not_exist=True,
):
    # If the model id contains quantization, then we should give each
    # quantization a dedicated cache dir.
    quant_suffix = ""
    for q in llm_spec.quantizations:
        if llm_spec.model_id and q in llm_spec.model_id:
            quant_suffix = q
            break
    cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b"
    )
    if quant_suffix:
        cache_dir_name += f"-{quant_suffix}"
    cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name))
    if create_if_not_exist and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_meta_path(
    cache_dir: str,
    model_format: str,
    model_hub: str,
    quantization: Optional[str] = None,
):
    if model_format == "pytorch":
        if model_hub == "huggingface":
            return os.path.join(cache_dir, "__valid_download")
        else:
            return os.path.join(cache_dir, f"__valid_download_{model_hub}")
    elif model_format in ["ggmlv3", "ggufv2", "gptq", "awq"]:
        assert quantization is not None
        if model_hub == "huggingface":
            return os.path.join(cache_dir, f"__valid_download_{quantization}")
        else:
            return os.path.join(
                cache_dir, f"__valid_download_{model_hub}_{quantization}"
            )
    else:
        raise ValueError(f"Unsupported format: {model_format}")


def _skip_download(
    cache_dir: str,
    model_format: str,
    model_hub: str,
    model_revision: Optional[str],
    quantization: Optional[str] = None,
) -> bool:
    if model_format == "pytorch":
        model_hub_to_meta_path = {
            "huggingface": _get_meta_path(
                cache_dir, model_format, "huggingface", quantization
            ),
            "modelscope": _get_meta_path(
                cache_dir, model_format, "modelscope", quantization
            ),
            "csghub": _get_meta_path(cache_dir, model_format, "csghub", quantization),
        }
        if valid_model_revision(model_hub_to_meta_path[model_hub], model_revision):
            logger.info(f"Cache {cache_dir} exists")
            return True
        else:
            for hub, meta_path in model_hub_to_meta_path.items():
                if hub != model_hub and os.path.exists(meta_path):
                    # PyTorch models from modelscope can also be loaded by transformers.
                    logger.warning(f"Cache {cache_dir} exists, but it was from {hub}")
                    return True
            return False
    elif model_format in ["ggmlv3", "ggufv2", "gptq", "awq"]:
        assert quantization is not None
        return os.path.exists(
            _get_meta_path(cache_dir, model_format, model_hub, quantization)
        )
    else:
        raise ValueError(f"Unsupported format: {model_format}")


def _generate_meta_file(
    meta_path: str,
    llm_family: "LLMFamilyV1",
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
):
    assert not valid_model_revision(
        meta_path, llm_spec.model_revision
    ), f"meta file {meta_path} should not be valid"
    with open(meta_path, "w") as f:
        import json

        from .core import LLMDescription

        desc = LLMDescription(None, None, llm_family, llm_spec, quantization)
        json.dump(desc.to_dict(), f)


def _generate_model_file_names(
    llm_spec: "LLMSpecV1", quantization: Optional[str] = None
) -> Tuple[List[str], str, bool]:
    file_names = []
    final_file_name = llm_spec.model_file_name_template.format(
        quantization=quantization
    )
    need_merge = False

    if llm_spec.quantization_parts is None:
        file_names.append(final_file_name)
    elif quantization is not None and quantization in llm_spec.quantization_parts:
        parts = llm_spec.quantization_parts[quantization]
        need_merge = True

        logger.info(
            f"Model {llm_spec.model_id} {llm_spec.model_format} {quantization} has {len(parts)} parts."
        )

        if llm_spec.model_file_name_split_template is None:
            raise ValueError(
                f"No model_file_name_split_template for model spec {llm_spec.model_id}"
            )

        for part in parts:
            file_name = llm_spec.model_file_name_split_template.format(
                quantization=quantization, part=part
            )
            file_names.append(file_name)

    return file_names, final_file_name, need_merge


def _merge_cached_files(
    cache_dir: str, input_file_names: List[str], output_file_name: str
):
    with open(os.path.join(cache_dir, output_file_name), "wb") as output_file:
        for file_name in input_file_names:
            logger.info(f"Merging file {file_name} into {output_file_name} ...")

            with open(os.path.join(cache_dir, file_name), "rb") as input_file:
                shutil.copyfileobj(input_file, output_file)

    logger.info(f"Merge complete.")


def cache_from_csghub(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from CSGHub. Return the cache directory.
    """
    from pycsghub.file_download import file_download
    from pycsghub.snapshot_download import snapshot_download

    cache_dir = _get_cache_dir(llm_family, llm_spec)

    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    if llm_spec.model_format in ["pytorch", "gptq", "awq"]:
        download_dir = retry_download(
            snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            endpoint="https://hub-stg.opencsg.com",
            token=os.environ.get(XINFERENCE_ENV_CSG_TOKEN),
        )
        create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggmlv3", "ggufv2"]:
        file_names, final_file_name, need_merge = _generate_model_file_names(
            llm_spec, quantization
        )

        for filename in file_names:
            download_path = retry_download(
                file_download,
                llm_family.model_name,
                {
                    "model_size": llm_spec.model_size_in_billions,
                    "model_format": llm_spec.model_format,
                },
                llm_spec.model_id,
                file_name=filename,
                endpoint="https://hub-stg.opencsg.com",
                token=os.environ.get(XINFERENCE_ENV_CSG_TOKEN),
            )
            symlink_local_file(download_path, cache_dir, filename)

        if need_merge:
            _merge_cached_files(cache_dir, file_names, final_file_name)
    else:
        raise ValueError(f"Unsupported format: {llm_spec.model_format}")

    meta_path = _get_meta_path(
        cache_dir, llm_spec.model_format, llm_spec.model_hub, quantization
    )
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir


def cache_from_modelscope(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Modelscope. Return the cache directory.
    """
    from modelscope.hub.file_download import model_file_download
    from modelscope.hub.snapshot_download import snapshot_download

    cache_dir = _get_cache_dir(llm_family, llm_spec)
    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    if llm_spec.model_format in ["pytorch", "gptq", "awq"]:
        download_dir = retry_download(
            snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            revision=llm_spec.model_revision,
        )
        create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggmlv3", "ggufv2"]:
        file_names, final_file_name, need_merge = _generate_model_file_names(
            llm_spec, quantization
        )

        for filename in file_names:
            download_path = retry_download(
                model_file_download,
                llm_family.model_name,
                {
                    "model_size": llm_spec.model_size_in_billions,
                    "model_format": llm_spec.model_format,
                },
                llm_spec.model_id,
                filename,
                revision=llm_spec.model_revision,
            )
            symlink_local_file(download_path, cache_dir, filename)

        if need_merge:
            _merge_cached_files(cache_dir, file_names, final_file_name)
    else:
        raise ValueError(f"Unsupported format: {llm_spec.model_format}")

    meta_path = _get_meta_path(
        cache_dir, llm_spec.model_format, llm_spec.model_hub, quantization
    )
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir


def cache_from_huggingface(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Hugging Face. Return the cache directory.
    """
    import huggingface_hub

    cache_dir = _get_cache_dir(llm_family, llm_spec)
    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    use_symlinks = {}
    if not IS_NEW_HUGGINGFACE_HUB:
        use_symlinks = {"local_dir_use_symlinks": True, "local_dir": cache_dir}

    if llm_spec.model_format in ["pytorch", "gptq", "awq"]:
        assert isinstance(llm_spec, PytorchLLMSpecV1)
        download_dir = retry_download(
            huggingface_hub.snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            revision=llm_spec.model_revision,
            **use_symlinks,
        )
        if IS_NEW_HUGGINGFACE_HUB:
            create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggmlv3", "ggufv2"]:
        assert isinstance(llm_spec, GgmlLLMSpecV1)
        file_names, final_file_name, need_merge = _generate_model_file_names(
            llm_spec, quantization
        )

        for file_name in file_names:
            download_file_path = retry_download(
                huggingface_hub.hf_hub_download,
                llm_family.model_name,
                {
                    "model_size": llm_spec.model_size_in_billions,
                    "model_format": llm_spec.model_format,
                },
                llm_spec.model_id,
                revision=llm_spec.model_revision,
                filename=file_name,
                **use_symlinks,
            )
            if IS_NEW_HUGGINGFACE_HUB:
                symlink_local_file(download_file_path, cache_dir, file_name)

        if need_merge:
            _merge_cached_files(cache_dir, file_names, final_file_name)
    else:
        raise ValueError(f"Unsupported model format: {llm_spec.model_format}")

    meta_path = _get_meta_path(
        cache_dir, llm_spec.model_format, llm_spec.model_hub, quantization
    )
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir


def _check_revision(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    builtin: list,
    meta_path: str,
) -> bool:
    for family in builtin:
        if llm_family.model_name == family.model_name:
            specs = family.model_specs
            for spec in specs:
                if (
                    spec.model_format == "pytorch"
                    and spec.model_size_in_billions == llm_spec.model_size_in_billions
                ):
                    return valid_model_revision(meta_path, spec.model_revision)
    return False


def get_cache_status(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
) -> Union[bool, List[bool]]:
    """
    When calling this function from above, `llm_family` is constructed only from BUILTIN_LLM_FAMILIES,
    so we should check both huggingface and modelscope cache files.
    """
    cache_dir = _get_cache_dir(llm_family, llm_spec, create_if_not_exist=False)
    # check revision for pytorch model
    if llm_spec.model_format == "pytorch":
        hf_meta_path = _get_meta_path(cache_dir, "pytorch", "huggingface", "none")
        ms_meta_path = _get_meta_path(cache_dir, "pytorch", "modelscope", "none")
        revisions = [
            _check_revision(llm_family, llm_spec, BUILTIN_LLM_FAMILIES, hf_meta_path),
            _check_revision(
                llm_family, llm_spec, BUILTIN_MODELSCOPE_LLM_FAMILIES, ms_meta_path
            ),
        ]
        return any(revisions)
    # just check meta file for ggml and gptq model
    elif llm_spec.model_format in ["ggmlv3", "ggufv2", "gptq", "awq"]:
        ret = []
        for q in llm_spec.quantizations:
            assert q is not None
            hf_meta_path = _get_meta_path(
                cache_dir, llm_spec.model_format, "huggingface", q
            )
            ms_meta_path = _get_meta_path(
                cache_dir, llm_spec.model_format, "modelscope", q
            )
            results = [os.path.exists(hf_meta_path), os.path.exists(ms_meta_path)]
            ret.append(any(results))
        return ret
    else:
        raise ValueError(f"Unsupported model format: {llm_spec.model_format}")


def _is_linux():
    return platform.system() == "Linux"


def _has_cuda_device():
    # `cuda_count` method already contains the logic for the
    # number of GPUs specified by `CUDA_VISIBLE_DEVICES`.
    from ...utils import cuda_count

    return cuda_count() > 0


def get_user_defined_llm_families():
    with UD_LLM_FAMILIES_LOCK:
        return UD_LLM_FAMILIES.copy()


def match_model_size(
    model_size: Union[int, str], spec_model_size: Union[int, str]
) -> bool:
    if isinstance(model_size, str):
        model_size = model_size.replace("_", ".")
    if isinstance(spec_model_size, str):
        spec_model_size = spec_model_size.replace("_", ".")

    if model_size == spec_model_size:
        return True

    try:
        ms = int(model_size)
        ss = int(spec_model_size)
        return ms == ss
    except ValueError:
        return False


def convert_model_size_to_float(
    model_size_in_billions: Union[float, int, str]
) -> float:
    if isinstance(model_size_in_billions, str):
        if "_" in model_size_in_billions:
            ms = model_size_in_billions.replace("_", ".")
            return float(ms)
        elif "." in model_size_in_billions:
            return float(model_size_in_billions)
        else:
            return int(model_size_in_billions)
    return model_size_in_billions


def match_llm(
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
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

    def _apply_format_to_model_id(spec: LLMSpecV1, q: str) -> LLMSpecV1:
        # Different quantized versions of some models use different model ids,
        # Here we check the `{}` in the model id to format the id.
        if spec.model_id and "{" in spec.model_id:
            spec.model_id = spec.model_id.format(quantization=q)
        return spec

    if download_from_modelscope():
        all_families = (
            BUILTIN_MODELSCOPE_LLM_FAMILIES
            + BUILTIN_LLM_FAMILIES
            + user_defined_llm_families
        )
    elif download_from_csghub():
        all_families = (
            BUILTIN_CSGHUB_LLM_FAMILIES
            + BUILTIN_LLM_FAMILIES
            + user_defined_llm_families
        )
    else:
        all_families = BUILTIN_LLM_FAMILIES + user_defined_llm_families

    for family in all_families:
        if model_name != family.model_name:
            continue
        for spec in family.model_specs:
            matched_quantization = _match_quantization(quantization, spec.quantizations)
            if (
                model_format
                and model_format != spec.model_format
                or model_size_in_billions
                and not match_model_size(
                    model_size_in_billions, spec.model_size_in_billions
                )
                or quantization
                and matched_quantization is None
            ):
                continue
            # Copy spec to avoid _apply_format_to_model_id modify the original spec.
            spec = spec.copy()
            if quantization:
                return (
                    family,
                    _apply_format_to_model_id(spec, matched_quantization),
                    matched_quantization,
                )
            else:
                # TODO: If user does not specify quantization, just use the first one
                _q = "none" if spec.model_format == "pytorch" else spec.quantizations[0]
                return family, _apply_format_to_model_id(spec, _q), _q
    return None


def register_llm(llm_family: LLMFamilyV1, persist: bool):
    from ..utils import is_valid_model_name
    from . import generate_engine_config_by_model_family

    if not is_valid_model_name(llm_family.model_name):
        raise ValueError(f"Invalid model name {llm_family.model_name}.")

    with UD_LLM_FAMILIES_LOCK:
        for family in BUILTIN_LLM_FAMILIES + UD_LLM_FAMILIES:
            if llm_family.model_name == family.model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {family.model_name}"
                )

        UD_LLM_FAMILIES.append(llm_family)
        generate_engine_config_by_model_family(llm_family)

    if persist:
        # We only validate model URL when persist is True.
        for spec in llm_family.model_specs:
            model_uri = spec.model_uri
            if model_uri and not is_valid_model_uri(model_uri):
                raise ValueError(f"Invalid model URI {model_uri}.")

        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "llm", f"{llm_family.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(llm_family.json())


def unregister_llm(model_name: str, raise_error: bool = True):
    with UD_LLM_FAMILIES_LOCK:
        llm_family = None
        for i, f in enumerate(UD_LLM_FAMILIES):
            if f.model_name == model_name:
                llm_family = f
                break
        if llm_family:
            UD_LLM_FAMILIES.remove(llm_family)
            del LLM_ENGINES[model_name]

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
            if raise_error:
                raise ValueError(f"Model {model_name} not found")
            else:
                logger.warning(f"Custom model {model_name} not found")


def check_engine_by_spec_parameters(
    model_engine: str,
    model_name: str,
    model_format: str,
    model_size_in_billions: Union[str, int],
    quantization: str,
) -> Type[LLM]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in LLM_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in LLM_ENGINES:
        raise ValueError(f"Model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in LLM_ENGINES[model_name]:
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = LLM_ENGINES[model_name][model_engine]
    for param in match_params:
        if (
            model_name == param["model_name"]
            and model_format == param["model_format"]
            and model_size_in_billions == param["model_size_in_billions"]
            and quantization in param["quantizations"]
        ):
            return param["llm_class"]
    raise ValueError(
        f"Model {model_name} cannot be run on engine {model_engine}, with format {model_format}, size {model_size_in_billions} and quantization {quantization}."
    )
