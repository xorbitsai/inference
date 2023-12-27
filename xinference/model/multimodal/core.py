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

import abc
import logging
import os
import platform
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, validator

from ...constants import XINFERENCE_CACHE_DIR
from ...core.utils import parse_replica_model_uid
from ...types import ChatCompletion, ChatCompletionChunk
from ..core import ModelDescription
from ..utils import (
    download_from_modelscope,
    is_model_cached,
    retry_download,
    symlink_local_file,
    valid_model_revision,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LENGTH = 2048
# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)


class LVLMSpecV1(BaseModel):
    model_format: Literal["pytorch", "gptq"]
    # Must in order that `str` first, then `int`
    model_size_in_billions: Union[str, int]
    quantizations: List[str]
    model_id: str
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


class LVLMPromptStyleV1(BaseModel):
    style_name: str
    system_prompt: str = ""
    roles: List[str]
    image_formatter: str = ""
    text_formatter: str = ""
    sep: str = ""


class LVLMFamilyV1(BaseModel):
    version: Literal[1]
    context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH
    model_name: str
    model_lang: List[str]
    model_ability: List[Literal["chat"]]
    model_description: Optional[str]
    model_specs: List["LVLMSpecV1"]
    prompt_style: Optional["LVLMPromptStyleV1"]


class LVLMDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_family: "LVLMFamilyV1",
        model_spec: "LVLMSpecV1",
        quantization: Optional[str],
    ):
        super().__init__(address, devices)
        self._model_family = model_family
        self._model_spec = model_spec
        self._quantization = quantization

    def to_dict(self):
        return {
            "model_type": "LVLM",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_family.model_name,
            "model_lang": self._model_family.model_lang,
            "model_ability": self._model_family.model_ability,
            "model_description": self._model_family.model_description,
            "model_format": self._model_spec.model_format,
            "model_size_in_billions": self._model_spec.model_size_in_billions,
            "quantization": self._quantization,
            "model_hub": self._model_spec.model_hub,
            "revision": self._model_spec.model_revision,
            "context_length": self._model_family.context_length,
        }


class LVLM(abc.ABC):
    def __init__(
        self,
        replica_model_uid: str,
        model_family: "LVLMFamilyV1",
        model_spec: "LVLMSpecV1",
        quantization: str,
        model_path: str,
        kwargs: Dict,
    ):
        self.model_uid, self.replica, self.rep_id = parse_replica_model_uid(
            replica_model_uid
        )
        self.model_family = model_family
        self.model_spec = model_spec
        self.quantization = quantization
        self.model_path = model_path
        self.kwargs = kwargs
        logger.info("Init model %s with kwargs: %s", self.model_uid, kwargs)

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        raise NotImplementedError

    @classmethod
    def match(
        cls, model_family: "LVLMFamilyV1", model_spec: "LVLMSpecV1", quantization: str
    ) -> bool:
        raise NotImplementedError


BUILTIN_LVLM_FAMILIES: List["LVLMFamilyV1"] = []
BUILTIN_MODELSCOPE_LVLM_FAMILIES: List["LVLMFamilyV1"] = []


def match_multimodal(
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
) -> Optional[Tuple[LVLMFamilyV1, LVLMSpecV1, str]]:
    """
    Find an LLM family, spec, and quantization that satisfy given criteria.
    """

    def _match_quantization(q: Union[str, None], quantizations: List[str]):
        # Currently, the quantization name could include both uppercase and lowercase letters,
        # so it is necessary to ensure that the case sensitivity does not
        # affect the matching results.
        if q is None:
            return q
        for quant in quantizations:
            if q.lower() == quant.lower():
                return quant

    def _apply_format_to_model_id(spec: LVLMSpecV1, q: str) -> LVLMSpecV1:
        # Different quantized versions of some models use different model ids,
        # Here we check the `{}` in the model id to format the id.
        if "{" in spec.model_id:
            spec.model_id = spec.model_id.format(quantization=q)
        return spec

    if download_from_modelscope():
        all_families = BUILTIN_MODELSCOPE_LVLM_FAMILIES + BUILTIN_LVLM_FAMILIES
    else:
        all_families = BUILTIN_LVLM_FAMILIES

    for family in all_families:
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
                return (
                    family,
                    _apply_format_to_model_id(spec, matched_quantization),
                    matched_quantization,
                )
            else:
                return family, _apply_format_to_model_id(spec, "none"), "none"
    return None


def create_multimodal_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
    **kwargs,
) -> Tuple[LVLM, LVLMDescription]:
    match_result = match_multimodal(
        model_name,
        model_format,
        model_size_in_billions,
        quantization,
    )
    if not match_result:
        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format},"
            f" size: {model_size_in_billions}, quantization: {quantization}"
        )
    model_family, model_spec, quantization = match_result

    assert quantization is not None
    save_path = cache(model_family, model_spec, quantization)

    cls = match_cls(model_family, model_spec, quantization)
    logger.debug(f"Launching {model_uid} with {cls.__name__}")

    model = cls(model_uid, model_family, model_spec, quantization, save_path, kwargs)
    return model, LVLMDescription(
        subpool_addr, devices, model_family, model_spec, quantization
    )


MODEL_CLASSES: List[Type[LVLM]] = []


def match_cls(
    model_family: LVLMFamilyV1, model_spec: "LVLMSpecV1", quantization: str
) -> Type[LVLM]:
    """
    Find an LLM implementation for given LLM family and spec.
    """
    for cls in MODEL_CLASSES:
        if cls.match(model_family, model_spec, quantization):
            return cls
    raise Exception(f"Model {model_family.model_name} is not supported")


def _get_cache_dir(
    model_family: LVLMFamilyV1,
    model_spec: "LVLMSpecV1",
    create_if_not_exist=True,
):
    cache_dir_name = (
        f"{model_family.model_name}-{model_spec.model_format}"
        f"-{model_spec.model_size_in_billions}b"
    )
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
    elif model_format in ["ggmlv3", "ggufv2", "gptq"]:
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
    else:
        raise ValueError(f"Unsupported format: {model_format}")


def _generate_meta_file(
    meta_path: str,
    model_family: "LVLMFamilyV1",
    model_spec: "LVLMSpecV1",
    quantization: Optional[str] = None,
):
    assert not valid_model_revision(
        meta_path, model_spec.model_revision
    ), f"meta file {meta_path} should not be valid"
    with open(meta_path, "w") as f:
        import json

        desc = LVLMDescription(None, None, model_family, model_spec, quantization)
        json.dump(desc.to_dict(), f)


def cache_from_modelscope(
    model_family: LVLMFamilyV1,
    model_spec: "LVLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Modelscope. Return the cache directory.
    """
    from modelscope.hub.snapshot_download import snapshot_download

    cache_dir = _get_cache_dir(model_family, model_spec)
    if _skip_download(
        cache_dir,
        model_spec.model_format,
        model_spec.model_hub,
        model_spec.model_revision,
        quantization,
    ):
        return cache_dir

    if model_spec.model_format in ["pytorch", "gptq"]:
        download_dir = retry_download(
            snapshot_download,
            model_family.model_name,
            {
                "model_size": model_spec.model_size_in_billions,
                "model_format": model_spec.model_format,
            },
            model_spec.model_id,
            revision=model_spec.model_revision,
        )
        for subdir, dirs, files in os.walk(download_dir):
            for file in files:
                relpath = os.path.relpath(os.path.join(subdir, file), download_dir)
                symlink_local_file(os.path.join(subdir, file), cache_dir, relpath)
    else:
        raise ValueError(f"Unsupported format: {model_spec.model_format}")

    meta_path = _get_meta_path(
        cache_dir, model_spec.model_format, model_spec.model_hub, quantization
    )
    _generate_meta_file(meta_path, model_family, model_spec, quantization)

    return cache_dir


def cache_from_huggingface(
    model_family: LVLMFamilyV1,
    model_spec: "LVLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Hugging Face. Return the cache directory.
    """
    import huggingface_hub

    cache_dir = _get_cache_dir(model_family, model_spec)
    if _skip_download(
        cache_dir,
        model_spec.model_format,
        model_spec.model_hub,
        model_spec.model_revision,
        quantization,
    ):
        return cache_dir

    if model_spec.model_format in ["pytorch"]:
        assert isinstance(model_spec, LVLMSpecV1)
        retry_download(
            huggingface_hub.snapshot_download,
            model_family.model_name,
            {
                "model_size": model_spec.model_size_in_billions,
                "model_format": model_spec.model_format,
            },
            model_spec.model_id,
            revision=model_spec.model_revision,
            local_dir=cache_dir,
            local_dir_use_symlinks=True,
        )
    else:
        raise ValueError(f"Unsupported model format: {model_spec.model_format}")

    meta_path = _get_meta_path(
        cache_dir, model_spec.model_format, model_spec.model_hub, quantization
    )
    _generate_meta_file(meta_path, model_family, model_spec, quantization)

    return cache_dir


def cache(
    llm_family: LVLMFamilyV1,
    llm_spec: "LVLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    if llm_spec.model_hub == "huggingface":
        logger.info(f"Caching from Hugging Face: {llm_spec.model_id}")
        return cache_from_huggingface(llm_family, llm_spec, quantization)
    elif llm_spec.model_hub == "modelscope":
        logger.info(f"Caching from Modelscope: {llm_spec.model_id}")
        return cache_from_modelscope(llm_family, llm_spec, quantization)
    else:
        raise ValueError(f"Unknown model hub: {llm_spec.model_hub}")


def get_cache_status(
    model_spec: LVLMSpecV1,
) -> bool:
    return is_model_cached(model_spec, MODEL_NAME_TO_REVISION)
