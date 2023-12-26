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
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, validator

from ...core.utils import parse_replica_model_uid
from ..core import ModelDescription
from ..utils import download_from_modelscope

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LENGTH = 2048


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
        llm_family: "LVLMFamilyV1",
        llm_spec: "LVLMSpecV1",
        quantization: Optional[str],
    ):
        super().__init__(address, devices)
        self._llm_family = llm_family
        self._llm_spec = llm_spec
        self._quantization = quantization

    def to_dict(self):
        return {
            "model_type": "LLM",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._llm_family.model_name,
            "model_lang": self._llm_family.model_lang,
            "model_ability": self._llm_family.model_ability,
            "model_description": self._llm_family.model_description,
            "model_format": self._llm_spec.model_format,
            "model_size_in_billions": self._llm_spec.model_size_in_billions,
            "quantization": self._quantization,
            "model_hub": self._llm_spec.model_hub,
            "revision": self._llm_spec.model_revision,
            "context_length": self._llm_family.context_length,
        }


class LVLM(abc.ABC):
    def __init__(
        self,
        replica_model_uid: str,
        model_family: "LVLMFamilyV1",
        model_spec: "LVLMSpecV1",
        quantization: str,
        model_path: str,
        *args,
        **kwargs,
    ):
        self.model_uid, self.replica, self.rep_id = parse_replica_model_uid(
            replica_model_uid
        )
        self.model_family = model_family
        self.model_spec = model_spec
        self.quantization = quantization
        self.model_path = model_path
        if args:
            raise ValueError(f"Unrecognized positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    @staticmethod
    def handle_model_size(model_size_in_billions: Union[str, int]) -> Union[int, float]:
        if isinstance(model_size_in_billions, str):
            if "_" in model_size_in_billions:
                ms = model_size_in_billions.replace("_", ".")
                return float(ms)
            else:
                raise ValueError("Invalid format for `model_size_in_billions`")
        return model_size_in_billions

    @staticmethod
    def _is_darwin_and_apple_silicon():
        return platform.system() == "Darwin" and platform.processor() == "arm"

    @staticmethod
    def _is_linux():
        return platform.system() == "Linux"

    @staticmethod
    def _has_cuda_device():
        from ...utils import cuda_count

        return cuda_count() > 0

    @staticmethod
    def _get_cuda_count():
        from ...utils import cuda_count

        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            return cuda_count()

        if cuda_visible_devices == "-1":
            return 0
        else:
            return len(cuda_visible_devices.split(","))

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @classmethod
    def match(
        cls, llm_family: "LVLMFamilyV1", llm_spec: "LVLMSpecV1", quantization: str
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
        all_families = BUILTIN_MODELSCOPE_LVLM_FAMILIES
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
    from ..llm.llm_family import cache

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

    logger.debug(f"Launching {model_uid} with {LVLM.__name__}")

    model = LVLM(model_uid, model_family, model_spec, quantization, save_path, kwargs)
    return model, LVLMDescription(
        subpool_addr, devices, model_family, model_spec, quantization
    )
