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
import platform
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

from ...core.utils import parse_replica_model_uid
from ..core import ModelDescription

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV1, LLMSpecV1

logger = logging.getLogger(__name__)


class LLM(abc.ABC):
    def __init__(
        self,
        replica_model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
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
    def _is_darwin_and_apple_silicon():
        return platform.system() == "Darwin" and platform.processor() == "arm"

    @staticmethod
    def _is_linux():
        return platform.system() == "Linux"

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        raise NotImplementedError


class LLMModelDescription(ModelDescription):
    def __init__(
        self, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ):
        self._llm_family = llm_family
        self._llm_spec = llm_spec
        self._quantization = quantization

    def to_dict(self):
        return {
            "model_type": "LLM",
            "model_name": self._llm_family.model_name,
            "model_lang": self._llm_family.model_lang,
            "model_ability": self._llm_family.model_ability,
            "model_description": self._llm_family.model_description,
            "model_format": self._llm_spec.model_format,
            "model_size_in_billions": self._llm_spec.model_size_in_billions,
            "quantization": self._quantization,
            "revision": self._llm_spec.model_revision,
            "context_length": self._llm_family.context_length,
        }


def create_llm_model_instance(
    model_uid: str,
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
    is_local_deployment: bool = False,
    **kwargs,
) -> Tuple[LLM, LLMModelDescription]:
    from . import match_llm, match_llm_cls
    from .llm_family import cache

    match_result = match_llm(
        model_name,
        model_format,
        model_size_in_billions,
        quantization,
        is_local_deployment,
    )
    if not match_result:
        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format},"
            f" size: {model_size_in_billions}, quantization: {quantization}"
        )
    llm_family, llm_spec, quantization = match_result

    assert quantization is not None
    save_path = cache(llm_family, llm_spec, quantization)

    llm_cls = match_llm_cls(llm_family, llm_spec)
    if not llm_cls:
        raise ValueError(
            f"Model not supported, name: {model_name}, format: {model_format},"
            f" size: {model_size_in_billions}, quantization: {quantization}"
        )
    logger.debug(f"Launching {model_uid} with {llm_cls.__name__}")

    model = llm_cls(model_uid, llm_family, llm_spec, quantization, save_path, kwargs)
    return model, LLMModelDescription(llm_family, llm_spec, quantization)
