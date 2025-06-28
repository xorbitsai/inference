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
import inspect
import logging
import os
import platform
import warnings
from abc import abstractmethod
from collections import defaultdict
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from ...core.utils import parse_replica_model_uid
from ...types import PeftModelConfig
from ..core import ModelDescription
from .reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV1, LLMSpecV1

logger = logging.getLogger(__name__)


LLM_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)


def get_llm_model_descriptions():
    import copy

    return copy.deepcopy(LLM_MODEL_DESCRIPTIONS)


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
        self.model_uid, self.rep_id = parse_replica_model_uid(replica_model_uid)
        self.raw_model_uid = replica_model_uid
        self.model_family = model_family
        self.model_spec = model_spec
        self.quantization = quantization
        self.model_path = model_path
        self.reasoning_parser = None
        if args:
            raise ValueError(f"Unrecognized positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    @classmethod
    @abstractmethod
    def check_lib(cls) -> bool:
        raise NotImplementedError

    @staticmethod
    def _is_darwin_and_apple_silicon():
        return platform.system() == "Darwin" and platform.processor() == "arm"

    @staticmethod
    def _is_linux():
        return platform.system() == "Linux"

    @staticmethod
    @lru_cache
    def _has_cuda_device():
        """
        Use pynvml to impl this interface.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

        device_count = 0
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
        except:
            pass
        finally:
            try:
                nvmlShutdown()
            except:
                pass

        return device_count > 0

    @staticmethod
    @lru_cache
    def _get_cuda_count():
        from ...device_utils import get_available_device_env_name
        from ...utils import cuda_count

        env_name = get_available_device_env_name()
        if env_name is None:
            return cuda_count()

        cuda_visible_devices = os.getenv(env_name, None)
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
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not cls.check_lib():
            return False
        return cls.match_json(llm_family, llm_spec, quantization)

    @classmethod
    @abstractmethod
    def match_json(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        raise NotImplementedError

    def prepare_parse_reasoning_content(
        self, reasoning_content: bool, enable_thinking: bool = True
    ):
        if "hybrid" not in self.model_family.model_ability and not enable_thinking:
            enable_thinking = True
            warnings.warn(
                "enable_thinking cannot be disabled for non hybrid model, will be ignored"
            )
        # Initialize reasoning parser if model has reasoning ability
        self.reasoning_parser = ReasoningParser(  # type: ignore
            reasoning_content,
            self.model_family.reasoning_start_tag,  # type: ignore
            self.model_family.reasoning_end_tag,  # type: ignore
            enable_thinking=enable_thinking,
        )


# Context variable for passing per-request chat context (e.g., chat_template_kwargs).
# This variable should be set at the beginning of each chat or stream_chat call.
# It allows downstream components (e.g., reasoning_parser) to access request-specific
# settings like 'enable_thinking', without requiring those values to be passed explicitly
# through every function layer.
#
# The context is automatically isolated per thread or coroutine, so concurrent requests
# will not interfere with each other.
chat_context_var: ContextVar[dict] = ContextVar("chat_context_var", default={})


class LLMDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        llm_family: "LLMFamilyV1",
        llm_spec: "LLMSpecV1",
        quantization: Optional[str],
        multimodal_projector: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._llm_family = llm_family
        self._llm_spec = llm_spec
        self._quantization = quantization
        self._multimodal_projector = multimodal_projector

    @property
    def spec(self):
        return self._llm_family

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
            "model_family": self._llm_family.model_family
            or self._llm_family.model_name,
            "quantization": self._quantization,
            "multimodal_projector": self._multimodal_projector,
            "model_hub": self._llm_spec.model_hub,
            "revision": self._llm_spec.model_revision,
            "context_length": self._llm_family.context_length,
        }

    def to_version_info(self):
        from .utils import get_file_location, get_model_version

        model_file_location, cache_status = get_file_location(
            self._llm_family, self._llm_spec, self._quantization
        )

        return {
            "model_version": get_model_version(
                self._llm_family, self._llm_spec, self._quantization
            ),
            "model_file_location": model_file_location,
            "cache_status": cache_status,
            "quantization": self._quantization,
            "multimodal_projector": self._multimodal_projector,
            "model_format": self._llm_spec.model_format,
            "model_size_in_billions": self._llm_spec.model_size_in_billions,
        }


def generate_llm_description(llm_family: "LLMFamilyV1") -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    for spec in llm_family.model_specs:
        multimodal_projectors = getattr(spec, "multimodal_projectors", None)
        for q in spec.quantizations:
            if multimodal_projectors:
                for mmproj in multimodal_projectors:
                    res[llm_family.model_name].append(
                        LLMDescription(
                            None, None, llm_family, spec, q, mmproj
                        ).to_version_info()
                    )
            else:
                res[llm_family.model_name].append(
                    LLMDescription(None, None, llm_family, spec, q).to_version_info()
                )
    return res


def create_llm_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
    peft_model_config: Optional[PeftModelConfig] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[LLM, LLMDescription]:
    from .llm_family import cache, check_engine_by_spec_parameters, match_llm

    if model_engine is None:
        raise ValueError("model_engine is required for LLM model")
    match_result = match_llm(
        model_name, model_format, model_size_in_billions, quantization, download_hub
    )

    if not match_result:
        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format},"
            f" size: {model_size_in_billions}, quantization: {quantization}"
        )
    llm_family, llm_spec, quantization = match_result
    assert quantization is not None

    llm_cls = check_engine_by_spec_parameters(
        model_engine,
        llm_family.model_name,
        llm_spec.model_format,
        llm_spec.model_size_in_billions,
        quantization,
    )
    logger.debug(f"Launching {model_uid} with {llm_cls.__name__}")

    multimodal_projector = kwargs.get("multimodal_projector")
    if not model_path:
        model_path = cache(llm_family, llm_spec, quantization, multimodal_projector)

    peft_model = peft_model_config.peft_model if peft_model_config else None
    if peft_model is not None:
        if "peft_model" in inspect.signature(llm_cls.__init__).parameters:
            model = llm_cls(
                model_uid,
                llm_family,
                llm_spec,
                quantization,
                model_path,
                kwargs,
                peft_model,
            )
        else:
            logger.warning(
                f"Model not supported with lora, name: {model_name}, format: {model_format}, engine: {model_engine}. "
                f"Load this without lora."
            )
            model = llm_cls(
                model_uid, llm_family, llm_spec, quantization, model_path, kwargs
            )
    else:
        model = llm_cls(
            model_uid, llm_family, llm_spec, quantization, model_path, kwargs
        )
    return model, LLMDescription(
        subpool_addr, devices, llm_family, llm_spec, quantization, multimodal_projector
    )
