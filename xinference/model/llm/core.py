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
from .reasoning_parser import ReasoningParser
from .tool_parsers import TOOL_PARSERS

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV2, LLMSpecV1

logger = logging.getLogger(__name__)


LLM_VERSION_INFOS: Dict[str, List[Dict]] = defaultdict(list)


def get_llm_version_infos():
    import copy

    return copy.deepcopy(LLM_VERSION_INFOS)


class LLM(abc.ABC):
    allow_batch = False

    def __init__(
        self,
        replica_model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        *args,
        **kwargs,
    ):
        self.model_uid, self.rep_id = parse_replica_model_uid(replica_model_uid)
        self.raw_model_uid = replica_model_uid
        self.model_family = model_family
        self.model_spec = model_family.model_specs[0]
        self.quantization = model_family.model_specs[0].quantization
        self.model_path = model_path
        self.reasoning_parser = None
        self.tool_parser = None
        if args:
            raise ValueError(f"Unrecognized positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    @classmethod
    @abstractmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
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
    def _has_mlu_device():
        """
        Use cnmon command to detect MLU devices.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        try:
            import subprocess

            result = subprocess.run(
                ["cnmon", "info"], capture_output=True, text=True, timeout=5
            )
            return "Card 0" in result.stdout
        except:
            return False

    @staticmethod
    @lru_cache
    def _has_vacc_device():
        """
        Use glob command to detect VACC devices.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        try:
            import glob

            return len(glob.glob("/dev/vacc*")) > 0
        except:
            return False

    @staticmethod
    @lru_cache
    def _has_musa_device():
        """
        Use pymtml to impl this interface.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        try:
            from pymtml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown
        except Exception:
            return False

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
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        lib_result = cls.check_lib()
        if lib_result != True:
            return False
        match_result = cls.match_json(llm_family, llm_spec, quantization)
        return match_result == True

    @classmethod
    @abstractmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
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

    def prepare_parse_tool_calls(self):
        if self.model_family.tool_parser is None:
            return
        if self.model_family.tool_parser not in TOOL_PARSERS:
            return
        tool_parser = TOOL_PARSERS[self.model_family.tool_parser]
        self.tool_parser = tool_parser()


# Context variable for passing per-request chat context (e.g., chat_template_kwargs).
# This variable should be set at the beginning of each chat or stream_chat call.
# It allows downstream components (e.g., reasoning_parser) to access request-specific
# settings like 'enable_thinking', without requiring those values to be passed explicitly
# through every function layer.
#
# The context is automatically isolated per thread or coroutine, so concurrent requests
# will not interfere with each other.
chat_context_var: ContextVar[dict] = ContextVar("chat_context_var", default={})


def generate_llm_version_info(llm_family: "LLMFamilyV2") -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    # Use model_specs from huggingface, as HuggingFace is the most comprehensive.
    hf_specs = [
        spec for spec in llm_family.model_specs if spec.model_hub == "huggingface"
    ]
    for spec in hf_specs:
        _llm_family = llm_family.copy()
        _llm_family.model_specs = [spec]
        multimodal_projectors = getattr(spec, "multimodal_projectors", None)
        if multimodal_projectors:
            for mmproj in multimodal_projectors:
                _llm_family.multimodal_projector = mmproj
                res[_llm_family.model_name].append(_llm_family.to_version_info())
        else:
            res[_llm_family.model_name].append(_llm_family.to_version_info())
    return res


def create_llm_model_instance(
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
) -> LLM:
    from .cache_manager import LLMCacheManager
    from .llm_family import (
        check_engine_by_spec_parameters,
        check_engine_by_spec_parameters_with_virtual_env,
        match_llm,
    )

    if model_engine is None:
        raise ValueError("model_engine is required for LLM model")
    llm_family = match_llm(
        model_name, model_format, model_size_in_billions, quantization, download_hub
    )

    if not llm_family:
        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format}, "
            f"size: {model_size_in_billions}, quantization: {quantization}"
        )

    enable_virtual_env = kwargs.pop("enable_virtual_env", None)
    if enable_virtual_env is None:
        from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

        enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV
    if enable_virtual_env:
        llm_cls = check_engine_by_spec_parameters_with_virtual_env(
            model_engine,
            llm_family.model_name,
            llm_family.model_specs[0].model_format,
            llm_family.model_specs[0].model_size_in_billions,
            llm_family.model_specs[0].quantization,
            llm_family=llm_family,
        )
    else:
        llm_cls = check_engine_by_spec_parameters(
            model_engine,
            llm_family.model_name,
            llm_family.model_specs[0].model_format,
            llm_family.model_specs[0].model_size_in_billions,
            llm_family.model_specs[0].quantization,
        )
    logger.debug(f"Launching {model_uid} with {llm_cls.__name__}")

    multimodal_projector = kwargs.get("multimodal_projector")
    if not model_path:
        cache_manager = LLMCacheManager(llm_family, multimodal_projector)
        model_path = cache_manager.cache()

    peft_model = peft_model_config.peft_model if peft_model_config else None
    if peft_model is not None:
        if "peft_model" in inspect.signature(llm_cls.__init__).parameters:
            model = llm_cls(
                model_uid,
                llm_family,
                model_path,
                kwargs,
                peft_model,
            )
        else:
            logger.warning(
                f"Model not supported with lora, name: {model_name}, format: {model_format}, engine: {model_engine}. "
                f"Load this without lora."
            )
            model = llm_cls(model_uid, llm_family, model_path, kwargs)
    else:
        model = llm_cls(model_uid, llm_family, model_path, kwargs)
    return model
