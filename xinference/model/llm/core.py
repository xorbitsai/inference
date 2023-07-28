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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV1, LLMSpecV1

logger = logging.getLogger(__name__)


class LLM(abc.ABC):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        *args,
        **kwargs,
    ):
        self.model_uid = model_uid
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
