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

import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from ..._compat import BaseModel, Field
from ...types import Rerank
from ..core import VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin
from .rerank_family import (
    check_engine_by_model_name_and_engine,
    check_engine_by_model_name_and_engine_with_virtual_env,
    match_rerank,
)

logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
RERANK_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
RERANK_EMPTY_CACHE_COUNT = int(os.getenv("XINFERENCE_RERANK_EMPTY_CACHE_COUNT", "10"))
assert RERANK_EMPTY_CACHE_COUNT > 0


def get_rerank_model_descriptions():
    import copy

    return copy.deepcopy(RERANK_MODEL_DESCRIPTIONS)


class TransformersRerankSpecV1(BaseModel):
    model_format: Literal["pytorch"]
    model_hub: str = "huggingface"
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    model_uri: Optional[str] = None
    quantization: str = "none"


class LlamaCppRerankSpecV1(BaseModel):
    model_format: Literal["ggufv2"]
    model_hub: str = "huggingface"
    model_id: Optional[str]
    model_uri: Optional[str]
    model_revision: Optional[str]
    quantization: str
    model_file_name_template: str
    model_file_name_split_template: Optional[str]
    quantization_parts: Optional[Dict[str, List[str]]]


RerankSpecV1 = Annotated[
    Union[TransformersRerankSpecV1, LlamaCppRerankSpecV1],
    Field(discriminator="model_format"),
]


class RerankModelFamilyV2(BaseModel, ModelInstanceInfoMixin):
    version: Literal[2]
    model_name: str
    model_specs: List[RerankSpecV1]
    language: List[str]
    type: Optional[str] = "unknown"
    max_tokens: Optional[int]
    cache_config: Optional[dict] = None
    virtualenv: Optional[VirtualEnvSettings]

    class Config:
        extra = "allow"

    def to_description(self):
        spec = self.model_specs[0]
        return {
            "model_type": "rerank",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "type": self.type,
            "model_name": self.model_name,
            "language": self.language,
            "model_revision": spec.model_revision,
        }

    def to_version_info(self):
        from .cache_manager import RerankCacheManager

        cache_manager = RerankCacheManager(self)
        return {
            "model_version": self.model_name,
            "model_file_location": cache_manager.get_cache_dir(),
            "cache_status": cache_manager.get_cache_status(),
            "language": self.language,
        }


def generate_rerank_description(
    model_spec: RerankModelFamilyV2,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[model_spec.model_name].append(model_spec.to_version_info())
    return res


class RerankModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_family: RerankModelFamilyV2,
        quantization: Optional[str],
        *,
        device: Optional[str] = None,
        use_fp16: bool = False,
        **kwargs,
    ):
        self.model_family = model_family
        self._model_spec = model_family.model_specs[0]
        self._model_uid = model_uid
        self._model_path = model_path
        self._quantization = quantization
        self._device = device
        self._use_fp16 = use_fp16
        self._model = None
        self._counter = 0
        self._kwargs = kwargs
        if model_family.type == "unknown":
            model_family.type = self._auto_detect_type(model_path)

    @classmethod
    @abstractmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        pass

    @classmethod
    @abstractmethod
    def match_json(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        pass

    @classmethod
    def match(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ):
        """
        Return if the model_spec can be matched.
        """
        lib_result = cls.check_lib()
        if lib_result != True:
            return False
        match_result = cls.match_json(model_family, model_spec, quantization)
        return match_result == True

    @staticmethod
    def _get_tokenizer(model_path):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer

    @staticmethod
    def _auto_detect_type(model_path):
        """This method may not be stable due to the fact that the tokenizer name may be changed.
        Therefore, we only use this method for unknown model types."""

        type_mapper = {
            "LlamaTokenizerFast": "LLM-based layerwise",
            "GemmaTokenizerFast": "LLM-based",
            "XLMRobertaTokenizerFast": "normal",
        }

        tokenizer = RerankModel._get_tokenizer(model_path)
        rerank_type = type_mapper.get(type(tokenizer).__name__)
        if rerank_type is None:
            logger.warning(
                f"Can't determine the rerank type based on the tokenizer {tokenizer}, use normal type by default."
            )
            return "normal"
        return rerank_type

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ) -> Rerank: ...


def create_rerank_model_instance(
    model_uid: str,
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> RerankModel:
    from .cache_manager import RerankCacheManager

    enable_virtual_env = kwargs.pop("enable_virtual_env", None)
    model_family = match_rerank(model_name, model_format, quantization, download_hub)
    if model_path is None:
        cache_manager = RerankCacheManager(model_family)
        model_path = cache_manager.cache()

    if model_engine is None:
        # unlike LLM and for compatibility,
        # we use sentence_transformers as the default engine for all models
        model_engine = "sentence_transformers"

    if enable_virtual_env is None:
        from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

        enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV

    if enable_virtual_env:
        rerank_cls = check_engine_by_model_name_and_engine_with_virtual_env(
            model_engine,
            model_name,
            model_format,
            quantization,
            model_family=model_family,
        )
    else:
        rerank_cls = check_engine_by_model_name_and_engine(
            model_engine,
            model_name,
            model_format,
            quantization,
        )
    model = rerank_cls(
        model_uid,
        model_path,
        model_family,
        quantization,
        **kwargs,
    )
    return model
