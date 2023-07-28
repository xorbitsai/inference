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
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from xinference.constants import XINFERENCE_CACHE_DIR

logger = logging.getLogger(__name__)


class GgmlLLMSpecV1(BaseModel):
    model_format: Literal["ggmlv3"]
    model_size_in_billions: int
    quantizations: List[str]
    model_id: str
    model_file_name_template: str
    model_local_path: Optional[str]


class PytorchLLMSpecV1(BaseModel):
    model_format: Literal["pytorch"]
    model_size_in_billions: int
    quantizations: List[str]
    model_id: str
    model_local_path: Optional[str]


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

LLM_FAMILIES: List[LLMFamilyV1] = []


def get_legacy_cache_path(
    model_name: str,
    model_format: str,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
) -> str:
    full_name = f"{model_name}-{model_format}-{model_size_in_billions}b-{quantization}"
    save_dir = os.path.join(XINFERENCE_CACHE_DIR, full_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.bin")
    return save_path


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
        logger.debug("Legacy cache path exists: %s", legacy_cache_path)
        return os.path.dirname(legacy_cache_path)
    else:
        return cache_from_huggingface(llm_family, llm_spec, quantization)


def cache_from_huggingface(
    llm_family: LLMFamilyV1,
    llm_spec: "LLMSpecV1",
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Hugging Face. Return the cache directory.
    """
    import huggingface_hub

    cache_dir_name = f"{llm_family.model_name}-{llm_spec.model_format}-{llm_spec.model_size_in_billions}b"
    cache_dir = os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if llm_spec.model_format == "pytorch":
        assert isinstance(llm_spec, PytorchLLMSpecV1)
        huggingface_hub.snapshot_download(
            llm_spec.model_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=True,
        )
    elif llm_spec.model_format == "ggmlv3":
        assert isinstance(llm_spec, GgmlLLMSpecV1)
        file_name = llm_spec.model_file_name_template.format(quantization=quantization)
        huggingface_hub.hf_hub_download(
            llm_spec.model_id,
            filename=file_name,
            local_dir=cache_dir,
            local_dir_use_symlinks=True,
        )

    return cache_dir
