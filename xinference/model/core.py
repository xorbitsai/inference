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
from typing import Any, List, Literal, Optional, Union

from .._compat import BaseModel
from ..types import PeftModelConfig


def create_model_instance(
    model_uid: str,
    model_type: str,
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
) -> Any:
    from .audio.core import create_audio_model_instance
    from .embedding.core import create_embedding_model_instance
    from .flexible.core import create_flexible_model_instance
    from .image.core import create_image_model_instance
    from .llm.core import create_llm_model_instance
    from .rerank.core import create_rerank_model_instance
    from .video.core import create_video_model_instance

    # enable_thinking is only meaningful for LLMs; drop it for other model types.
    if model_type != "LLM":
        kwargs.pop("enable_thinking", None)

    if model_type == "LLM":
        return create_llm_model_instance(
            model_uid,
            model_name,
            model_engine,
            model_format,
            model_size_in_billions,
            quantization,
            peft_model_config,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "embedding":
        # allow trust_remote_code for engines that require it (e.g. vLLM)
        if model_engine and model_engine.lower() != "vllm":
            kwargs.pop("trust_remote_code", None)
        return create_embedding_model_instance(
            model_uid,
            model_name,
            model_engine,
            model_format,
            quantization,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "image":
        kwargs.pop("trust_remote_code", None)
        return create_image_model_instance(
            model_uid,
            model_name,
            peft_model_config,
            download_hub,
            model_path,
            model_engine,
            model_format,
            quantization,
            **kwargs,
        )
    elif model_type == "rerank":
        kwargs.pop("trust_remote_code", None)
        return create_rerank_model_instance(
            model_uid,
            model_name,
            model_engine,
            model_format,
            quantization,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "audio":
        kwargs.pop("trust_remote_code", None)
        return create_audio_model_instance(
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "video":
        kwargs.pop("trust_remote_code", None)
        return create_video_model_instance(
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "flexible":
        kwargs.pop("trust_remote_code", None)
        return create_flexible_model_instance(
            model_uid, model_name, model_path, **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")


class CacheableModelSpec(BaseModel):
    model_name: str
    model_id: str
    model_revision: Optional[str]
    model_hub: str = "huggingface"
    cache_config: Optional[dict]


class VirtualEnvSettings(BaseModel):
    packages: List[str]
    inherit_pip_config: bool = True
    index_url: Optional[str] = None
    extra_index_url: Optional[Union[str, List[str]]] = None
    find_links: Optional[Union[str, List[str]]] = None
    trusted_host: Optional[Union[str, List[str]]] = None
    index_strategy: Optional[str] = None
    no_build_isolation: Optional[bool] = None
