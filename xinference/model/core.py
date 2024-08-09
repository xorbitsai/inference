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

from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Tuple, Union

from .._compat import BaseModel
from ..types import PeftModelConfig


class ModelDescription(ABC):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_path: Optional[str] = None,
    ):
        self.address = address
        self.devices = devices
        self._model_path = model_path

    def to_dict(self):
        """
        Return a dict to describe some information about model.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def to_version_info(self):
        """
        Return a dict to describe version info about a model instance
        """


def create_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_type: str,
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
    peft_model_config: Optional[PeftModelConfig] = None,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[Any, ModelDescription]:
    from .audio.core import create_audio_model_instance
    from .embedding.core import create_embedding_model_instance
    from .flexible.core import create_flexible_model_instance
    from .image.core import create_image_model_instance
    from .llm.core import create_llm_model_instance
    from .rerank.core import create_rerank_model_instance
    from .video.core import create_video_model_instance

    if model_type == "LLM":
        return create_llm_model_instance(
            subpool_addr,
            devices,
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
        # embedding model doesn't accept trust_remote_code
        kwargs.pop("trust_remote_code", None)
        return create_embedding_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "image":
        kwargs.pop("trust_remote_code", None)
        return create_image_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            peft_model_config,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "rerank":
        kwargs.pop("trust_remote_code", None)
        return create_rerank_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "audio":
        kwargs.pop("trust_remote_code", None)
        return create_audio_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "video":
        kwargs.pop("trust_remote_code", None)
        return create_video_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            download_hub,
            model_path,
            **kwargs,
        )
    elif model_type == "flexible":
        kwargs.pop("trust_remote_code", None)
        return create_flexible_model_instance(
            subpool_addr, devices, model_uid, model_name, model_path, **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")


class CacheableModelSpec(BaseModel):
    model_name: str
    model_id: str
    model_revision: Optional[str]
    model_hub: str = "huggingface"
