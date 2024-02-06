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
from typing import Any, List, Optional, Tuple

from .._compat import BaseModel


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
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
    is_local_deployment: bool = False,
    **kwargs,
) -> Tuple[Any, ModelDescription]:
    from .audio.core import create_audio_model_instance
    from .embedding.core import create_embedding_model_instance
    from .image.core import create_image_model_instance
    from .llm.core import create_llm_model_instance
    from .rerank.core import create_rerank_model_instance

    if model_type == "LLM":
        return create_llm_model_instance(
            subpool_addr,
            devices,
            model_uid,
            model_name,
            model_format,
            model_size_in_billions,
            quantization,
            is_local_deployment,
            **kwargs,
        )
    elif model_type == "embedding":
        # embedding model doesn't accept trust_remote_code
        kwargs.pop("trust_remote_code", None)
        return create_embedding_model_instance(
            subpool_addr, devices, model_uid, model_name, **kwargs
        )
    elif model_type == "image":
        kwargs.pop("trust_remote_code", None)
        return create_image_model_instance(
            subpool_addr, devices, model_uid, model_name, **kwargs
        )
    elif model_type == "rerank":
        kwargs.pop("trust_remote_code", None)
        return create_rerank_model_instance(
            subpool_addr, devices, model_uid, model_name, **kwargs
        )
    elif model_type == "audio":
        kwargs.pop("trust_remote_code", None)
        return create_audio_model_instance(
            subpool_addr, devices, model_uid, model_name, **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")


class CacheableModelSpec(BaseModel):
    model_name: str
    model_id: str
    model_revision: Optional[str]
    model_hub: str = "huggingface"
