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

from abc import ABC
from typing import Any, Optional, Tuple


class ModelDescription(ABC):
    def to_dict(self):
        """
        Return a dict to describe some information about model.
        :return:
        """


def create_model_instance(
    model_uid: str,
    model_type: str,
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[int] = None,
    quantization: Optional[str] = None,
    is_local_deployment: bool = False,
    **kwargs,
) -> Tuple[Any, ModelDescription]:
    from .embedding.core import create_embedding_model_instance
    from .llm.core import create_llm_model_instance

    if model_type == "LLM":
        return create_llm_model_instance(
            model_uid,
            model_name,
            model_format,
            model_size_in_billions,
            quantization,
            is_local_deployment,
            **kwargs,
        )
    elif model_type == "embedding":
        return create_embedding_model_instance(model_uid, model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")
