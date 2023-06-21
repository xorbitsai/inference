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

from typing import Optional

from ..config import VICNA_CONFIG
from .core import LlamaCppChatModel, LlamaCppModelConfig, register_model


@register_model(VICNA_CONFIG)
class VicunaUncensoredGgml(LlamaCppChatModel):
    name = "Vicuna"
    _system_prompt = (
        "A chat between a curious human and an artificial intelligence assistant."
        " The assistant gives helpful, detailed, and polite answers to the human's"
        " questions. "
    )
    _sep = "###"
    _user_name = "User"
    _assistant_name = "Assistant"

    def __init__(
        self,
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(
            model_path,
            system_prompt=self._system_prompt,
            sep=self._sep,
            user_name=self._user_name,
            assistant_name=self._assistant_name,
            llamacpp_model_config=llamacpp_model_config,
        )
