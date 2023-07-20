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

from typing import TYPE_CHECKING, Optional

from .core import PytorchChatModel, PytorchModelConfig

if TYPE_CHECKING:
    from ... import ModelSpec


class StarchatAlphaPytorchChat(PytorchChatModel):
    _system_prompt = "<system>\n"
    _sep = "<|end|>"
    _user_name = "<|user|>"
    _assistant_name = "<|assistant|>"
    _stop = "<|end|>"
    _stop_token_ids = [0, 49155]

    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_spec,
            model_path,
            system_prompt=self._system_prompt,
            sep=self._sep,
            user_name=self._user_name,
            assistant_name=self._assistant_name,
            pytorch_model_config=pytorch_model_config,
        )
        self._use_fast_tokenizer = False


class StarchatBetaPytorchChat(PytorchChatModel):
    _system_prompt = "<system>\n"
    _sep = "<|end|>"
    _user_name = "<|user|>"
    _assistant_name = "<|assistant|>"
    _stop = "<|end|>"
    _stop_token_ids = [0, 49155]

    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_spec,
            model_path,
            system_prompt=self._system_prompt,
            sep=self._sep,
            user_name=self._user_name,
            assistant_name=self._assistant_name,
            pytorch_model_config=pytorch_model_config,
        )
        self._use_fast_tokenizer = False
