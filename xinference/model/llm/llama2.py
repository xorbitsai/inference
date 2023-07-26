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

from .. import ModelSpec
from .core import LlamaCppChatModel, LlamaCppModelConfig


class Llama2ChatGgml(LlamaCppChatModel):
    _system_prompt = (
        "System: You are a helpful, respectful and honest assistant. Always answer as helpfully as"
        " possible, while being safe.  Your answers should not include any harmful, unethical,"
        " racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses "
        "are socially unbiased and positive in nature. If a question does not make any sense, or "
        "is not factually coherent, explain why instead of answering something not correct. If you "
        "don't know the answer to a question, please don't share false information."
    )
    _sep = "\n"
    _user_name = "User"
    _assistant_name = "Assistant"
    _stop = "\n"

    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_spec,
            model_path,
            system_prompt=self._system_prompt,
            sep=self._sep,
            user_name=self._user_name,
            assistant_name=self._assistant_name,
            llamacpp_model_config=llamacpp_model_config,
        )
