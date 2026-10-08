# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
from abc import abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from .....types import ChatCompletion, ChatCompletionChunk, PytorchGenerateConfig
from ....utils import cache_clean
from ..core import PytorchChatModel
from ..direct_chat import PytorchDirectChatMixin


class PytorchMultiModalModel(PytorchDirectChatMixin, PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._device = None
        self._processor = None
        self._model = None

    @abstractmethod
    def decide_device(self):
        """
        Update self._device
        """
        pass

    @abstractmethod
    def load_processor(self):
        """
        Load self._processor and self._tokenizer
        """
        pass

    @abstractmethod
    def load_multimodal_model(self):
        """
        Load self._model
        """
        pass

    def load(self):
        self.decide_device()
        reasoning_content = self._pytorch_model_config.pop("reasoning_content")
        enable_thinking = self._pytorch_model_config.pop("enable_thinking", False)
        self.prepare_parse_reasoning_content(
            reasoning_content, enable_thinking=enable_thinking
        )
        self.prepare_parse_tool_calls()
        self.load_processor()
        self.load_multimodal_model()

    @cache_clean
    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        return self.build_direct_chat_result(messages, generate_config)
