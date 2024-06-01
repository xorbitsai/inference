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
import base64
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, Iterator, List, Optional, Tuple, Union

import requests
import torch
from PIL import Image

from ...utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionUsage,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)

class MiniCPMV25Model(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = None
        self._tokenizer = None
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "MiniCPM-Llama3-V-2_5".lower() in family.lower():
            return True
        return False

    def load(self, **kwargs):
        from transformers import AutoModel, AutoTokenizer
        from transformers.generation import GenerationConfig

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)
        self._device = "auto" if self._device == "cuda" else self._device

        if 'int4' in self.model_path:
            if device == 'mps':
                print('Error: running int4 model with bitsandbytes on Mac is not supported right now.')
                exit()
            model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = model.eval()
        self._tokenizer = tokenizer

        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    def chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        return None
