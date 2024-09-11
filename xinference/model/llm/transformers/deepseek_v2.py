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
from typing import Dict, Iterator, List, Optional, Union

import torch

from ....types import ChatCompletion, ChatCompletionChunk, PytorchGenerateConfig
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion
from .core import PytorchChatModel

logger = logging.getLogger(__name__)


class DeepSeekV2PytorchChatModel(PytorchChatModel):
    def _load_model(self, **kwargs):
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                GenerationConfig,
            )
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=kwargs["trust_remote_code"],
        )
        logger.info(f"kwargs:{kwargs}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model, tokenizer

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        model_family = llm_family.model_family or llm_family.model_name
        if "deepseek" not in model_family:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    # def generate(
    #         self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    # ) -> Union[Completion, Iterator[CompletionChunk]]:
    #
    #     self._tokenizer.apply_chat_template()
    #     outputs = self._model.generate(input_tensor.to(model.device), max_new_tokens=100)
    #
    #     result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    #     print(result)

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        input_tensor = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        outputs = self._model.generate(
            input_tensor.to(self._model.device), max_new_tokens=100
        )

        result = self._tokenizer.decode(
            outputs[0][input_tensor.shape[1] :], skip_special_tokens=True
        )
        logger.info(result)
        # tools = generate_config.pop("tools", []) if generate_config else None
        # model_family = self.model_family.model_family or self.model_family.model_name
        # full_context_kwargs = {}
        # if tools and model_family in QWEN_TOOL_CALL_FAMILY:
        #     full_context_kwargs["tools"] = tools
        # assert self.model_family.chat_template is not None
        # full_prompt = self.get_full_context(
        #     messages,
        #     self.model_family.chat_template,
        #     tokenizer=self._tokenizer,
        #     **full_context_kwargs,
        # )
        #
        # generate_config = self._sanitize_generate_config(generate_config)
        #
        # stream = generate_config.get("stream", False)
        # if stream:
        #     it = self.generate(full_prompt, generate_config)
        #     assert isinstance(it, Iterator)
        #     return self._to_chat_completion_chunks(it)
        # else:
        #     c = self.generate(full_prompt, generate_config)
        #     assert not isinstance(c, Iterator)
        #     if tools:
        #         return self._tool_calls_completion(self.model_family, self.model_uid, c)
        #     return self._to_chat_completion(c)
        return generate_chat_completion(
            self.model_uid,
            result,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
