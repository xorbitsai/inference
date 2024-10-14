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

import torch

from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchModel

logger = logging.getLogger(__name__)


class DeepSeekV2PytorchModel(PytorchModel):
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
        if "deepseek-v2" not in model_family:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    # def generate(
    #     self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    # ) -> Union[Completion, Iterator[CompletionChunk]]:
    #     input_tensor = self._tokenizer(prompt, return_tensors="pt")
    #     generate_config = self._sanitize_generate_config(generate_config)
    #     default_generate_config = self._model.generation_config
    #     generate_kwargs = {
    #         "input_ids": input_tensor["input_ids"].cuda(),
    #         "attention_mask": input_tensor["attention_mask"].cuda(),
    #         "temperature": float(
    #             generate_config.get("temperature", default_generate_config.temperature)
    #         ),
    #         "repetition_penalty": float(generate_config.get("repetition_penalty", 1.0)),
    #         "top_p": float(generate_config.get("top_p", default_generate_config.top_p)),
    #         "top_k": int(generate_config.get("top_k", -1)),
    #         "max_new_tokens": generate_config.get("max_tokens", 512),
    #         "bos_token_id": default_generate_config.bos_token_id,
    #         "do_sample": default_generate_config.do_sample,
    #         "eos_token_id": default_generate_config.eos_token_id,
    #     }
    #
    #     stream = generate_config.get("stream", False)
    #     if stream:
    #         return self._generate_stream(generate_kwargs, input_tensor)
    #     else:
    #         return self._generate(generate_kwargs, input_tensor)
    #
    # def _generate(self, generate_kwargs, input_ids) -> Completion:
    #     prompt_tokens = len(input_ids[0])
    #     logger.info(f"generate_kwargs:{generate_kwargs}")
    #     generation_output = self._model.generate(**generate_kwargs)
    #     completion_tokens = len(generation_output[0])
    #     response = self._tokenizer.decode(
    #         generation_output[0], skip_special_tokens=True
    #     )
    #     return generate_completion(
    #         self.model_uid,
    #         response,
    #         prompt_tokens=prompt_tokens,
    #         completion_tokens=completion_tokens,
    #         total_tokens=prompt_tokens + completion_tokens,
    #     )
    #
    # def _generate_stream(self, generate_kwargs, input_ids):
    #     from threading import Thread
    #
    #     from transformers import TextIteratorStreamer
    #
    #     # Initialize the streamer
    #     streamer = TextIteratorStreamer(
    #         self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10
    #     )
    #     # Define the generation configuration
    #     generate_kwargs["streamer"] = streamer
    #     # Start the model chat in a separate thread
    #     thread = Thread(
    #         target=self._model.generate,
    #         kwargs=generate_kwargs,
    #     )
    #     thread.start()
    #
    #     completion_id = str(uuid.uuid1())
    #     prompt_tokens = len(input_ids[0])
    #     total_tokens, completion_tokens = 0, 0
    #     # Loop through the streamer to get the new text as it is generated
    #     for i, new_text in enumerate(streamer):
    #         completion_tokens = i
    #         total_tokens = prompt_tokens + completion_tokens
    #         yield generate_completion_chunk(
    #             chunk_text=new_text,
    #             finish_reason=None,
    #             chunk_id=completion_id,
    #             model_uid=self.model_uid,
    #             prompt_tokens=prompt_tokens,
    #             completion_tokens=completion_tokens,
    #             total_tokens=total_tokens,
    #         )
    #     yield generate_completion_chunk(
    #         chunk_text=None,
    #         finish_reason="stop",
    #         chunk_id=completion_id,
    #         model_uid=self.model_uid,
    #         prompt_tokens=prompt_tokens,
    #         completion_tokens=completion_tokens,
    #         total_tokens=total_tokens,
    #         has_choice=True,
    #         has_content=False,
    #     )


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
        if "deepseek-v2" not in model_family:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True
