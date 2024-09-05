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
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from ....core.scheduler import InferenceRequest
from ....types import ChatCompletion, ChatCompletionChunk, LoRA, PytorchGenerateConfig
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion, generate_completion_chunk, parse_messages
from .core import PytorchChatModel, PytorchModelConfig


class Internlm2PytorchChatModel(PytorchChatModel):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            pytorch_model_config=pytorch_model_config,
            peft_model=peft_model,
        )

    def _get_model_class(self):
        from transformers import AutoModel

        return AutoModel

    def _load_model(self, **kwargs):
        try:
            from transformers import AutoModel, AutoTokenizer
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
            encode_special_tokens=True,
            revision=kwargs["revision"],
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            **kwargs,
        )
        return model, tokenizer

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        model_family = llm_family.model_family or llm_family.model_name
        if model_family in ["internlm2-chat", "internlm2.5-chat"]:
            return True
        return False

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        Overwrite this func for this special model.
        Cannot use the default configuration, which works poorly on this model.
        """
        raw_config = req.inference_kwargs.get("raw_params", {})
        temperature = raw_config.get("temperature", None)
        if temperature is None:
            raw_config["temperature"] = 0.8
        top_p = raw_config.get("top_p", None)
        if top_p is None:
            raw_config["top_p"] = 0.8
        return raw_config

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        kwargs: Dict[str, Any] = {}
        generate_config = generate_config or {}
        temperature = generate_config.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        top_p = generate_config.get("top_p")
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        max_new_tokens = generate_config.get("max_tokens")
        if max_new_tokens is not None:
            kwargs["max_length"] = int(max_new_tokens)

        stream = generate_config.get("stream", False)
        stream_options = generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        prompt, system_prompt, chat_history = parse_messages(messages)
        if chat_history:
            input_history = [
                (chat_history[i]["content"], (chat_history[i + 1]["content"]))
                for i in range(0, len(chat_history), 2)
            ]
        else:
            input_history = []
        if system_prompt:
            kwargs["meta_instruction"] = system_prompt
        if stream:

            def _stream_generator():
                last_chunk_text_length = 0
                chunk_id = "chat-" + str(uuid.uuid1())
                prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
                inputs = self._tokenizer([prompt], return_tensors="pt")
                inputs = inputs.to(self._model.device)
                prompt_tokens = len(inputs["input_ids"][0])
                for chunk_text, _ in self._model.stream_chat(
                    self._tokenizer, prompt, input_history, **kwargs
                ):
                    completion_tokens = completion_tokens + 1
                    total_tokens = prompt_tokens + completion_tokens
                    chunk_text = chunk_text[last_chunk_text_length:]
                    last_chunk_text_length += len(chunk_text)

                    yield generate_completion_chunk(
                        chunk_text,
                        finish_reason=None,
                        chunk_id=chunk_id,
                        model_uid=self.model_uid,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                yield generate_completion_chunk(
                    None,
                    finish_reason="stop",
                    chunk_id=chunk_id,
                    model_uid=self.model_uid,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    has_choice=True,
                    has_content=False,
                )
                if include_usage:
                    yield generate_completion_chunk(
                        None,
                        finish_reason=None,
                        chunk_id=chunk_id,
                        model_uid=self.model_uid,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        has_choice=False,
                    )

            return self._to_chat_completion_chunks(_stream_generator())
        else:
            response, _ = self._model.chat(
                self._tokenizer, prompt, input_history, **kwargs
            )
            return generate_chat_completion(self.model_uid, response)
