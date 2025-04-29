# Copyright 2022-2025 XProbe Inc.
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
import sys
import uuid
from typing import Iterator, List, Optional, Union

from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    CompletionChunk,
    PytorchModelConfig,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion, generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import cache_clean

logger = logging.getLogger(__name__)


class Gemma3TextChatModel(PytorchChatModel):
    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "gemma-3-1b-it".lower() in llm_family.lower():
            return True
        return False


class Gemma3ChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._processor = None

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "gemma-3-it".lower() in llm_family.lower():
            return True
        return False

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        pytorch_model_config = super()._sanitize_model_config(pytorch_model_config)
        assert pytorch_model_config is not None
        pytorch_model_config.setdefault("min_pixels", 256 * 28 * 28)
        pytorch_model_config.setdefault("max_pixels", 1280 * 28 * 28)
        return pytorch_model_config

    def load(self):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device
        # for multiple GPU, set back to auto to make multiple devices work
        device = "auto" if device == "cuda" else device
        min_pixels = self._pytorch_model_config.get("min_pixels")
        max_pixels = self._pytorch_model_config.get("max_pixels")
        kwargs = self.apply_bnb_quantization()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._tokenizer = self._processor.tokenizer
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path, device_map="auto", torch_dtype="bfloat16", **kwargs
        )

    @cache_clean
    def chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        messages = self._transform_messages(messages)

        generate_config = generate_config if generate_config else {}

        stream = generate_config.get("stream", False) if generate_config else False

        if stream:
            it = self._generate_stream(messages, generate_config)
            return self._to_chat_completion_chunks(it)
        else:
            c = self._generate(messages, generate_config)
            return c

    def _generate(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> ChatCompletion:
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)
        input_len = inputs["input_ids"].shape[-1]

        generation = self._model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 1),
        )
        generation = generation[0][input_len:]

        decoded = self._processor.decode(generation, skip_special_tokens=True)
        return generate_chat_completion(self.model_uid, decoded)

    def _generate_stream(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> Iterator[CompletionChunk]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        tokenizer = self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {"streamer": streamer, **inputs}
        error = None

        def model_generate():
            try:
                return self._model.generate(
                    **gen_kwargs,
                    max_new_tokens=config.get("max_tokens", 512),
                    temperature=config.get("temperature", 1),
                )
            except Exception:
                nonlocal error
                error = sys.exc_info()
                streamer.end()
                raise

        thread = Thread(target=model_generate)
        thread.start()

        completion_id = str(uuid.uuid1())
        for new_text in streamer:
            yield generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=-1,
                completion_tokens=-1,
                total_tokens=-1,
                has_choice=True,
                has_content=True,
            )

        if error:
            _, err, tb = error  # type: ignore
            raise err.with_traceback(tb)

        yield generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=-1,
            completion_tokens=-1,
            total_tokens=-1,
            has_choice=True,
            has_content=False,
        )
