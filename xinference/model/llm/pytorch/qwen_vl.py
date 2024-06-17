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
import operator
import tempfile
import time
import uuid
from typing import Dict, Iterator, List, Optional, Union

from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


class QwenVLChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if "qwen" in model_family.model_name and "vision" in model_family.model_ability:
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        # for multiple GPU, set back to auto to make multiple devices work
        device = "auto" if device == "cuda" else device

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device,
            trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        ).eval()
        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        )
        self._apply_lora()

    def _message_content_to_qwen(self, content) -> str:
        def _ensure_url(_url):
            if _url.startswith("data:"):
                logging.info("Parse url by base64 decoder.")
                # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                # e.g. f"data:image/jpeg;base64,{base64_image}"
                _type, data = _url.split(";")
                _, ext = _type.split("/")
                data = data[len("base64,") :]
                data = base64.b64decode(data.encode("utf-8"))

                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                    f.write(data)
                logging.info("Dump base64 data to %s", f.name)
                return f.name
            else:
                if len(_url) > 2048:
                    raise Exception(f"Image url is too long, {len(_url)} > 2048.")
                return _url

        if not isinstance(content, str):
            # TODO(codingl2k1): Optimize _ensure_url
            content = [
                (
                    {"image": _ensure_url(c["image_url"]["url"]), "type": "image"}
                    if c.get("type") == "image_url"
                    else c
                )
                for c in content
            ]
            content = sorted(content, key=operator.itemgetter("type"))
            return self._tokenizer.from_list_format(content)
        return content

    def chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        prompt = self._message_content_to_qwen(prompt)
        # Convert openai history to qwen vl history
        qwen_history = []
        query_to_response: List = []
        for h in chat_history or []:
            role = h["role"]
            content = self._message_content_to_qwen(h["content"])
            if len(query_to_response) == 0 and role == "user":
                query_to_response.append(content)
            if len(query_to_response) == 1 and role == "assistant":
                query_to_response.append(content)
            if len(query_to_response) == 2:
                qwen_history.append(query_to_response)
                query_to_response = []

        stream = generate_config.get("stream", False) if generate_config else False
        stream_options = (
            generate_config.pop("stream_options", None) if generate_config else None
        )
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )
        if stream:
            it = self._generate_stream(prompt, qwen_history, include_usage)
            return self._to_chat_completion_chunks(it)
        else:
            c = self._generate(prompt, qwen_history)
            return self._to_chat_completion(c)

    def _generate(self, prompt: str, qwen_history: List) -> Completion:
        response, history = self._model.chat(
            self._tokenizer, query=prompt, history=qwen_history
        )
        c = Completion(
            id=str(uuid.uuid1()),
            object="text_completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[
                CompletionChoice(
                    index=0, text=response, finish_reason="stop", logprobs=None
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
            ),
        )
        return c

    def _generate_stream(
        self, prompt: str, qwen_history: List, include_usage
    ) -> Iterator[CompletionChunk]:
        # response, history = model.chat(tokenizer, message, history=history)
        response_generator = self._model.chat_stream(
            self._tokenizer, query=prompt, history=qwen_history
        )
        completion_id = str(uuid.uuid1())
        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        input_ids = self._tokenizer(prompt, allowed_special="all").input_ids
        prompt_tokens = len(input_ids)
        full_response = ""
        for response in response_generator:
            inc_content = response[len(full_response) :]
            full_response = response
            completion_choice = CompletionChoice(
                text=inc_content, index=0, logprobs=None, finish_reason=None
            )
            completion_chunk = CompletionChunk(
                id=completion_id,
                object="text_completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[completion_choice],
            )
            completion_tokens = completion_tokens + 1
            total_tokens = prompt_tokens + completion_tokens
            completion_usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            completion_chunk["usage"] = completion_usage
            yield completion_chunk

        completion_choice = CompletionChoice(
            text="", index=0, logprobs=None, finish_reason="stop"
        )
        completion_chunk = CompletionChunk(
            id=completion_id,
            object="text_completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[completion_choice],
        )
        completion_usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        completion_chunk["usage"] = completion_usage
        yield completion_chunk
        if include_usage:
            chunk = CompletionChunk(
                id=completion_id,
                object="text_completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[],
            )
            chunk["usage"] = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            yield chunk
