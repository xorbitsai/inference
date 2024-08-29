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
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Union

import torch

from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ...utils import select_device
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import (
    _decode_image,
    generate_chat_completion,
    generate_completion_chunk,
    parse_messages,
)
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

    def _get_model_class(self):
        from transformers import AutoModel

        return AutoModel

    def load(self, **kwargs):
        from transformers import AutoModel, AutoTokenizer
        from transformers.generation import GenerationConfig

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)
        self._device = "auto" if self._device == "cuda" else self._device

        if "int4" in self.model_path and device == "mps":
            logger.error(
                "Error: running int4 model with bitsandbytes on Mac is not supported right now."
            )
            exit()

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer()
            return

        if "int4" in self.model_path:
            model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self._device,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = model.eval()
        self._tokenizer = tokenizer

        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._save_tensorizer()

    def _message_content_to_chat(self, content):
        if not isinstance(content, str):
            texts = []
            image_urls = []
            for c in content:
                c_type = c.get("type")
                if c_type == "text":
                    texts.append(c["text"])
                elif c_type == "image_url":
                    image_urls.append(c["image_url"]["url"])
            image_futures = []
            with ThreadPoolExecutor() as executor:
                for image_url in image_urls:
                    fut = executor.submit(_decode_image, image_url)
                    image_futures.append(fut)
            images = [fut.result() for fut in image_futures]
            text = " ".join(texts)
            if len(images) == 0:
                return text, []
            elif len(images) == 1:
                return text, images
            else:
                raise RuntimeError("Only one image per message is supported")
        return content, []

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        stream = generate_config.get("stream", False) if generate_config else False
        prompt, _, chat_history = parse_messages(messages)
        content, images_chat = self._message_content_to_chat(prompt)

        msgs = []
        query_to_response: List[Dict] = []
        images_history = []
        for h in chat_history or []:
            role = h["role"]
            content_h, images_tmp = self._message_content_to_chat(h["content"])
            if images_tmp != []:
                images_history = images_tmp
            if len(query_to_response) == 0 and role == "user":
                query_to_response.append({"role": "user", "content": content_h})
            if len(query_to_response) == 1 and role == "assistant":
                query_to_response.append({"role": "assistant", "content": content_h})
            if len(query_to_response) == 2:
                msgs.extend(query_to_response)
                query_to_response = []
        image = None
        if len(images_chat) > 0:
            image = images_chat[0]
        elif len(images_history) > 0:
            image = images_history[0]
        msgs.append({"role": "user", "content": content})

        chat = self._model.chat(
            image=image,
            msgs=json.dumps(msgs, ensure_ascii=True),
            tokenizer=self._tokenizer,
            sampling=True,
            **generate_config
        )
        if stream:
            it = self.chat_stream(chat)
            return self._to_chat_completion_chunks(it)
        else:
            return generate_chat_completion(self.model_uid, chat)

    def chat_stream(self, chat) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        for new_text in chat:
            yield generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=-1,
                completion_tokens=-1,
                total_tokens=-1,
            )

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
