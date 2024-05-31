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

from ....model.utils import select_device
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CogVLM2Model(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._torch_type = None
        self._device = None
        self._tokenizer = None
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "cogvlm" in family.lower():
            return True
        return False

    def load(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)
        self._torch_type = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_type,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()

        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    def _message_content_to_cogvlm2(self, content):
        def _load_image(_url):
            if _url.startswith("data:"):
                logging.info("Parse url by base64 decoder.")
                # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                # e.g. f"data:image/jpeg;base64,{base64_image}"
                _type, data = _url.split(";")
                _, ext = _type.split("/")
                data = data[len("base64,") :]
                data = base64.b64decode(data.encode("utf-8"))
                return Image.open(BytesIO(data)).convert("RGB")
            else:
                try:
                    response = requests.get(_url)
                except requests.exceptions.MissingSchema:
                    return Image.open(_url).convert("RGB")
                else:
                    return Image.open(BytesIO(response.content)).convert("RGB")

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
                    fut = executor.submit(_load_image, image_url)
                    image_futures.append(fut)
            images = [fut.result() for fut in image_futures]
            text = " ".join(texts)
            if len(images) == 0:
                return text, None
            elif len(images) == 1:
                return text, images
            else:
                raise RuntimeError(
                    "Only one image per message is supported by CogVLM2."
                )
        return content, None

    def _history_content_to_cogvlm2(
        self, system_prompt: str, chat_history: List[ChatCompletionMessage]
    ):
        def _image_to_piexl_values(image):
            if image.startswith("data:"):
                logging.info("Parse url by base64 decoder.")
                # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                # e.g. f"data:image/jpeg;base64,{base64_image}"
                _type, data = image.split(";")
                _, ext = _type.split("/")
                data = data[len("base64,") :]
                data = base64.b64decode(data.encode("utf-8"))
                return Image.open(BytesIO(data)).convert("RGB")
            else:
                try:
                    response = requests.get(image)
                except requests.exceptions.MissingSchema:
                    return Image.open(image).convert("RGB")
                else:
                    return Image.open(BytesIO(response.content)).convert("RGB")

        query = system_prompt
        history: List[Tuple] = []
        pixel_values = None
        for i in range(0, len(chat_history), 2):
            user = chat_history[i]["content"]
            if isinstance(user, List):
                for content in user:
                    c_type = content.get("type")
                    if c_type == "text":
                        user = content["text"]
                    elif c_type == "image_url" and not pixel_values:
                        pixel_values = _image_to_piexl_values(
                            content["image_url"]["url"]
                        )
            assistant = chat_history[i + 1]["content"]
            query = query + f" USER: {user} ASSISTANT:"
            history.append((query, assistant))
            query = query + f" {assistant}"
        return query, history, [pixel_values]

    def chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        system_prompt = system_prompt if system_prompt else ""
        if generate_config and generate_config.get("stream"):
            raise Exception(
                f"Chat with model {self.model_family.model_name} does not support stream."
            )

        sanitized_config = {
            "pad_token_id": 128002,
            "max_new_tokens": generate_config.get("max_tokens", 512)
            if generate_config
            else 512,
        }

        content, image = self._message_content_to_cogvlm2(prompt)

        history = []
        query = ""
        history_image = None
        if chat_history:
            query, history, history_image = self._history_content_to_cogvlm2(
                system_prompt, chat_history
            )

        if image and history_image:
            history = []
            query = system_prompt + f" USER: {content} ASSISTANT:"
        else:
            image = image if image else history_image
            query = query + f" USER: {content} ASSISTANT:"

        input_by_model = self._model.build_conversation_input_ids(
            self._tokenizer,
            query=query,
            history=history,
            images=image,
            template_version="chat",
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self._device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self._device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self._device),
            "images": [
                [input_by_model["images"][0].to(self._device).to(self._torch_type)]
            ]
            if image is not None
            else None,
        }
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **sanitized_config)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self._tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]

        chunk = Completion(
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
        return self._to_chat_completion(chunk)
