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
from threading import Thread
from typing import Dict, Iterator, List, Optional, Union

import requests
import torch
from PIL import Image

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from ...utils import select_device
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


class Glm4VModel(PytorchChatModel):
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
        if "glm-4v" in family.lower():
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

        kwargs = {"device_map": self._device}
        quantization = self.quantization

        # referenced from PytorchModel.load
        if quantization != "none":
            if self._device == "cuda" and self._is_linux():
                kwargs["device_map"] = "auto"
                self._device = "auto"
                if quantization == "4-bit":
                    kwargs["load_in_4bit"] = True
                elif quantization == "8-bit":
                    kwargs["load_in_8bit"] = True
                else:
                    raise ValueError(
                        f"Quantization {quantization} is not supported in temporary"
                    )
            else:
                if quantization != "8-bit":
                    raise ValueError(
                        f"Only 8-bit quantization is supported if it is not linux system or cuda device"
                    )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        self._model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._tokenizer = tokenizer

    def _message_content_to_chat(self, content):
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
            # images = []
            # for image_url in image_urls:
            #     images.append(_load_image(image_url))
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
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        from transformers import TextIteratorStreamer

        if not generate_config:
            generate_config = {}

        stream = generate_config.get("stream", False)
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
        msgs.append({"role": "user", "content": content, "image": image})

        inputs = self._tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )  # chat mode
        inputs = inputs.to(self._model.device)

        generate_kwargs = {
            **inputs,
            "eos_token_id": [151329, 151336, 151338],
            "do_sample": True,
            "max_length": generate_config.get("max_tokens", 2048),
            "temperature": generate_config.get("temperature", 0.7),
        }
        stop_str = "<|endoftext|>"

        if stream:
            streamer = TextIteratorStreamer(
                tokenizer=self._tokenizer,
                timeout=60,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs = {
                **generate_kwargs,
                "streamer": streamer,
            }
            t = Thread(target=self._model.generate, kwargs=generate_kwargs)
            t.start()

            it = self.chat_stream(streamer, stop_str)
            return self._to_chat_completion_chunks(it)
        else:
            with torch.no_grad():
                outputs = self._model.generate(**generate_kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = self._tokenizer.decode(outputs[0])
                if response.endswith(stop_str):
                    response = response[: -len(stop_str)]
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
            return self._to_chat_completion(c)

    def chat_stream(self, streamer, stop_str) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        for new_text in streamer:
            if not new_text.endswith(stop_str):
                completion_choice = CompletionChoice(
                    text=new_text, index=0, logprobs=None, finish_reason=None
                )
                chunk = CompletionChunk(
                    id=completion_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=self.model_uid,
                    choices=[completion_choice],
                )
                completion_usage = CompletionUsage(
                    prompt_tokens=-1,
                    completion_tokens=-1,
                    total_tokens=-1,
                )
                chunk["usage"] = completion_usage
                yield chunk

        completion_choice = CompletionChoice(
            text="", index=0, logprobs=None, finish_reason="stop"
        )
        chunk = CompletionChunk(
            id=completion_id,
            object="text_completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[completion_choice],
        )
        completion_usage = CompletionUsage(
            prompt_tokens=-1,
            completion_tokens=-1,
            total_tokens=-1,
        )
        chunk["usage"] = completion_usage
        yield chunk
