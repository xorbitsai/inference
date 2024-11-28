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
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ...utils import select_device
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import (
    _decode_image_without_rgb,
    generate_chat_completion,
    generate_completion_chunk,
)
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import cache_clean

logger = logging.getLogger(__name__)


class GlmEdgeVModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = None
        self._tokenizer = None
        self._model = None
        self._processor = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "glm-edge-v" in family.lower():
            return True
        return False

    def load(self):
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

        kwargs = {"device_map": self._device}
        quantization = self.quantization

        # referenced from PytorchModel.load
        if quantization != "none":
            if self._device == "cuda" and self._is_linux():
                kwargs["device_map"] = "auto"
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

        processor = AutoImageProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._processor = processor

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self._model = model

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._tokenizer = tokenizer

    @staticmethod
    def _get_processed_msgs(
        messages: List[Dict],
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        res = []
        img = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, str):
                res.append({"role": role, "content": content})
            else:
                texts = []
                image_urls = []
                for c in content:
                    c_type = c.get("type")
                    if c_type == "text":
                        texts.append(c["text"])
                    else:
                        assert (
                            c_type == "image_url"
                        ), "Please follow the image input of the OpenAI API."
                        image_urls.append(c["image_url"]["url"])
                if len(image_urls) > 1:
                    raise RuntimeError("Only one image per message is supported")
                image_futures = []
                with ThreadPoolExecutor() as executor:
                    for image_url in image_urls:
                        fut = executor.submit(_decode_image_without_rgb, image_url)
                        image_futures.append(fut)
                images = [fut.result() for fut in image_futures]
                assert len(images) <= 1
                text = " ".join(texts)
                img.extend(images)
                if images:
                    res.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text},
                            ],
                        }
                    )
                else:
                    res.append({"role": role, "content": text})
        return res, img

    @cache_clean
    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        from transformers import TextIteratorStreamer

        if not generate_config:
            generate_config = {}

        stream = generate_config.get("stream", False)
        msgs, imgs = self._get_processed_msgs(messages)

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
        }
        if len(imgs) > 0:
            generate_kwargs["pixel_values"] = torch.tensor(
                self._processor(imgs[-1]).pixel_values
            ).to(self._model.device)
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
                outputs = outputs[0][len(inputs["input_ids"][0]) :]
                response = self._tokenizer.decode(outputs)
                if response.endswith(stop_str):
                    response = response[: -len(stop_str)]
            return generate_chat_completion(self.model_uid, response)

    def chat_stream(self, streamer, stop_str) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        for new_text in streamer:
            if not new_text.endswith(stop_str):
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
