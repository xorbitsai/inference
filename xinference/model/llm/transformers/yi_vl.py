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
from typing import Dict, Iterator, List, Optional, Union

import torch

from ....model.utils import select_device
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import (
    _decode_image,
    generate_chat_completion,
    generate_completion_chunk,
    parse_messages,
)
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


class YiVLChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None
        self._image_processor = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        llm_family = model_family.model_family or model_family.model_name
        if "yi-vl" in llm_family:
            return True
        return False

    def load(self):
        from ....thirdparty.llava.mm_utils import load_pretrained_model
        from ....thirdparty.llava.model.constants import key_info

        self._device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(self._device)
        # for multiple GPU, set back to auto to make multiple devices work
        self._device = "auto" if self._device == "cuda" else self._device

        key_info["model_path"] = self.model_path
        # Default device_map is auto, it can loads model to multiple cards.
        # If the device_map is set to cuda, then only 1 card can be used.
        (
            self._tokenizer,
            self._model,
            self._image_processor,
            _,
        ) = load_pretrained_model(self.model_path, device_map=self._device)
        self._apply_lora()

    @staticmethod
    def _message_content_to_yi(content) -> Union[str, tuple]:
        if not isinstance(content, str):
            from ....thirdparty.llava.model.constants import DEFAULT_IMAGE_TOKEN

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
            if DEFAULT_IMAGE_TOKEN not in text:
                text = DEFAULT_IMAGE_TOKEN + "\n" + text
            if len(images) == 0:
                return text
            elif len(images) == 1:
                return text, images[0], "Pad"
            else:
                raise RuntimeError("Only one image per message is supported by Yi VL.")
        return content

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        from transformers import TextIteratorStreamer

        if not generate_config:
            generate_config = {}

        stream = generate_config.get("stream", False)
        stream_options = generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        from ....thirdparty.llava.conversation import conv_templates
        from ....thirdparty.llava.mm_utils import (
            KeywordsStoppingCriteria,
            tokenizer_image_token,
        )
        from ....thirdparty.llava.model.constants import IMAGE_TOKEN_INDEX

        # Convert chat history to llava state
        state = conv_templates["mm_default"].copy()
        prompt, _, chat_history = parse_messages(messages)
        for message in chat_history:
            content = self._message_content_to_yi(message["content"])
            state.append_message(message["role"], content)
        state.append_message(state.roles[0], self._message_content_to_yi(prompt))
        state.append_message(state.roles[1], None)

        prompt = state.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._model.device)
        )

        images = state.get_images(return_pil=True)
        if images:
            image = images[0]
            image_tensor = self._image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]

        stop_str = state.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self._tokenizer, input_ids
        )
        streamer = TextIteratorStreamer(
            self._tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
        )
        top_p = generate_config.get("top_p", 0.7)
        temperature = generate_config.get("temperature", 0.2)
        max_new_tokens = generate_config.get("max_tokens", 512)
        generate_kwargs = {
            "input_ids": input_ids,
            "images": image_tensor.unsqueeze(0)
            .to(dtype=torch.bfloat16)
            .to(self._model.device)
            if images
            else None,
            "streamer": streamer,
            "do_sample": True,
            "top_p": float(top_p),
            "temperature": float(temperature),
            "stopping_criteria": [stopping_criteria],
            "use_cache": True,
            "max_new_tokens": min(int(max_new_tokens), 1536),
        }
        t = Thread(target=self._model.generate, kwargs=generate_kwargs)
        t.start()

        if stream:
            it = self._generate_stream(streamer, stop_str, input_ids, include_usage)
            return self._to_chat_completion_chunks(it)
        else:
            return self._generate(streamer, stop_str)

    def _generate(self, streamer, stop_str) -> ChatCompletion:
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[: -len(stop_str)]
        return generate_chat_completion(self.model_uid, generated_text)

    def _generate_stream(
        self, streamer, stop_str, input_ids, include_usage
    ) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        prompt_tokens = len(input_ids[0])
        for i, new_text in enumerate(streamer):
            if not new_text.endswith(stop_str):
                completion_tokens = i
                total_tokens = prompt_tokens + completion_tokens
                yield generate_completion_chunk(
                    chunk_text=new_text,
                    finish_reason=None,
                    chunk_id=completion_id,
                    model_uid=self.model_uid,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
        yield generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            has_choice=True,
            has_content=False,
        )
        if include_usage:
            yield generate_completion_chunk(
                chunk_text=None,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                has_choice=False,
                has_content=False,
            )
