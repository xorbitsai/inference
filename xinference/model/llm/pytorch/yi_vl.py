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
from typing import Dict, Iterator, List, Optional, Union

import requests
import torch
from PIL import Image

from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    CompletionUsage,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
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
        if "yi" in model_family.model_name:
            return True
        return False

    def load(self):
        from ....thirdparty.llava.mm_utils import load_pretrained_model
        from ....thirdparty.llava.model.constants import key_info

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)

        key_info["model_path"] = self.model_path
        (
            self._tokenizer,
            self._model,
            self._image_processor,
            _,
        ) = load_pretrained_model(self.model_path, device_map=device)

    def _message_content_to_yi(self, content) -> Union[str, tuple]:
        def _load_image(_url):
            if _url.startswith("data:"):
                logging.info("Parse url by base64 decoder.")
                # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                # e.g. f"data:image/jpeg;base64,{base64_image}"
                _type, data = _url.split(";")
                _, ext = _type.split("/")
                data = data[len("base64,") :]
                data = base64.b64decode(data.encode("utf-8"))

                return Image.open(BytesIO(data))
            else:
                try:
                    response = requests.get(_url)
                except requests.exceptions.MissingSchema:
                    return Image.open(_url)
                else:
                    return Image.open(BytesIO(response.content))

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
                    fut = executor.submit(_load_image, image_url)
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
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        if generate_config and generate_config.get("stream"):
            raise Exception(
                f"Chat with model {self.model_family.model_name} does not support stream."
            )
        from ....thirdparty.llava.conversation import conv_templates
        from ....thirdparty.llava.mm_utils import (
            KeywordsStoppingCriteria,
            tokenizer_image_token,
        )
        from ....thirdparty.llava.model.constants import IMAGE_TOKEN_INDEX

        # Convert chat history to llava state
        state = conv_templates["mm_default"].copy()
        for message in chat_history or []:
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
            .cuda()
        )

        images = state.get_images(return_pil=True)
        image = images[0]

        image_tensor = self._image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        stop_str = state.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self._tokenizer, input_ids
        )
        top_p = generate_config.get("top_p", 0.7)
        temperature = generate_config.get("temperature", 0.2)
        max_new_tokens = generate_config.get("max_tokens", 512)
        generate_kwargs = {
            "input_ids": input_ids,
            "images": image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            "do_sample": True,
            "top_p": float(top_p),
            "temperature": float(temperature),
            "stopping_criteria": [stopping_criteria],
            "use_cache": True,
            "max_new_tokens": min(int(max_new_tokens), 1536),
        }
        r = self._model.generate(**generate_kwargs)
        print("fffffffffffffff", r)
        return ChatCompletion(
            id="chat" + str(uuid.uuid1()),
            object="chat.completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": r},
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
            ),
        )
