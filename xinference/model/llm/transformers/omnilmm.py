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
import json
import logging
import operator
import tempfile
from typing import Dict, Iterator, List, Optional, Tuple, Union

from ....thirdparty.omnilmm.chat import OmniLMMChat, img2base64
from ....types import ChatCompletion, ChatCompletionChunk
from ...utils import select_device
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion, parse_messages
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


class OmniLMMModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        llm_family = model_family.model_family or model_family.model_name
        if "OmniLMM" in llm_family:
            return True
        return False

    def load(self):
        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._model = OmniLMMChat(self.model_path, device_map=device)

    def _message_content_to_OmniLMM(
        self, content
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
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
            images = []
            other_content = []

            for c in content:
                if c.get("type") == "image_url":
                    images.append(
                        {"image": _ensure_url(c["image_url"]["url"]), "type": "image"}
                    )
                else:
                    other_content.append(c)

            images = sorted(images, key=operator.itemgetter("type"))
            other_content = sorted(other_content, key=operator.itemgetter("type"))

            return images, other_content
        return [], [{"type": "text", "text": content}]

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        if generate_config and generate_config.get("stream"):
            raise Exception(
                f"Chat with model {self.model_family.model_name} does not support stream."
            )
        prompt, _, chat_history = parse_messages(messages)
        image_first, prompt = self._message_content_to_OmniLMM(prompt)

        msgs = []
        query_to_response: List[Dict] = []
        image_another = []
        for h in chat_history or []:
            role = h["role"]
            image_tmp, content = self._message_content_to_OmniLMM(h["content"])
            if image_tmp != []:
                image_another = image_tmp
            if len(query_to_response) == 0 and role == "user":
                query_to_response.append(
                    {"role": "user", "content": content[0]["text"]}
                )
            if len(query_to_response) == 1 and role == "assistant":
                query_to_response.append(
                    {"role": "assistant", "content": content[0]["text"]}
                )
            if len(query_to_response) == 2:
                msgs.extend(query_to_response)
                query_to_response = []
        if image_first != []:
            image = image_first
        if image_another != []:
            image = image_another
        im_64 = img2base64(image[0]["image"])
        msgs.append({"role": "user", "content": prompt[0]["text"]})
        input = {"image": im_64, "question": json.dumps(msgs, ensure_ascii=True)}
        answer = self._model.chat(input=input)

        return generate_chat_completion(self.model_uid, answer)
