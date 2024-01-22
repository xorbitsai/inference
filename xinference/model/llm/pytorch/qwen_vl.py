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
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
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
        if "qwen" in model_family.model_name:
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)

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
                {"image": _ensure_url(c["image_url"]["url"]), "type": "image"}
                if c.get("type") == "image_url"
                else c
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
        if generate_config and generate_config.get("stream"):
            raise Exception(
                f"Chat with model {self.model_family.model_name} does not support stream."
            )
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
        response, history = self._model.chat(
            self._tokenizer, query=prompt, history=qwen_history
        )
        return ChatCompletion(
            id="chat" + str(uuid.uuid1()),
            object="chat.completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": response},
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
            ),
        )
