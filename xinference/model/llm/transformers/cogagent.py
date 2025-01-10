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
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Literal, Optional, Union

import torch

from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    CogagentGenerateConfig,
    CompletionChunk,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import (
    _decode_image,
    generate_chat_completion,
    generate_completion_chunk,
    parse_messages,
)
from .core import PytorchChatModel
from .utils import cache_clean

logger = logging.getLogger(__name__)


class CogAgentChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._torch_type = None
        self._device = None
        self._tokenizer = None
        self._model = None
        self._platform: Literal["Mac", "WIN", "Mobile"] | None = "Mac"
        self._format: Literal[
            "(Answer in Action-Operation-Sensitive format.)",
            "(Answer in Status-Plan-Action-Operation format.)",
            "(Answer in Status-Action-Operation-Sensitive format.)",
            "(Answer in Status-Action-Operation format.)",
            "(Answer in Action-Operation format.)",
        ] | None = "(Answer in Action-Operation-Sensitive format.)"

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "cogagent" in family.lower():
            return True
        return False

    def load(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.quantization == "4-bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif self.quantization == "8-bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self._device,
            quantization_config=quantization_config,
        ).eval()

    def _message_content_to_cogagent(self, content):
        assert isinstance(content, list)
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
            raise RuntimeError(
                "CogAgent requires image input to perform GUI Agent tasks. Pure text-based interaction cannot execute such tasks."
            )
        elif len(images) == 1:
            return text, images[-1]
        else:
            logger.warning(
                "There are multiple images in the prompt, CogAgent will automatically use the most recently provided image as the input."
            )
            return text, images[-1]

    def _history_content_to_cogagent(self, chat_history: List[Dict]):
        grounded_pattern = r"Grounded Operation:\s*(.*)"
        action_pattern = r"Action:\s*(.*)"

        def extract_operations(_content: str):
            """extract grounded operation and action operation"""
            _history_step = []
            _history_action = []

            matches_history = re.search(grounded_pattern, _content)
            matches_actions = re.search(action_pattern, _content)

            if matches_history:
                grounded_operation = matches_history.group(1)
                _history_step.append(grounded_operation)
            if matches_actions:
                action_operation = matches_actions.group(1)
                _history_action.append(action_operation)

            return _history_step, _history_action

        history_step = []
        history_action = []

        for i in range(0, len(chat_history) - 1, 2):
            content = chat_history[i + 1].get("content")
            if isinstance(content, str):  # 如果内容是字符串
                steps, actions = extract_operations(content)
                history_step.extend(steps)
                history_action.extend(actions)

            elif isinstance(content, list):  # 如果内容是列表
                for c in content:
                    c_content = c.get("content")
                    if isinstance(c_content, str):  # 确保是字符串类型
                        steps, actions = extract_operations(c_content)
                        history_step.extend(steps)
                        history_action.extend(actions)

        return history_step, history_action

    def get_query_and_history(
        self,
        prompt: Union[str, List[Dict]],
        chat_history: Optional[List[Dict]] = None,
    ):
        task, image = self._message_content_to_cogagent(prompt)

        history_step, history_action = [], []

        if chat_history:
            history_step, history_action = self._history_content_to_cogagent(
                chat_history
            )

        # Verify history lengths match
        if len(history_step) != len(history_action):
            raise ValueError("Mismatch in lengths of history_step and history_action.")

        # Format history steps for output
        history_str = "\nHistory steps: "
        for index, (step, action) in enumerate(zip(history_step, history_action)):
            history_str += f"\n{index}. {step}\t{action}"

        # Compose the query with task, platform, and selected format instructions
        query = f"Task: {task}{history_str}\n{self._platform}{self._format}"
        logger.info(f"query:{query}")
        return query, image

    @cache_clean
    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[CogagentGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        if generate_config is not None:
            self._platform = generate_config.pop("platform", self._platform)
            self._format = generate_config.pop("format", self._format)

        sanitize_generate_config = self._sanitize_generate_config(generate_config)
        stream = sanitize_generate_config.get("stream")
        sanitized_config = {
            "max_length": sanitize_generate_config.get("max_tokens", 512),
            "top_k": sanitize_generate_config.get("top_k", 1),
            "do_sample": True,
        }
        prompt, _, chat_history = parse_messages(messages)

        query, image = self.get_query_and_history(prompt, chat_history)

        full_context_kwargs = {
            "return_tensors": "pt",
            "return_dict": True,
        }
        assert self.model_family.chat_template is not None
        inputs = self.get_full_context(
            [{"role": "user", "image": image, "content": query}],
            self.model_family.chat_template,
            self._tokenizer,
            tokenize=True,
            **full_context_kwargs,
        )
        inputs.to(self._model.device)

        if stream:
            it = self._streaming_chat_response(inputs, sanitized_config)
            return self._to_chat_completion_chunks(it)
        else:
            # Generate response
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **sanitized_config)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            return generate_chat_completion(self.model_uid, response)

    def _streaming_chat_response(
        self, inputs: Dict, config: Dict
    ) -> Iterator[CompletionChunk]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {**inputs, **config}

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
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
