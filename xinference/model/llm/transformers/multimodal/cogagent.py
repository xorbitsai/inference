# Copyright 2022-2026 XProbe Inc.
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
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch

from .....model.utils import select_device
from ...core import chat_context_var
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ...utils import _decode_image, parse_messages
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("ChatGLMForConditionalGeneration")
class CogAgentChatModel(PytorchMultiModalModel):
    COGAGENT_ARCHITECTURES = {"ChatGLMForConditionalGeneration"}

    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self._platform: Optional[Literal["Mac", "WIN", "Mobile"]] = "Mac"  # type: ignore
        self._format: Optional[  # type: ignore
            Literal[
                "(Answer in Action-Operation-Sensitive format.)",
                "(Answer in Status-Plan-Action-Operation format.)",
                "(Answer in Status-Action-Operation-Sensitive format.)",
                "(Answer in Status-Action-Operation format.)",
                "(Answer in Action-Operation format.)",
            ]
        ] = "(Answer in Action-Operation-Sensitive format.)"

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not model_family.has_architecture(*cls.COGAGENT_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not CogAgent",
            )
        if "vision" not in model_family.model_ability:
            return False, "CogAgent transformer requires vision ability"
        return True

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

    def load_processor(self):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def load_multimodal_model(self):
        from transformers import AutoModelForCausalLM

        kwargs = self.apply_quantization_config()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self._device,
            **kwargs,
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

    def _get_query_and_history(
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

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        prompt, _, chat_history = parse_messages(messages)

        query, image = self._get_query_and_history(prompt, chat_history)

        full_context_kwargs = {
            "return_tensors": "pt",
            "return_dict": True,
        }
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        chat_context_var.set(chat_template_kwargs)
        full_context_kwargs.update(chat_template_kwargs)
        assert self.model_family.chat_template is not None
        inputs = self.get_full_context(
            [{"role": "user", "image": image, "content": query}],
            self.model_family.chat_template,
            self._tokenizer,
            tokenize=True,
            **full_context_kwargs,
        )
        inputs.to(self._model.device)
        return inputs

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        generate_config = {} if generate_config is None else generate_config
        self._platform = generate_config.pop("platform", self._platform)
        self._format = generate_config.pop("format", self._format)
        return {
            "max_length": generate_config.get("max_tokens") or 512,
            "top_k": generate_config.get("top_k", 1),
            "do_sample": True,
        }

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        config = self.build_generate_kwargs(generate_config)
        inputs = self.build_inputs_from_messages(messages, generate_config)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {**inputs, **config, "streamer": streamer}

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer, len(inputs.input_ids[0])
