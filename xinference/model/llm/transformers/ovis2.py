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
from typing import Dict, Iterator, List, Optional, Union

import torch

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    CompletionChunk,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import cache_clean
from ..utils import (
    generate_chat_completion,
    generate_completion_chunk,
)

logger = logging.getLogger(__name__)


class Ovis2ChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._processor = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "ovis2".lower() in llm_family.lower():
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM

        # load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True,
        ).cuda()
        self._text_tokenizer = self._model.get_text_tokenizer()
        self._visual_tokenizer = self._model.get_visual_tokenizer()

    @cache_clean
    def chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        messages = self._transform_messages(messages)

        generate_config = generate_config if generate_config else {}

        stream = generate_config.get("stream", False) if generate_config else False

        if stream:
            raise NotImplementedError("Stream is not supported for Ovis2 model.")
            it = self._generate_stream(messages, generate_config)
            return self._to_chat_completion_chunks(it)
        else:
            c = self._generate(messages, generate_config)
            return c

    def _generate(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> ChatCompletion:
        from qwen_vl_utils import process_vision_info

        messages_ovis = self.parse_messages_ovis(messages)
        max_partition = None
        prompt = messages_ovis[-1]["value"]

        # Preparation for inference
        image_inputs, video_inputs = process_vision_info(messages)

        image_inputs = image_inputs if image_inputs else []

        if image_inputs and len(image_inputs) > 0:
            if len(image_inputs) == 1:
                max_partition = 9
                prompt = f"<image>\n{prompt}"
            else:
                max_partition = len(image_inputs) + 1
                prompt = (
                    "\n".join(
                        [f"Image {i+1}: <image>" for i in range(len(image_inputs))]
                    )
                    + "\n"
                    + prompt
                )
        elif video_inputs and len(video_inputs) > 0:
            max_partition = 1
            prompt = "\n".join(["<image>"] * len(video_inputs)) + "\n" + prompt
        else:
            max_partition = 0
            prompt = prompt

        messages_ovis[-1]["value"] = prompt

        logger.info(
            f"===[Ovis2][img cnt: {len(image_inputs)}] prompt: {prompt}, ovis_msgs: {messages_ovis}"
        )
        # format conversation
        prompt, input_ids, pixel_values = self._model.preprocess_inputs(
            messages_ovis, image_inputs, max_partition=max_partition
        )
        logger.info(f"===[Ovis2][preprocess_inputs] prompt: {prompt}")

        attention_mask = torch.ne(input_ids, self._text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self._model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self._model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self._visual_tokenizer.dtype, device=self._visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=config.get("max_tokens", 1024),
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=config.get("temperature", None),
                repetition_penalty=None,
                eos_token_id=self._model.generation_config.eos_token_id,
                pad_token_id=self._text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self._model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = self._text_tokenizer.decode(output_ids, skip_special_tokens=True)
            logger.info(f"===[Ovis2] output: {output}")
        return generate_chat_completion(self.model_uid, output)

    def parse_messages_ovis(self, messages: List[Dict]) -> List[Dict]:
        """
        Some older models still follow the old way of parameter passing.
        This function helps to parse out the needed information from OpenAI-compatible `messages`.
        """
        logger.info(f"===[Ovis2] parse_messages_ovis: {messages}")
        ovis_msgs = []
        for mess in messages:
            contents = mess["content"]
            role = mess["role"]
            if role == "user":
                role = "human"
            elif role == "assistant":
                role = "gpt"
            elif role == "system":
                role = "system"

            for content in contents:
                if content["type"] == "text":
                    ovis_msgs.append({"from": role, "value": content["text"]})

        return ovis_msgs
