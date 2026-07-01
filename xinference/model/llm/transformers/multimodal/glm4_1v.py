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
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Dict, Iterator, List, Tuple, Union

import torch

from .....model.utils import select_device
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ...utils import _decode_image
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model(
    "Glm4vForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration",
)
class Glm4_1VModel(PytorchMultiModalModel):
    GLM4V_ARCHITECTURES = {
        "Glm4vForConditionalGeneration",
        "Glm4vMoeForConditionalGeneration",
    }

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not model_family.has_architecture(*cls.GLM4V_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not GLM-4.1V/4.5V",
            )
        if "vision" not in model_family.model_ability:
            return False, "GLM-4.1V transformer requires vision ability"
        return True

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

    def load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        from transformers import Glm4vForConditionalGeneration

        kwargs = {"device_map": "auto"}
        kwargs = self.apply_quantization_config(kwargs)

        model = Glm4vForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        self._model = model.eval()
        self._device = self._model.device

    @staticmethod
    def _get_processed_msgs(messages: List[Dict]) -> List[Dict]:
        res = []
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
                        fut = executor.submit(_decode_image, image_url)
                        image_futures.append(fut)
                images = [fut.result() for fut in image_futures]
                assert len(images) <= 1
                text = " ".join(texts)
                if images:
                    content = [
                        {"type": "image", "image": images[0]},
                        {"type": "text", "text": text},
                    ]
                    res.append({"role": role, "content": content})
                else:
                    res.append(
                        {"role": role, "content": {"type": "text", "text": text}}
                    )
        return res

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        msgs = self._get_processed_msgs(messages)
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        tools = generate_config.get("tools", None) if generate_config else None
        if tools:
            chat_template_kwargs["tools"] = tools
        inputs = self._processor.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            **chat_template_kwargs,
        )  # chat mode
        inputs = inputs.to(self._model.device)
        return inputs

    def get_stop_strs(self) -> List[str]:
        return ["<|endoftext|>"]

    def get_builtin_stop_token_ids(self) -> Tuple:
        from transformers import AutoConfig

        return tuple(AutoConfig.from_pretrained(self.model_path).eos_token_id)

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        return dict(
            do_sample=True,
            top_p=generate_config.get("top_p", 1e-5),
            repetition_penalty=generate_config.get("repetition_penalty", 1.1),
            top_k=generate_config.get("top_k", 2),
            max_new_tokens=generate_config.get("max_tokens") or 512,
        )

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        generate_kwargs = self.build_generate_kwargs(generate_config)
        inputs = self.build_inputs_from_messages(messages, generate_config)
        streamer = TextIteratorStreamer(
            tokenizer=self._tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=False,
        )
        kwargs = {
            **inputs,
            **generate_kwargs,
            "streamer": streamer,
        }
        logger.debug("Generate with kwargs: %s", generate_kwargs)
        t = Thread(target=self._model.generate, kwargs=kwargs)
        t.start()
        return streamer, len(inputs.input_ids[0])
