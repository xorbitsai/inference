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
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .....model.utils import select_device
from .....types import PytorchModelConfig
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("Gemma3ForConditionalGeneration")
class Gemma3ChatModel(PytorchMultiModalModel):
    GEMMA3_MM_ARCHITECTURES = {"Gemma3ForConditionalGeneration"}

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Gemma-3 multimodal transformer supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if not model_family.has_architecture(*cls.GEMMA3_MM_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not Gemma-3-it",
            )
        return True

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        pytorch_model_config = super()._sanitize_model_config(pytorch_model_config)
        assert pytorch_model_config is not None
        pytorch_model_config.setdefault("min_pixels", 256 * 28 * 28)
        pytorch_model_config.setdefault("max_pixels", 1280 * 28 * 28)
        return pytorch_model_config

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device

    def load_processor(self):
        from transformers import AutoProcessor

        min_pixels = self._pytorch_model_config.get("min_pixels")
        max_pixels = self._pytorch_model_config.get("max_pixels")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        from transformers import Gemma3ForConditionalGeneration

        kwargs = self.apply_quantization_config()
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path, device_map="auto", torch_dtype="bfloat16", **kwargs
        )

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        messages = self._transform_messages(messages)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)
        return inputs

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        return dict(
            max_new_tokens=generate_config.get("max_tokens") or 512,
            temperature=generate_config.get("temperature", 1),
        )

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        inputs = self.build_inputs_from_messages(messages, generate_config)
        configs = self.build_generate_kwargs(generate_config)

        tokenizer = self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {"streamer": streamer, **inputs, **configs}
        t = Thread(target=self._model.generate, kwargs=gen_kwargs)
        t.start()
        return streamer, len(inputs.input_ids[0])
