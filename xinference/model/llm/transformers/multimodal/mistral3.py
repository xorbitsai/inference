# Copyright 2022-2025 XProbe Inc.
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

import importlib.util
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple
from io import BytesIO
from PIL import Image

from .....core.model import register_batching_multimodal_models
from .....core.scheduler import InferenceRequest
from .....device_utils import is_npu_available
from .....model.utils import select_device
from .....types import PytorchModelConfig
from ...llm_family import LLMFamilyV1, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_batching_multimodal_models("mistral-small-3.2-instruct")
@register_transformer
@register_non_default_model("mistral-small-3.2-instruct")
class MistralAWQMultimodalModel(PytorchMultiModalModel):
    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        pytorch_model_config = super()._sanitize_model_config(pytorch_model_config)
        assert pytorch_model_config is not None
        return pytorch_model_config

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "mistral-small-3.2-instruct" in llm_family.lower():
            return True
        return False

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device

    def load_processor(self):
        from transformers import AutoProcessor
        from transformers import AutoTokenizer

        
        min_pixels = self._pytorch_model_config.get("min_pixels")
        max_pixels = self._pytorch_model_config.get("max_pixels")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        # 加载 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def load_multimodal_model(self):
        from transformers import Mistral3ForConditionalGeneration

        kwargs = self.apply_bnb_quantization()
        device = "auto" if self._device == "cuda" else self._device
        
        self._model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_path, 
            device_map="cuda",
            torch_dtype="bfloat16",
            **kwargs
        ).eval()

    def build_inputs_from_messages(
        self, messages: List[Dict], generate_config: Dict
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

    def build_generate_kwargs(self, generate_config: Dict) -> Dict[str, Any]:
        return dict(
            max_new_tokens=generate_config.get("max_tokens", 512),
            temperature=generate_config.get("temperature", 1),
        )

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from threading import Thread
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

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        from transformers import GenerationConfig

        gen_config = GenerationConfig.from_pretrained(self.model_path).to_dict()
        raw_config = req.inference_kwargs.get("raw_params", {})
        gen_config.update(raw_config)
        return gen_config

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        return messages

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        import torch

        texts = []
        for p in prompts:
            if hasattr(self._tokenizer, "apply_chat_template"):
                text = self._tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": p.get("content", "")}
                    ], tokenize=False, add_generation_prompt=True
                )
            else:
                text = p.get("content", "")
            texts.append(text)

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self._device)

        for r, ids, attn_mask in zip(req_list, inputs.input_ids, inputs.attention_mask):
            r.prompt_tokens = ids.tolist()
            real_len = torch.sum(attn_mask).item()
            r.padding_len = attn_mask.numel() - real_len
            r.extra_kwargs["attention_mask_seq_len"] = real_len

        batch_size, seq_len = inputs.input_ids.shape
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)
        return {**inputs, "position_ids": position_ids}